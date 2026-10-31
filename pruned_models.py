import torch
import torch.nn.functional as F
from torch import nn

from nbfnet.models import NBFNet

SCORER_SELECTORS = {"learned", "astar"}


# ─────────────────────────────────────────────────────────────────
# Layer: sparse (physical) or dense (reference) execution
# Both take the SAME inputs: selected (query, edge) pairs b_idx / e_idx
# and a flat (B*V,) per-node gate (or None). Both return (new_h, recv(B,V)).
# ─────────────────────────────────────────────────────────────────
class SparsePrunedConv(nn.Module):
    """Distmult + sum NBFNet layer with physical sparse execution."""

    def __init__(self, in_dim, out_dim, num_relation, query_dim, layer_norm=False,
                 dependent=True, sparse_exec=True, short_cut=False):
        super().__init__()
        self.num_relation, self.dependent = num_relation, dependent
        self.sparse_exec, self.short_cut = sparse_exec, short_cut
        self.linear = nn.Linear(in_dim * 2, out_dim)
        self.norm = nn.LayerNorm(out_dim) if layer_norm else None
        if dependent:
            self.relation_linear = nn.Linear(query_dim, num_relation * in_dim)
        else:
            self.relation = nn.Embedding(num_relation, in_dim)

    def _relations(self, query, B, d):
        if self.dependent:
            return self.relation_linear(query).view(B, self.num_relation, d)
        return self.relation.weight.unsqueeze(0).expand(B, -1, -1)

    def _post(self, out):
        if self.norm is not None:
            out = self.norm(out)
        return F.relu(out)

    def forward(self, h, query, boundary, edge_index, edge_type, node_gate, b_idx, e_idx):
        if self.sparse_exec:
            return self._forward_sparse(h, query, boundary, edge_index, edge_type, node_gate, b_idx, e_idx)
        return self._forward_dense(h, query, boundary, edge_index, edge_type, node_gate, b_idx, e_idx)

    def _forward_sparse(self, h, query, boundary, edge_index, edge_type, node_gate, b_idx, e_idx):
        B, V, d = h.shape
        src, dst = edge_index
        recv = torch.zeros(B * V, dtype=torch.bool, device=h.device)
        if b_idx.numel() == 0:
            return h, recv.view(B, V)

        rel = self._relations(query, B, d)
        s_, t_, r_ = src[e_idx], dst[e_idx], edge_type[e_idx]
        h_flat = h.reshape(B * V, d)
        src_key = b_idx * V + s_
        msg = h_flat[src_key] * rel[b_idx, r_]                         # (K, d)
        if node_gate is not None:
            msg = msg * node_gate[src_key].unsqueeze(-1)

        # Sort-free dedup: O(K + B*V), no sort on K keys every layer.
        # mark[] scatter gives a sorted uniq (ascending), so downstream is unchanged.
        key = b_idx * V + t_
        mark = torch.zeros(B * V, dtype=torch.bool, device=h.device)
        mark[key] = True
        uniq = mark.nonzero(as_tuple=True)[0]
        idmap = torch.empty(B * V, dtype=torch.long, device=h.device)
        idmap[uniq] = torch.arange(len(uniq), device=h.device)
        inv = idmap[key]

        agg = h.new_zeros(len(uniq), d).index_add(0, inv, msg) + boundary.reshape(B * V, d)[uniq]
        out = self._post(self.linear(torch.cat([h_flat[uniq], agg], dim=-1)))   # linear ONLY on receivers
        if self.short_cut:
            out = out + h_flat[uniq]
        recv[uniq] = True
        # In-place update at eval avoids cloning the full B*V*d state tensor.
        # Out-of-place is kept during training so autograd works.
        if not torch.is_grad_enabled():
            h_flat = h_flat.clone()
            h_flat.index_copy_(0, uniq, out)
            new_h = h_flat.view(B, V, d)
        else:
            new_h = h_flat.index_copy(0, uniq, out).view(B, V, d)
        return new_h, recv.view(B, V)

    def _forward_dense(self, h, query, boundary, edge_index, edge_type, node_gate, b_idx, e_idx):
        """Reference path (original masked behaviour). Used for equivalence tests / dense baseline."""
        B, V, d = h.shape
        src, dst = edge_index
        esel = torch.zeros(B, src.numel(), dtype=torch.bool, device=h.device)
        esel[b_idx, e_idx] = True
        gate = esel.float()
        if node_gate is not None:
            gate = gate * node_gate.view(B, V)[:, src]
        rel = self._relations(query, B, d)
        msg = h[:, src] * rel[:, edge_type] * gate.unsqueeze(-1)
        agg = torch.zeros_like(h).index_add(1, dst, msg) + boundary
        out = self._post(self.linear(torch.cat([h, agg], dim=-1)))
        if self.short_cut:
            out = out + h
        recv = torch.zeros(B, V, device=h.device).index_add(1, dst, esel.float()) > 0
        return torch.where(recv.unsqueeze(-1), out, h), recv


MaskedConv = SparsePrunedConv  # Backward-compatible alias


# ─────────────────────────────────────────────────────────────────
# Selection / graph helpers (all cost ~ selected set, not B*E)
# ─────────────────────────────────────────────────────────────────
def select_nodes(score, cand, outdeg, budget, force_first):
    """Top nodes by score until cumulative out-degree exceeds budget. Returns bool (B, V).

    Optimisation: if every query's total reachable out-degree already fits inside
    the budget, selection can't prune anything, so we skip the O(V log V) argsort
    and return the full candidate set immediately.
    """
    # Per-query total out-degree of reached (candidate) nodes.
    # Shape: (B,).  If all rows fit the budget, no pruning is possible.
    total_deg = (cand.float() * outdeg).sum(dim=1)   # (B,)
    if (total_deg <= budget).all():
        # Nothing to prune: return a fresh copy so that callers computing
        # rest = cand & ~sel still get a non-aliased tensor -- important
        # for the exploration branch in _select_nodes_budget during training.
        return cand.clone()

    s = score.masked_fill(~cand, float("-inf"))
    order = s.argsort(dim=1, descending=True)
    cand_sorted = cand.gather(1, order)
    cum = outdeg[order].cumsum(dim=1)
    keep = (cum <= budget) & cand_sorted
    if force_first:
        keep[:, 0] = cand_sorted[:, 0]
    return torch.zeros_like(cand).scatter(1, order, keep)


def build_csr(index, V):
    """CSR over `index` (src for out-edges, dst for in-edges). Returns rowptr (V+1), perm (E), deg (V)."""
    perm = torch.argsort(index, stable=True)
    deg = torch.bincount(index, minlength=V)
    rowptr = F.pad(deg.cumsum(0), (1, 0))
    return rowptr, perm, deg


def expand_edges(b_idx, n_idx, rowptr, perm):
    """For selected (query b, node n) pairs, list all their CSR edges WITHOUT a (B, E) tensor.
    Returns (b_idx_edges, edge_ids)."""
    deg = rowptr[n_idx + 1] - rowptr[n_idx]
    rep = torch.repeat_interleave(torch.arange(len(deg), device=deg.device), deg)
    start = deg.cumsum(0) - deg
    off = torch.arange(len(rep), device=deg.device) - start[rep]
    e = perm[rowptr[n_idx][rep] + off]
    return b_idx[rep], e


def segment_topm(b, score, B, M):
    """Per-query top-M among expanded edges. Returns indices into b/score."""
    o1 = score.argsort(descending=True)
    o2 = b[o1].argsort(stable=True)
    order = o1[o2]                                    # sorted by query, then by score desc
    bs = b[order]
    counts = torch.bincount(b, minlength=B)
    starts = counts.cumsum(0) - counts
    rank = torch.arange(len(b), device=b.device) - starts[bs]
    return order[rank < M]


# ─────────────────────────────────────────────────────────────────
# Distance decay target helper – exponential, rate=0.5 (FIXED)
# ─────────────────────────────────────────────────────────────────
_DECAY_RATE = 0.5   # y = exp(-0.5 * d), proven best config


def _compute_distance_target(dist):
    """y = exp(-0.5 * d); unreachable (inf) -> 0."""
    y = torch.exp(-_DECAY_RATE * dist)
    return y.masked_fill(torch.isinf(dist), 0.0)


# ─────────────────────────────────────────────────────────────────
# MLP builder – GELU, no LayerNorm, no Dropout (fixed best config)
# ─────────────────────────────────────────────────────────────────
def _build_mlp(in_dim, hidden_dim, out_dim, depth):
    """Scorer MLP: [Linear -> GELU] * depth, then Linear(hidden, out). depth in {1, 2}."""
    if depth not in (1, 2):
        raise ValueError("scorer_depth must be 1 or 2, got %s" % depth)

    def _hidden_block(in_f, out_f):
        return [nn.Linear(in_f, out_f), nn.GELU()]

    layers = _hidden_block(in_dim, hidden_dim)
    if depth == 2:
        layers += _hidden_block(hidden_dim, hidden_dim)
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


class PrunedNBFNet(NBFNet):
    def __init__(self, selector="learned", rho=0.5, explore=0.2, aux_weight=0.5,
                 modified_loss=False, scorer_dim=64, scorer_depth=2, sparse_exec=True, **kwargs):
        super().__init__(**kwargs)
        assert selector in ("full", "random", "learned", "astar")
        if modified_loss and selector not in SCORER_SELECTORS:
            raise ValueError(
                "modified_loss=True requires selector in %s, got '%s'" % (SCORER_SELECTORS, selector))
        assert kwargs.get("message_func", "distmult") == "distmult"
        assert len(set(self.dims)) == 1, "all hidden dims must equal input_dim"
        self.selector, self.rho, self.explore = selector, rho, explore
        self.aux_weight, self.modified_loss = aux_weight, modified_loss
        self.scorer_dim, self.scorer_depth = scorer_dim, scorer_depth
        self.sparse_exec = sparse_exec
        d = self.dims[0]
        self.layers = nn.ModuleList(
            SparsePrunedConv(self.dims[i], self.dims[i + 1], self.num_relation, d,
                       kwargs.get("layer_norm", False), kwargs.get("dependent", True),
                       sparse_exec=sparse_exec, short_cut=self.short_cut)
            for i in range(len(self.dims) - 1)
        )
        # "learned" scorer (shared across layers): input = [h || query]
        # only built when used, so random/full baselines carry no scorer params
        self.scorer = _build_mlp(2 * d, scorer_dim, 1, scorer_depth) if selector == "learned" else None
        # "astar" scorer (A*Net-style, separate parameters)
        if selector == "astar":
            self.astar_query_proj = nn.Linear(2 * d, d)
            self.astar_mlp = _build_mlp(d, scorer_dim, 1, scorer_depth)
        self.aux_loss = torch.zeros(())
        self.last_reached, self._t_pos = None, None
        self._csr_cache = None
        self._in_csr_cache = None
        self.reset_stats()

    # ---------------------------------------------------------------- cost & stats
    def parameter_breakdown(self):
        """Count total parameters, base GNN parameters, and pruner parameters."""
        total = sum(p.numel() for p in self.parameters())
        pruner = 0
        if self.scorer is not None:
            pruner += sum(p.numel() for p in self.scorer.parameters())
        if hasattr(self, "astar_mlp") and self.astar_mlp is not None:
            pruner += sum(p.numel() for p in self.astar_mlp.parameters())
        if hasattr(self, "astar_query_proj") and self.astar_query_proj is not None:
            pruner += sum(p.numel() for p in self.astar_query_proj.parameters())
        base = total - pruner
        pruner_pct = (pruner / total * 100.0) if total > 0 else 0.0
        return {
            "total": total,
            "base": base,
            "pruner": pruner,
            "pruner_pct": pruner_pct,
        }

    def reset_stats(self):
        self._acc_active = 0            # python int: messages actually computed
        self._acc_full = 0.0            # tensor/float: messages sparse-"full" would compute
        self._acc_possible = 0          # B*E*L (dense upper bound)
        num_layers = len(self.layers) if hasattr(self, "layers") else 0
        self._layer_nodes_kept = [0] * num_layers
        self._layer_edges_kept = [0] * num_layers
        self._layer_nodes_total = 0     # sum of B * V across batches
        self._layer_edges_total = 0     # sum of B * E across batches
        self._last_nodes_frac = []
        self._last_edges_frac = []

    def efficiency_stats(self):
        """Call only at log time (does one sync)."""
        full = float(self._acc_full)
        act = float(self._acc_active)
        return {
            "active_messages": act,
            "full_equiv_messages": full,
            "possible_messages": float(self._acc_possible),
            "sparsity_vs_full": 1.0 - act / max(1.0, full),          # what pruning itself saves
            "sparsity_vs_dense": 1.0 - act / max(1.0, float(self._acc_possible)),
        }

    @property
    def stats(self):
        eff = self.efficiency_stats()
        nodes_frac = (
            [k / max(1, self._layer_nodes_total) for k in self._layer_nodes_kept]
            if getattr(self, "_layer_nodes_total", 0) > 0 else list(getattr(self, "_last_nodes_frac", []))
        )
        edges_frac = (
            [k / max(1, self._layer_edges_total) for k in self._layer_edges_kept]
            if getattr(self, "_layer_edges_total", 0) > 0 else list(getattr(self, "_last_edges_frac", []))
        )
        return {
            **eff,
            "physical_sparsity": eff["sparsity_vs_dense"],
            "nodes_frac": nodes_frac,
            "edges_frac": edges_frac,
        }

    # ---------------------------------------------------------------- CSR (cached in eval / reverse)
    def _out_csr(self, edge_index, V):
        src = edge_index[0]
        if not self.training:
            key = (src.data_ptr(), src.numel(), V)
            if self._csr_cache is not None and self._csr_cache[0] == key:
                return self._csr_cache[1]
            csr = build_csr(src, V)
            self._csr_cache = (key, csr)
            return csr
        return build_csr(src, V)

    def _in_csr(self, dst, V):
        key = (dst.data_ptr(), dst.numel(), V)
        if self._in_csr_cache is not None and self._in_csr_cache[0] == key:
            return self._in_csr_cache[1]
        csr = build_csr(dst, V)
        self._in_csr_cache = (key, csr)
        return csr

    # ---------------------------------------------------------------- aux target: sparse reverse BFS
    @staticmethod
    def _dist_to_target(edge_index, V, t_pos, hops, rowptr_in=None, perm_in=None):
        """Hop distance to target for every (query, node); returns flat (B*V,) float, inf if > hops."""
        src, dst = edge_index
        B, dev = len(t_pos), t_pos.device
        if rowptr_in is None or perm_in is None:
            rowptr_in, perm_in, _ = build_csr(dst, V)                      # incoming edges per node
        rowptr, perm = rowptr_in, perm_in
        dist = torch.full((B * V,), float("inf"), device=dev)
        seen = torch.zeros(B * V, dtype=torch.bool, device=dev)
        fb = torch.arange(B, device=dev)
        fn = t_pos
        key0 = fb * V + fn
        dist[key0] = 0
        seen[key0] = True
        for k in range(1, hops + 1):
            if fb.numel() == 0:
                break
            b, e = expand_edges(fb, fn, rowptr, perm)
            key = b * V + src[e]
            key = key[~seen[key]]
            if key.numel() == 0:
                break
            # Sort-free dedup using mark tensor avoids expensive torch.unique on GPU
            mark = torch.zeros(B * V, dtype=torch.bool, device=dev)
            mark[key] = True
            key = mark.nonzero(as_tuple=True)[0]
            seen[key] = True
            dist[key] = k
            fb, fn = key // V, key % V
        return dist

    # ---------------------------------------------------------------- scorers (reached nodes only)
    def _astar_score(self, h_nodes, q_nodes):
        proj = self.astar_query_proj(torch.cat([h_nodes, q_nodes], dim=-1))
        return self.astar_mlp(h_nodes * proj).squeeze(-1)

    @staticmethod
    def _scatter(s_r, ridx, BV):
        return s_r.new_zeros(BV).index_copy(0, ridx, s_r)       # differentiable wrt s_r

    # ---------------------------------------------------------------- selectors
    def _select_nodes_budget(self, score, reached, outdeg, E):
        """Budgeted node selection used by learned / random.
        Uses degree-aware priority density to prevent hub nodes from monopolizing the budget."""
        B, V = reached.shape
        budget = self.rho * E
        explore = self.training and self.selector == "learned" and self.explore > 0
        main_budget = budget * (1 - self.explore) if explore else budget

        # Degree-aware priority density: sigmoid(score) / sqrt(outdeg)
        # Prevents 1 or 2 hub nodes from instantly consuming the entire edge budget.
        prio = torch.sigmoid(score) / outdeg.clamp(min=1).pow(0.5)
        sel = select_nodes(prio, reached, outdeg, main_budget, True)
        if explore:
            rest = reached & ~sel
            sel = sel | select_nodes(torch.rand(B, V, device=score.device), rest, outdeg,
                                     budget * self.explore, False)
        return sel

    def _select_astar(self, h_flat, query, reached, ridx, s_full, src, dst, rowptr, perm, E, B, V):
        """A*Net-style: top-K reached nodes, then per-query top-M edges ranked by heuristic score.
        Uses source node heuristic priority to avoid unreached destination zero-vector evaluation."""
        K = min(V, max(1, int(self.rho * V)))
        M = max(1, int(self.rho * E))
        node_score = s_full.detach().view(B, V).masked_fill(~reached, float("-inf"))
        _, topk_idx = node_score.topk(K, dim=1)
        sel = torch.zeros(B, V, dtype=torch.bool, device=h_flat.device).scatter_(1, topk_idx, True) & reached

        b_n, n_n = sel.nonzero(as_tuple=True)
        b_idx, e_idx = expand_edges(b_n, n_n, rowptr, perm)
        # Only prune edges if they actually exceed the per-query budget M
        if b_idx.numel() > M:
            # Score each expanded edge by its source node's heuristic score
            src_edge_score = s_full[b_idx * V + src[e_idx]]
            keep = segment_topm(b_idx, src_edge_score, B, M)
            b_idx, e_idx = b_idx[keep], e_idx[keep]
        return sel, b_idx, e_idx, len(b_n)

    # ---------------------------------------------------------------- core
    def bellmanford(self, data, h_index, r_index, separate_grad=False):
        B, V, dev = len(r_index), data.num_nodes, h_index.device
        src, dst = data.edge_index
        E = src.numel()
        d = self.dims[0]
        ar = torch.arange(B, device=dev)
        query = self.query(r_index)
        boundary = torch.zeros(B, V, d, device=dev)
        boundary[ar, h_index] = query
        reached = torch.zeros(B, V, dtype=torch.bool, device=dev)
        reached[ar, h_index] = True

        rowptr, perm, deg = self._out_csr(data.edge_index, V)
        outdeg = deg.float()

        use_aux = (self.training and self.modified_loss
                   and self.selector in SCORER_SELECTORS and self._t_pos is not None)
        dist_target = None
        if use_aux:
            rowptr_in, perm_in, _ = self._in_csr(dst, V)
            raw_dist = self._dist_to_target(data.edge_index, V, self._t_pos, len(self.layers),
                                            rowptr_in=rowptr_in, perm_in=perm_in)
            dist_target = _compute_distance_target(raw_dist)      # (B*V,)

        aux, full_equiv, active = [], [], 0
        current_nodes_frac, current_edges_frac = [], []
        h, hiddens = boundary, []
        for l_idx, layer in enumerate(self.layers):
            h_flat = h.reshape(B * V, d)
            full_equiv.append((reached.float() * outdeg).sum())   # what sparse-full would send this layer
            node_gate = None

            if self.selector == "full":
                sel = reached
                b_n, n_n = sel.nonzero(as_tuple=True)
                b_idx, e_idx = expand_edges(b_n, n_n, rowptr, perm)
                num_nodes_l = len(b_n)

            elif self.selector == "random":
                score = torch.rand(B, V, device=dev)              # no scorer at all
                sel = select_nodes(score, reached, outdeg, self.rho * E, True)
                b_n, n_n = sel.nonzero(as_tuple=True)
                b_idx, e_idx = expand_edges(b_n, n_n, rowptr, perm)
                num_nodes_l = len(b_n)

            elif self.selector == "learned":
                ridx = reached.reshape(-1).nonzero(as_tuple=True)[0]
                # Detach backbone representations when using aux loss to prevent gradient conflict
                h_in = h_flat[ridx].detach() if self.modified_loss else h_flat[ridx]
                q_in = query[ridx // V].detach() if self.modified_loss else query[ridx // V]
                s_r = self.scorer(torch.cat([h_in, q_in], dim=-1)).squeeze(-1)
                s_full = self._scatter(s_r, ridx, B * V)
                sel = self._select_nodes_budget(s_full.detach().view(B, V), reached, outdeg, E)
                # When using distance aux loss, do not squash messages with negative logits
                node_gate = None if self.modified_loss else torch.sigmoid(s_full)
                b_n, n_n = sel.nonzero(as_tuple=True)
                b_idx, e_idx = expand_edges(b_n, n_n, rowptr, perm)
                num_nodes_l = len(b_n)
                if use_aux:
                    aux.append(F.binary_cross_entropy_with_logits(s_r, dist_target[ridx]))

            else:  # astar
                ridx = reached.reshape(-1).nonzero(as_tuple=True)[0]
                h_in = h_flat[ridx].detach() if self.modified_loss else h_flat[ridx]
                q_in = query[ridx // V].detach() if self.modified_loss else query[ridx // V]
                s_r = self._astar_score(h_in, q_in)
                s_full = self._scatter(s_r, ridx, B * V)
                sel, b_idx, e_idx, num_nodes_l = self._select_astar(h_flat, query, reached, ridx, s_full,
                                                                    src, dst, rowptr, perm, E, B, V)
                node_gate = None if self.modified_loss else torch.sigmoid(s_full)
                if use_aux:
                    aux.append(F.binary_cross_entropy_with_logits(s_r, dist_target[ridx]))

            h, recv = layer(h, query, boundary, data.edge_index, data.edge_type, node_gate, b_idx, e_idx)
            reached = reached | recv
            hiddens.append(h)
            num_edges_l = b_idx.numel()
            active += num_edges_l
            current_nodes_frac.append(num_nodes_l / max(1, B * V))
            current_edges_frac.append(num_edges_l / max(1, B * E))
            if hasattr(self, "_layer_nodes_kept") and l_idx < len(self._layer_nodes_kept):
                self._layer_nodes_kept[l_idx] += num_nodes_l
                self._layer_edges_kept[l_idx] += num_edges_l

        self.aux_loss = ((torch.stack(aux).mean() * self.aux_weight)
                         if (aux and self.aux_weight > 0) else torch.zeros((), device=dev))
        self.last_reached = reached
        self._acc_active += active
        self._acc_full = self._acc_full + torch.stack(full_equiv).sum().detach()
        self._acc_possible += B * E * len(self.layers)
        self._layer_nodes_total += B * V
        self._layer_edges_total += B * E
        self._last_nodes_frac = current_nodes_frac
        self._last_edges_frac = current_edges_frac

        node_query = query.unsqueeze(1).expand(-1, V, -1)
        out = torch.cat((hiddens if self.concat_hidden else hiddens[-1:]) + [node_query], dim=-1)
        return {"node_feature": out, "edge_weights": [None] * len(self.layers)}

    def forward(self, data, batch):
        h_index, t_index, r_index = batch.unbind(-1)
        if self.training:
            data = self.remove_easy_edges(data, h_index, t_index, r_index)
        shape = h_index.shape
        h_index, t_index, r_index = self.negative_sample_to_tail(h_index, t_index, r_index)
        assert (h_index[:, [0]] == h_index).all()
        assert (r_index[:, [0]] == r_index).all()

        # column 0 is the positive target after negative_sample_to_tail
        self._t_pos = t_index[:, 0] if self.training else None

        output = self.bellmanford(data, h_index[:, 0], r_index[:, 0])
        feature = output["node_feature"]
        index = t_index.unsqueeze(-1).expand(-1, -1, feature.shape[-1])
        feature = feature.gather(1, index)
        score = self.mlp(feature).squeeze(-1)
        return score.view(shape)