
import torch
import torch.nn.functional as F
from torch import nn

from nbfnet.models import NBFNet

SCORER_SELECTORS = {"learned", "astar"}


class MaskedConv(nn.Module):
    """Distmult + sum NBFNet layer with a per-query edge gate (B, E)."""

    def __init__(self, in_dim, out_dim, num_relation, query_dim, layer_norm=False, dependent=True):
        super().__init__()
        self.num_relation, self.dependent = num_relation, dependent
        self.linear = nn.Linear(in_dim * 2, out_dim)
        self.norm = nn.LayerNorm(out_dim) if layer_norm else None
        if dependent:
            self.relation_linear = nn.Linear(query_dim, num_relation * in_dim)
        else:
            self.relation = nn.Embedding(num_relation, in_dim)

    def forward(self, h, query, boundary, edge_index, edge_type, gate):
        B, V, d = h.shape
        src, dst = edge_index
        if self.dependent:
            rel = self.relation_linear(query).view(B, self.num_relation, d)
        else:
            rel = self.relation.weight.unsqueeze(0).expand(B, -1, -1)
        msg = h[:, src] * rel[:, edge_type] * gate.unsqueeze(-1)          # (B, E, d)
        agg = torch.zeros_like(h).index_add(1, dst, msg) + boundary       # sum + boundary
        out = self.linear(torch.cat([h, agg], dim=-1))
        if self.norm is not None:
            out = self.norm(out)
        return F.relu(out)


def select_nodes(score, cand, outdeg, budget, force_first):
    """Top nodes by score until cumulative out-degree exceeds budget. Returns bool (B, V)."""
    s = score.masked_fill(~cand, float("-inf"))
    order = s.argsort(dim=1, descending=True)
    cand_sorted = cand.gather(1, order)
    cum = outdeg[order].cumsum(dim=1)
    keep = (cum <= budget) & cand_sorted
    if force_first:
        keep[:, 0] = cand_sorted[:, 0]
    return torch.zeros_like(cand).scatter(1, order, keep)


# ─────────────────────────────────────────────────────────────────
# Distance decay target helper – exponential, rate=0.5 (FIXED)
# ─────────────────────────────────────────────────────────────────

_DECAY_RATE = 0.5   # y = exp(-0.5 * d), proven best config


def _compute_distance_target(dist):
    """
    Convert integer BFS distances to soft supervision targets in [0, 1].

    Formula (fixed): y = exp(-0.5 * d)
        - Exponential decay with rate 0.5 gives sharp, local supervision.
        - Unreachable nodes (dist == inf) are mapped to 0.

    Args:
        dist : (B, V) float tensor of BFS distances from the target entity

    Returns:
        y    : (B, V) float tensor of target values in [0, 1]
    """
    y = torch.exp(-_DECAY_RATE * dist)
    y = y.masked_fill(torch.isinf(dist), 0.0)
    return y


# ─────────────────────────────────────────────────────────────────
# MLP builder – GELU, no LayerNorm, no Dropout (fixed best config)
# ─────────────────────────────────────────────────────────────────
def _build_mlp(in_dim, hidden_dim, out_dim, depth):
    """
    Build a scorer MLP with GELU activation, no LayerNorm, no Dropout.

    Experimental findings (scorer_comparison.docx):
        - GELU is the best activation (Val MRR 0.3609, Test MRR 0.3160).
        - LayerNorm degrades test MRR (0.2507 vs 0.2837 baseline).
        - Dropout 0.1 causes over-regularisation (Test MRR 0.2632).
      => Structural regularisation suppresses fine-grained distance signals.

    Architecture per hidden layer:
        Linear -> GELU

    Final layer:
        Linear(hidden_dim, out_dim)

    Args:
        in_dim     : input feature size
        hidden_dim : width of every hidden layer
        out_dim    : output size (usually 1 for scoring)
        depth      : number of hidden layers (1 or 2)
    """
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
                 modified_loss=False, scorer_dim=64, scorer_depth=2, **kwargs):
        super().__init__(**kwargs)
        assert selector in ("full", "random", "learned", "astar")
        if modified_loss and selector not in SCORER_SELECTORS:
            raise ValueError(
                "modified_loss=True requires selector in %s, got '%s'" % (SCORER_SELECTORS, selector))
        assert kwargs.get("message_func", "distmult") == "distmult"
        assert kwargs.get("aggregate_func", "sum") == "sum"
        assert len(set(self.dims)) == 1, "all hidden dims must equal input_dim"
        self.selector, self.rho, self.explore = selector, rho, explore
        self.aux_weight, self.modified_loss = aux_weight, modified_loss
        self.scorer_dim, self.scorer_depth = scorer_dim, scorer_depth
        d = self.dims[0]
        self.layers = nn.ModuleList(
            MaskedConv(self.dims[i], self.dims[i + 1], kwargs["num_relation"], d,
                       kwargs.get("layer_norm", False), kwargs.get("dependent", True))
            for i in range(len(self.dims) - 1)
        )
        # ── Scorer MLPs (GELU, no LayerNorm, no Dropout) ────────────
        # "learned" scorer (shared across layers): input = [h || query]
        self.scorer = _build_mlp(2 * d, scorer_dim, 1, scorer_depth)
        # "astar" scorer (A*Net-style, separate parameters)
        self.astar_query_proj = nn.Linear(2 * d, d)
        self.astar_mlp = _build_mlp(d, scorer_dim, 1, scorer_depth)
        self.aux_loss = torch.zeros(())
        self.stats, self.last_reached, self._t_pos = {}, None, None

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _dist_to_target(edge_index, V, t_pos, hops):
        src, dst = edge_index
        B, dev = len(t_pos), t_pos.device
        ar = torch.arange(B, device=dev)
        dist = torch.full((B, V), float("inf"), device=dev)
        frontier = torch.zeros(B, V, dtype=torch.bool, device=dev)
        dist[ar, t_pos] = 0
        frontier[ar, t_pos] = True
        seen = frontier.clone()
        for k in range(1, hops + 1):
            nxt = torch.zeros(B, V, device=dev).index_add(1, src, frontier[:, dst].float()) > 0
            nxt &= ~seen
            dist[nxt] = k
            seen |= nxt
            frontier = nxt
        return dist

    def _select(self, h, query, reached, outdeg, E):
        B, V, _ = h.shape
        s = self.scorer(torch.cat([h, query.unsqueeze(1).expand(-1, V, -1)], dim=-1)).squeeze(-1)
        budget = self.rho * E
        if self.selector == "learned":
            score = s.detach()
        elif self.selector == "random":
            score = torch.rand(B, V, device=h.device)
        else:
            score = torch.rand(B, V, device=h.device)
        explore = self.training and self.selector == "learned" and self.explore > 0
        main_budget = budget * (1 - self.explore) if explore else budget
        sel = select_nodes(score, reached, outdeg, main_budget, True)
        if explore:
            rest = reached & ~sel
            sel = sel | select_nodes(torch.rand(B, V, device=h.device), rest, outdeg,
                                     budget * self.explore, False)
        gate = torch.sigmoid(s) if self.selector == "learned" else torch.ones_like(s)
        return sel, gate, s

    def _select_astar(self, h, query, reached, outdeg, src, dst, E, V):
        """A*Net-style selection: node scoring, top-K nodes among reached, top-M edges by dst score."""
        B = h.shape[0]
        d = h.shape[2]
        q_exp = query.unsqueeze(1).expand(-1, V, -1)             # (B, V, d)
        proj = self.astar_query_proj(torch.cat([h, q_exp], dim=-1))  # (B, V, d)
        distmult_msg = h * proj                                       # (B, V, d)
        s = self.astar_mlp(distmult_msg).squeeze(-1)                  # (B, V)

        score_det = s.detach()
        K = max(1, int(self.rho * V))
        M = max(1, int(self.rho * E))

        # --- node selection: top-K among reached nodes ---
        node_score = score_det.masked_fill(~reached, float("-inf"))
        n_reached = reached.sum(dim=1)                                # (B,)
        actual_k = min(K, int(n_reached.max().item()))
        if actual_k < V:
            _, topk_idx = node_score.topk(actual_k, dim=1)
            sel = torch.zeros(B, V, dtype=torch.bool, device=h.device)
            sel.scatter_(1, topk_idx, True)
            sel &= reached  # safety: mask out unreached that slipped in via padding
        else:
            sel = reached.clone()

        # --- edge selection: among edges from kept nodes, keep top-M by dst score ---
        src_kept = sel[:, src]                                        # (B, E)
        dst_score = score_det[:, dst]                                 # (B, E)
        edge_score = dst_score.masked_fill(~src_kept, float("-inf"))
        n_cand = src_kept.sum(dim=1)                                  # (B,)
        actual_m = min(M, int(n_cand.max().item()))
        if actual_m < E:
            _, edge_topk = edge_score.topk(actual_m, dim=1)
            esel = torch.zeros(B, E, dtype=torch.bool, device=h.device)
            esel.scatter_(1, edge_topk, True)
            esel &= src_kept  # safety
        else:
            esel = src_kept.clone()

        gate = esel.float() * torch.sigmoid(s)[:, src]
        return sel, gate, esel, s

    # ---------------------------------------------------------------- core
    def bellmanford(self, data, h_index, r_index, separate_grad=False):
        B, V, dev = len(r_index), data.num_nodes, h_index.device
        src, dst = data.edge_index
        E = src.numel()
        ar = torch.arange(B, device=dev)
        query = self.query(r_index)
        boundary = torch.zeros(B, V, self.dims[0], device=dev)
        boundary[ar, h_index] = query
        reached = torch.zeros(B, V, dtype=torch.bool, device=dev)
        reached[ar, h_index] = True
        outdeg = torch.bincount(src, minlength=V).float()

        use_aux = (self.training and self.modified_loss
                   and self.selector in SCORER_SELECTORS and self._t_pos is not None)

        # ── Distance decay targets: exp(-0.5 * d), computed once per pass ──
        if use_aux:
            raw_dist = self._dist_to_target(data.edge_index, V, self._t_pos, len(self.layers))
            dist_target = _compute_distance_target(raw_dist)
        else:
            dist_target = None
        aux, edges_frac, nodes_frac = [], [], []

        h, hiddens = boundary, []
        for layer in self.layers:
            if self.selector == "full":
                sel, node_gate = reached, torch.ones(B, V, device=dev)
                esel = sel[:, src]
                gate = esel.float() * node_gate[:, src]
            elif self.selector == "astar":
                sel, gate, esel, s = self._select_astar(h, query, reached, outdeg, src, dst, E, V)
                if use_aux:
                    aux.append(F.binary_cross_entropy_with_logits(s[reached], dist_target[reached]))
            else:
                sel, node_gate, s = self._select(h, query, reached, outdeg, E)
                if use_aux:
                    aux.append(F.binary_cross_entropy_with_logits(s[reached], dist_target[reached]))
                esel = sel[:, src]
                gate = esel.float() * node_gate[:, src]
            recv = torch.zeros(B, V, device=dev).index_add(1, dst, esel.float()) > 0
            new = layer(h, query, boundary, data.edge_index, data.edge_type, gate)
            if self.short_cut and new.shape == h.shape:
                new = new + h
            h = torch.where(recv.unsqueeze(-1), new, h)                      # carry over otherwise
            reached = reached | recv
            hiddens.append(h)
            edges_frac.append((sel.float() * outdeg).sum(1).mean().item() / E)
            nodes_frac.append(sel.float().sum(1).mean().item() / V)

        self.aux_loss = (torch.stack(aux).mean() * self.aux_weight) if (aux and self.aux_weight > 0) else torch.zeros((), device=dev)
        self.last_reached = reached
        self.stats = {"edges_frac": edges_frac, "nodes_frac": nodes_frac}

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

        # column 0 is the positive target after negative_sample_to_tail for both head and tail queries
        self._t_pos = t_index[:, 0] if self.training else None

        output = self.bellmanford(data, h_index[:, 0], r_index[:, 0])
        feature = output["node_feature"]
        index = t_index.unsqueeze(-1).expand(-1, -1, feature.shape[-1])
        feature = feature.gather(1, index)
        score = self.mlp(feature).squeeze(-1)
        return score.view(shape)