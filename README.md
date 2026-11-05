# Budgeted Node Expansion for NBFNet

Learned and A*-style pruning with a distance-aware auxiliary loss, for faster and lighter knowledge graph link prediction.

Built on [NBFNet-PyG](https://github.com/KiddoZhu/NBFNet-PyG). A*Net ([AStarNet](https://github.com/DeepGraphLearning/AStarNet)) is used as a reference for the A*-style pruner. This is a Deep Learning semester project.

---

## 1. Motivation

NBFNet predicts a missing link `(head, relation, ?)` by running Bellman-Ford-style message passing from the head node over `L` layers. At every layer, **every reached node sends messages along all its edges**. On large graphs the number of active nodes and edges grows quickly, so time and memory become the bottleneck.

A*Net showed that most of this work is unnecessary: only a small, well-chosen part of the graph needs to be expanded at each layer.

**Research goal:** make NBFNet cheaper in **time and memory**, accepting a small loss in accuracy, and find out which pruning strategy gives the best accuracy per unit of compute.

## 2. Research questions

1. How much accuracy (MRR, Hits@k) do we lose when only a fraction `rho` of the graph is expanded per layer?
2. Does a **learned MLP pruner** beat random pruning at the same budget?
3. How does it compare with an **A*Net-style pruner**, the most efficient pruner so far?
4. Does adding a **distance-to-target auxiliary loss** make the pruner keep more useful nodes, for either learned pruner?

## 3. Method overview

```
NBFNet backbone (L Bellman-Ford layers)
        |
        v   at every layer, given node states h and query q
Pruning selector  (full | random | learned | astar)
        |
        +--> Expansion budget: keep only a fraction rho of nodes / edges
        |
        +--> Message gating: multiply messages by sigmoid(s)   (learned, astar)
        |
        v
Sparse message passing (distmult messages + sum aggregation)
        |
        v   only if training and modified_loss=true
Auxiliary distance loss
```

### 3.1 Selectors

| Selector | What it does | Trained scorer |
|---|---|---|
| `full` | Every reached node expands. This is NBFNet (with sum aggregation). | no |
| `random` | Random node scores, same budget as the others. | no |
| `learned` | MLP pruner. `s = MLP([h_u \|\| q])`, shared across layers. Keeps the top-scoring nodes until their out-edges reach the budget `rho * \|E\|`. During training a fraction `explore` of the budget goes to random nodes. | yes |
| `astar` | A*Net-style pruner. `s = MLP(h_u * W[h_u \|\| q])`. Keeps the top `K = rho * \|V\|` reached nodes, then the top `M = rho * \|E\|` of their out-edges ranked by the **destination** node's score. | yes |

Notes:

- `rho` is the fraction of nodes and edges allowed to expand per layer. It is the same number for every selector, so all runs have the same per-layer edge budget.
- Top-K selection has no gradient. For `learned` and `astar`, each kept node's messages are multiplied by `sigmoid(s)`. That is how the scorer receives gradient from the main loss, as in A*Net.
- `random` and `full` have no scorer parameters.

### 3.2 Modified loss (distance penalty)

With `--modified_loss true`, the total training loss is

```
L = L_main + lambda * L_dist
```

- `L_main`: the usual NBFNet loss (binary cross-entropy with adversarial negative weighting).
- `L_dist`: binary cross-entropy between the pruner's score `s` of each reached node and a distance target `y = exp(-0.5 * d)`, where `d` is the hop distance from that node to the true tail. Unreachable nodes get `y = 0`.
- `lambda` is `--aux_weight`.

The idea: the pruner should score nodes that are close to the answer higher, so the budget is spent on useful nodes. The distance is computed only at training time, when the target tail is known, using a reverse breadth-first search limited to `L` hops. At test time the pruner uses only its learned score.

`modified_loss=true` is only valid for `learned` and `astar`, because `full` and `random` have no trained scorer. Other combinations raise an error.

### 3.3 Variants evaluated

| Run | selector | modified_loss |
|---|---|---|
| Baseline | `full` | - |
| Random pruner | `random` | - |
| MLP pruner | `learned` | false |
| MLP pruner + distance loss | `learned` | true |
| A*Net-style pruner | `astar` | false |
| A*Net-style pruner + distance loss | `astar` | true |

### 3.4 Sparse execution

Early versions of the pruner only **masked** messages, so the full computation still ran and no time or memory was saved. The current implementation (`--sparse_exec true`, default) builds only the selected `(query, edge)` pairs, runs the layer's linear update only on receiving nodes, and uses CSR indexing to expand selected nodes without creating a `(batch, num_edges)` tensor. The old dense path is kept (`--sparse_exec false`) as a reference for equivalence tests.

## 4. Metrics

**Accuracy:** MRR, MR, Hits@1/3/10, and **tail coverage** (the share of test queries whose true tail is reached by the kept nodes; a low value means the pruner throws the answer away).

**Efficiency (logged every epoch and at test time):**

- time per epoch and per evaluation, throughput (triplets/s)
- peak GPU memory
- nodes kept and edges kept per layer (fraction of the full graph)
- active messages vs a sparse `full` run (`sparsity_vs_full`) and vs the dense upper bound (`sparsity_vs_dense`)
- parameter count, with the pruner's share reported separately

**Trade-off:** MRR vs edges kept across several `rho` values, MRR retained relative to `full`, and speedup / memory reduction relative to `full`.

## 5. Experimental protocol (equal budget)

All variants run from this single codebase with the same:

- dataset and split, hidden size, layers, learning rate, batch size, epochs
- `rho`, seeds (2-3 seeds, report mean and std)
- GPU type (use the same type for every timing run)

Timing rules: warm-up pass first, `torch.cuda.synchronize()` before stopping timers, data loading excluded. Everything is config-driven and a checkpoint is saved periodically, so each run is launched once.

## 6. Repository structure

```
nbfnet/            NBFNet model, layers, datasets, utilities (from NBFNet-PyG)
pruned_models.py   PrunedNBFNet: selectors, sparse layer, auxiliary loss, efficiency stats
script/run.py      Training and evaluation loop (with efficiency logging)
config/            YAML configs (transductive / inductive)
```

## 7. Installation

```bash
pip install torch torch-geometric easydict pyyaml
```

`torch-scatter` and `torch-sparse` are not needed: scatter operations come from `torch_geometric.utils`. Datasets (WN18RR, FB15k-237) are downloaded automatically by PyTorch Geometric.

## 8. Usage

`--gpus` fills the `{{ gpus }}` variable in the YAML config. Use `"[0]"` for the first GPU and `"[]"` for CPU.

```bash
# Baseline (plain NBFNet behaviour)
python script/run.py -c config/transductive/wn18rr.yaml --gpus "[0]" --selector full

# Random pruner
python script/run.py -c config/transductive/wn18rr.yaml --gpus "[0]" --selector random --rho 0.5

# MLP pruner
python script/run.py -c config/transductive/wn18rr.yaml --gpus "[0]" --selector learned --modified_loss false

# MLP pruner + distance loss
python script/run.py -c config/transductive/wn18rr.yaml --gpus "[0]" --selector learned --modified_loss true

# A*Net-style pruner (+ distance loss)
python script/run.py -c config/transductive/wn18rr.yaml --gpus "[0]" --selector astar --modified_loss true
```

In a Kaggle notebook, use `%cd` to enter the repo and put `!` in front of each command.

| Argument | Default | Meaning |
|---|---|---|
| `--selector` | `learned` | `full`, `random`, `learned` or `astar` |
| `--modified_loss` | `false` | add the distance loss (`learned` and `astar` only) |
| `--aux_weight` | `0.5` | lambda for the distance loss |
| `--rho` | `0.5` | fraction of nodes / edges expanded per layer |
| `--explore` | `0.2` | share of the budget spent on random nodes (`learned`, training only) |
| `--sparse_exec` | `true` | physical sparse execution; `false` runs the dense reference path |
| `--seed` | `1024` | random seed |

## 9. Results

Results will be added here once all runs finish (dataset: WN18RR / FB15k-237, `rho` = ..., seeds = ...).

| Method | MRR | Hits@1 | Hits@10 | Tail coverage | Edges kept | Time/epoch | Peak memory |
|---|---|---|---|---|---|---|---|
| full | TBD | TBD | TBD | TBD | 100% | TBD | TBD |
| random | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| learned | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| learned + dist. loss | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| astar | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| astar + dist. loss | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## 10. Limitations

- Only DistMult messages and **sum** aggregation are supported. The original NBFNet uses PNA aggregation, so `full` here is NBFNet with sum aggregation, not the original PNA baseline.
- All hidden dimensions must equal the input dimension.
- The sparse path has its own overhead (sorting, unique operations, scorer), so real speedups depend on `rho`, graph size and GPU.
- `astar` is a re-implementation of A*Net's priority function and selection rule inside this codebase, using one `rho` for both node and edge budgets (equal to A*Net with `degree_ratio = 1`). It is not the original TorchDrug code.
- The distance target uses hop distance on the training graph, limited to `L` hops.

## 11. References

- Z. Zhu et al. *Neural Bellman-Ford Networks: A General Graph Neural Network Framework for Link Prediction.* NeurIPS 2021. Code: [NBFNet-PyG](https://github.com/KiddoZhu/NBFNet-PyG).
- Z. Zhu et al. *A\*Net: A Scalable Path-based Reasoning Approach for Knowledge Graphs.* NeurIPS 2023. Code: [AStarNet](https://github.com/DeepGraphLearning/AStarNet).

## 12. Team

Add names, roles and contributions here (who built the MLP pruner and distance loss, who ran the baseline, random and A*Net experiments).

## License

Inherits the license of NBFNet-PyG (see `LICENSE`).