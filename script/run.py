import os
import sys
import time
import math
import pprint

import torch
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch import distributed as dist
from torch.utils import data as torch_data
from torch_geometric.data import Data

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from nbfnet import tasks, util

from pruned_models import PrunedNBFNet


separator = ">" * 30
line = "-" * 30


def train_and_validate(cfg, model, train_data, valid_data, filtered_data=None):
    if cfg.train.num_epoch == 0:
        return

    world_size = util.get_world_size()
    rank = util.get_rank()

    train_triplets = torch.cat([train_data.target_edge_index, train_data.target_edge_type.unsqueeze(0)]).t()
    sampler = torch_data.DistributedSampler(train_triplets, world_size, rank)
    train_loader = torch_data.DataLoader(train_triplets, cfg.train.batch_size, sampler=sampler)

    cls = cfg.optimizer.pop("class")
    optimizer = getattr(optim, cls)(model.parameters(), **cfg.optimizer)
    if world_size > 1:
        parallel_model = nn.parallel.DistributedDataParallel(model, device_ids=[device])
    else:
        parallel_model = model

    step = math.ceil(cfg.train.num_epoch / 10)
    best_result = float("-inf")
    best_epoch = -1

    batch_id = 0
    for i in range(0, cfg.train.num_epoch, step):
        parallel_model.train()
        for epoch in range(i, min(cfg.train.num_epoch, i + step)):
            if util.get_rank() == 0:
                logger.warning(separator)
                logger.warning("Epoch %d begin" % epoch)

            losses = []
            epoch_start = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            m = getattr(model, "module", model)
            if hasattr(m, "reset_stats"):
                m.reset_stats()
            sampler.set_epoch(epoch)
            for batch in train_loader:
                batch = tasks.negative_sampling(train_data, batch, cfg.task.num_negative,
                                                strict=cfg.task.strict_negative)
                pred = parallel_model(train_data, batch)
                target = torch.zeros_like(pred)
                target[:, 0] = 1
                loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
                neg_weight = torch.ones_like(pred)
                if cfg.task.adversarial_temperature > 0:
                    with torch.no_grad():
                        neg_weight[:, 1:] = F.softmax(pred[:, 1:] / cfg.task.adversarial_temperature, dim=-1)
                else:
                    neg_weight[:, 1:] = 1 / cfg.task.num_negative
                loss = (loss * neg_weight).sum(dim=-1) / neg_weight.sum(dim=-1)
                loss = loss.mean()
                main_loss = loss.item()
                dist_loss = model.aux_loss.item()
                loss = loss + model.aux_loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if util.get_rank() == 0 and batch_id % cfg.train.log_interval == 0:
                    logger.warning(separator)
                    logger.warning("main_loss: %g  dist_loss: %g  total: %g" % (main_loss, dist_loss, loss.item()))
                losses.append(loss.item())
                batch_id += 1

            if util.get_rank() == 0:
                epoch_time = time.perf_counter() - epoch_start
                avg_loss = sum(losses) / len(losses)
                train_speed = len(train_triplets) / max(1e-5, epoch_time)
                peak_mem = (torch.cuda.max_memory_allocated() / (1024 ** 2)) if torch.cuda.is_available() else 0.0
                stats = getattr(getattr(model, "module", model), "stats", {})
                sparsity_dense = stats.get("sparsity_vs_dense", stats.get("physical_sparsity", 0.0)) * 100
                sparsity_full = stats.get("sparsity_vs_full", 0.0) * 100
                nodes_frac = stats.get("nodes_frac", [])
                edges_frac = stats.get("edges_frac", [])
                logger.warning(separator)
                logger.warning("Epoch %d end" % epoch)
                logger.warning(line)
                logger.warning("average binary cross entropy: %g" % avg_loss)
                logger.warning("Epoch %d efficiency: time=%.2fs  speed=%.1f triplets/s  peak_mem=%.1f MB  sparsity_vs_dense=%.1f%%  sparsity_vs_full=%.1f%%" %
                               (epoch, epoch_time, train_speed, peak_mem, sparsity_dense, sparsity_full))
                if nodes_frac:
                    nodes_str = ", ".join("L%d: %.1f%%" % (idx + 1, f * 100) for idx, f in enumerate(nodes_frac))
                    logger.warning("Epoch %d nodes kept:  %s" % (epoch, nodes_str))
                if edges_frac:
                    edges_str = ", ".join("L%d: %.1f%%" % (idx + 1, f * 100) for idx, f in enumerate(edges_frac))
                    logger.warning("Epoch %d edges kept:  %s" % (epoch, edges_str))

        epoch = min(cfg.train.num_epoch, i + step)
        if rank == 0:
            logger.warning("Save checkpoint to model_epoch_%d.pth" % epoch)
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            torch.save(state, "model_epoch_%d.pth" % epoch)
        util.synchronize()

        if rank == 0:
            logger.warning(separator)
            logger.warning("Evaluate on valid")
        result = test(cfg, model, valid_data, filtered_data=filtered_data)
        if result > best_result:
            best_result = result
            best_epoch = epoch

    if rank == 0:
        logger.warning("Load checkpoint from model_epoch_%d.pth" % best_epoch)
    state = torch.load("model_epoch_%d.pth" % best_epoch, map_location=device,weights_only=False)
    model.load_state_dict(state["model"])
    util.synchronize()


@torch.no_grad()
def test(cfg, model, test_data, filtered_data=None):
    world_size = util.get_world_size()
    rank = util.get_rank()

    test_triplets = torch.cat([test_data.target_edge_index, test_data.target_edge_type.unsqueeze(0)]).t()
    sampler = torch_data.DistributedSampler(test_triplets, world_size, rank)
    test_loader = torch_data.DataLoader(test_triplets, cfg.train.batch_size, sampler=sampler)

    test_start = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    m = getattr(model, "module", model)
    if hasattr(m, "reset_stats"):
        m.reset_stats()

    model.eval()
    rankings = []
    num_negatives = []
    tail_coverages = []
    head_coverages = []
    for batch in test_loader:
        t_batch, h_batch = tasks.all_negative(test_data, batch)
        t_pred = model(test_data, t_batch)
        m = getattr(model, "module", model)
        t_reached = getattr(m, "last_reached", None)
        h_pred = model(test_data, h_batch)
        h_reached = getattr(m, "last_reached", None)

        if filtered_data is None:
            t_mask, h_mask = tasks.strict_negative_mask(test_data, batch)
        else:
            t_mask, h_mask = tasks.strict_negative_mask(filtered_data, batch)
        pos_h_index, pos_t_index, pos_r_index = batch.t()
        t_ranking = tasks.compute_ranking(t_pred, pos_t_index, t_mask)
        h_ranking = tasks.compute_ranking(h_pred, pos_h_index, h_mask)
        num_t_negative = t_mask.sum(dim=-1)
        num_h_negative = h_mask.sum(dim=-1)

        rankings += [t_ranking, h_ranking]
        num_negatives += [num_t_negative, num_h_negative]

        if t_reached is not None:
            t_covered = t_reached[torch.arange(len(pos_t_index), device=pos_t_index.device), pos_t_index].float()
            tail_coverages.append(t_covered)
        if h_reached is not None:
            h_covered = h_reached[torch.arange(len(pos_h_index), device=pos_h_index.device), pos_h_index].float()
            head_coverages.append(h_covered)

    ranking = torch.cat(rankings)
    num_negative = torch.cat(num_negatives)
    all_size = torch.zeros(world_size, dtype=torch.long, device=device)
    all_size[rank] = len(ranking)
    if world_size > 1:
        dist.all_reduce(all_size, op=dist.ReduceOp.SUM)
    cum_size = all_size.cumsum(0)
    all_ranking = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
    all_ranking[cum_size[rank] - all_size[rank]: cum_size[rank]] = ranking
    all_num_negative = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
    all_num_negative[cum_size[rank] - all_size[rank]: cum_size[rank]] = num_negative
    if world_size > 1:
        dist.all_reduce(all_ranking, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_num_negative, op=dist.ReduceOp.SUM)

    all_tail_cov = None
    all_head_cov = None
    if len(tail_coverages) > 0:
        tail_cov = torch.cat(tail_coverages)
        head_cov = torch.cat(head_coverages) if len(head_coverages) > 0 else None
        cov_size = torch.zeros(world_size, dtype=torch.long, device=device)
        cov_size[rank] = len(tail_cov)
        if world_size > 1:
            dist.all_reduce(cov_size, op=dist.ReduceOp.SUM)
        cum_cov_size = cov_size.cumsum(0)
        all_tail_cov = torch.zeros(cov_size.sum(), dtype=torch.float, device=device)
        all_tail_cov[cum_cov_size[rank] - cov_size[rank]: cum_cov_size[rank]] = tail_cov
        if world_size > 1:
            dist.all_reduce(all_tail_cov, op=dist.ReduceOp.SUM)

        if head_cov is not None:
            all_head_cov = torch.zeros(cov_size.sum(), dtype=torch.float, device=device)
            all_head_cov[cum_cov_size[rank] - cov_size[rank]: cum_cov_size[rank]] = head_cov
            if world_size > 1:
                dist.all_reduce(all_head_cov, op=dist.ReduceOp.SUM)

    if rank == 0:
        test_time = time.perf_counter() - test_start
        test_speed = len(test_triplets) / max(1e-5, test_time)
        peak_mem = (torch.cuda.max_memory_allocated() / (1024 ** 2)) if torch.cuda.is_available() else 0.0
        stats = getattr(getattr(model, "module", model), "stats", {})
        sparsity_dense = stats.get("sparsity_vs_dense", stats.get("physical_sparsity", 0.0)) * 100
        sparsity_full = stats.get("sparsity_vs_full", 0.0) * 100
        nodes_frac = stats.get("nodes_frac", [])
        edges_frac = stats.get("edges_frac", [])
        logger.warning(line)
        logger.warning("Test efficiency: eval_time=%.2fs  speed=%.1f triplets/s  peak_mem=%.1f MB  sparsity_vs_dense=%.1f%%  sparsity_vs_full=%.1f%%" %
                       (test_time, test_speed, peak_mem, sparsity_dense, sparsity_full))
        if nodes_frac:
            nodes_str = ", ".join("L%d: %.1f%%" % (idx + 1, f * 100) for idx, f in enumerate(nodes_frac))
            logger.warning("Test nodes kept per layer: %s" % nodes_str)
        if edges_frac:
            edges_str = ", ".join("L%d: %.1f%%" % (idx + 1, f * 100) for idx, f in enumerate(edges_frac))
            logger.warning("Test edges kept per layer: %s" % edges_str)

        for metric in cfg.task.metric:
            if metric == "mr":
                score = all_ranking.float().mean()
            elif metric == "mrr":
                score = (1 / all_ranking.float()).mean()
            elif metric.startswith("hits@"):
                values = metric[5:].split("_")
                threshold = int(values[0])
                if len(values) > 1:
                    num_sample = int(values[1])
                    # unbiased estimation
                    fp_rate = (all_ranking - 1).float() / all_num_negative
                    score = 0
                    for i in range(threshold):
                        # choose i false positive from num_sample - 1 negatives
                        num_comb = math.factorial(num_sample - 1) / \
                                   math.factorial(i) / math.factorial(num_sample - i - 1)
                        score += num_comb * (fp_rate ** i) * ((1 - fp_rate) ** (num_sample - i - 1))
                    score = score.mean()
                else:
                    score = (all_ranking <= threshold).float().mean()
            elif metric == "tail_coverage":
                score = all_tail_cov.mean() if all_tail_cov is not None else 1.0
            logger.warning("%s: %g" % (metric, score))

        if all_tail_cov is not None and "tail_coverage" not in cfg.task.metric:
            logger.warning("tail_coverage: %g" % all_tail_cov.mean())
            if all_head_cov is not None:
                logger.warning("head_coverage: %g" % all_head_cov.mean())
                all_target_cov = torch.cat([all_tail_cov, all_head_cov])
                logger.warning("target_coverage: %g" % all_target_cov.mean())
    mrr = (1 / all_ranking.float()).mean()

    return mrr


if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    working_dir = util.create_working_directory(cfg)

    torch.manual_seed(args.seed + util.get_rank())

    logger = util.get_root_logger()
    if util.get_rank() == 0:
        logger.warning("Random seed: %d" % args.seed)
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))
    is_inductive = cfg.dataset["class"].startswith("Ind")
    dataset = util.build_dataset(cfg)
    cfg.model.num_relation = dataset.num_relations
    cfg.model.pop("class", None)
    scorer_dim = getattr(args, "scorer_dim", 64)
    scorer_depth = getattr(args, "scorer_depth", 2)
    sparse_exec = getattr(args, "sparse_exec", True)
    model = PrunedNBFNet(
        selector=args.selector,
        rho=args.rho,
        explore=args.explore,
        aux_weight=args.aux_weight,
        modified_loss=args.modified_loss,
        scorer_dim=scorer_dim,
        scorer_depth=scorer_depth,
        sparse_exec=sparse_exec,
        # decay=exp(-0.5*d), activation=GELU, layer_norm=OFF, dropout=OFF (all fixed)
        **cfg.model)
    if util.get_rank() == 0:
        logger.warning(
            "Pruning config: selector=%s  modified_loss=%s  aux_weight=%g  "
            "rho=%g  explore=%g  scorer_dim=%d  scorer_depth=%d  sparse_exec=%s" %
            (args.selector, args.modified_loss, args.aux_weight,
             args.rho, args.explore, scorer_dim, scorer_depth, sparse_exec))
        logger.warning(
            "Fixed config: distance_decay=exponential(rate=0.5)  "
            "scorer=GELU/no-LayerNorm/no-Dropout")

        # Parameter count breakdown
        if hasattr(model, "parameter_breakdown"):
            param_counts = model.parameter_breakdown()
        else:
            tot = sum(p.numel() for p in model.parameters())
            param_counts = {"total": tot, "base": tot, "pruner": 0, "pruner_pct": 0.0}

        total_params = param_counts["total"]
        base_params = param_counts["base"]
        pruner_params = param_counts["pruner"]
        pruner_pct = param_counts["pruner_pct"]

        logger.warning(
            "Parameter Breakdown: total=%d (%.2fK) | base_gnn=%d (%.2fK) | pruner=%d (%.2fK, %.2f%%)" %
            (total_params, total_params / 1e3, base_params, base_params / 1e3,
             pruner_params, pruner_params / 1e3, pruner_pct))
        logger.warning(
            "The pruner adds only %.2f thousand parameters (%.2f%% of total model size)." %
            (pruner_params / 1e3, pruner_pct))

    device = util.get_device(cfg)
    model = model.to(device)
    train_data, valid_data, test_data = dataset[0], dataset[1], dataset[2]
    train_data = train_data.to(device)
    valid_data = valid_data.to(device)
    test_data = test_data.to(device)
    if is_inductive:
        # for inductive setting, use only the test fact graph for filtered ranking
        filtered_data = None
    else:
        # for transductive setting, use the whole graph for filtered ranking
        filtered_data = Data(edge_index=dataset.data.target_edge_index, edge_type=dataset.data.target_edge_type)
        filtered_data = filtered_data.to(device)

    train_and_validate(cfg, model, train_data, valid_data, filtered_data=filtered_data)
    if util.get_rank() == 0:
        logger.warning(separator)
        logger.warning("Evaluate on valid")
    test(cfg, model, valid_data, filtered_data=filtered_data)
    if util.get_rank() == 0:
        logger.warning(separator)
        logger.warning("Evaluate on test")
    test(cfg, model, test_data, filtered_data=filtered_data)

