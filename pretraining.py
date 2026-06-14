from __future__ import annotations

import torch
import torch.nn.functional as F

from config import Config
from graph import InteractionGraph, _build_sparse_adjacency


def _dropout_graph_view(graph: InteractionGraph, drop_rate: float) -> InteractionGraph:
    if graph.edge_index.numel() == 0 or drop_rate <= 0:
        return graph
    num_edges = graph.edge_index.size(1)
    keep_mask = torch.rand(num_edges, device=graph.edge_index.device) > float(drop_rate)
    if int(keep_mask.sum().item()) == 0:
        keep_mask[torch.randint(0, num_edges, (1,), device=graph.edge_index.device)] = True
    edge_index = graph.edge_index[:, keep_mask]
    deg = torch.bincount(edge_index[0], minlength=graph.num_nodes).float().clamp_min(1.0)
    norm = (deg[edge_index[0]].rsqrt() * deg[edge_index[1]].rsqrt()).float()
    adj_t = _build_sparse_adjacency(graph.num_nodes, edge_index, norm)
    return InteractionGraph(graph.num_users, graph.num_items, edge_index, norm, adj_t)


def _nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    logits = torch.matmul(z1, z2.t()) / max(float(temperature), 1e-6)
    labels = torch.arange(z1.size(0), device=z1.device)
    loss12 = F.cross_entropy(logits, labels)
    loss21 = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss12 + loss21)


def graph_contrastive_pretrain(
    model,
    graph: InteractionGraph,
    train_edges,
    user_seen,
    cfg: Config,
    device: torch.device,
) -> None:
    del train_edges, user_seen
    epochs = max(0, int(cfg.warmup_epochs))
    if epochs == 0:
        return
    if graph.edge_index.numel() == 0:
        return

    optimizer = torch.optim.Adam(
        list(model.user_emb.parameters()) + list(model.item_emb.parameters()),
        lr=cfg.warmup_lr,
        weight_decay=cfg.weight_decay,
    )
    drop_rate = float(cfg.warmup_aug_drop)
    temperature = float(cfg.warmup_contrastive_temp)

    model.train()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        view1 = _dropout_graph_view(graph, drop_rate)
        view2 = _dropout_graph_view(graph, drop_rate)
        user_z1, item_z1 = model.encode_graph(view1)
        user_z2, item_z2 = model.encode_graph(view2)
        loss = _nt_xent(user_z1, user_z2, temperature) + _nt_xent(item_z1, item_z2, temperature)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()


warm_up_interaction_graph = graph_contrastive_pretrain
