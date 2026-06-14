from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch


@dataclass
class InteractionGraph:
    num_users: int
    num_items: int
    edge_index: torch.Tensor
    norm: torch.Tensor
    adj_t: torch.Tensor | None = None

    @property
    def num_nodes(self) -> int:
        return self.num_users + self.num_items

    def to(self, device: torch.device) -> "InteractionGraph":
        return InteractionGraph(
            num_users=self.num_users,
            num_items=self.num_items,
            edge_index=self.edge_index.to(device),
            norm=self.norm.to(device),
            adj_t=None if self.adj_t is None else self.adj_t.to(device),
        )


def _build_sparse_adjacency(
    num_nodes: int,
    edge_index: torch.Tensor,
    norm: torch.Tensor,
) -> torch.Tensor:
    return torch.sparse_coo_tensor(
        edge_index[[1, 0]],
        norm,
        size=(num_nodes, num_nodes),
        device=edge_index.device,
    ).coalesce()


def build_interaction_graph(
    num_users: int,
    num_items: int,
    train_edges: Iterable[Tuple[int, int]],
) -> InteractionGraph:
    src = []
    dst = []
    for uid, item in train_edges:
        u = int(uid)
        v = int(item) + num_users
        src.extend([u, v])
        dst.extend([v, u])
    if not src:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        norm = torch.empty((0,), dtype=torch.float32)
        return InteractionGraph(num_users, num_items, edge_index, norm)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    deg = torch.bincount(edge_index[0], minlength=num_users + num_items).float().clamp_min(1.0)
    norm = (deg[edge_index[0]].rsqrt() * deg[edge_index[1]].rsqrt()).float()
    adj_t = _build_sparse_adjacency(num_users + num_items, edge_index, norm)
    return InteractionGraph(num_users, num_items, edge_index, norm, adj_t)


def lightgcn_propagate(x: torch.Tensor, graph: InteractionGraph, num_layers: int) -> torch.Tensor:
    if num_layers <= 0 or graph.edge_index.numel() == 0:
        return x
    use_sparse = graph.adj_t is not None
    if use_sparse and x.is_cuda:
        # CUDA sparse matmul does not support fp16 under AMP.
        x = x.float()
    states = [x]
    h = x
    adj_t = graph.adj_t
    if adj_t is not None:
        adj_t = adj_t.to(dtype=torch.float32 if x.is_cuda else x.dtype, device=x.device)
    else:
        src, dst = graph.edge_index
        norm = graph.norm.to(dtype=x.dtype, device=x.device)
    for _ in range(num_layers):
        if adj_t is None:
            out = torch.zeros_like(h)
            out.index_add_(0, dst, h[src] * norm.unsqueeze(-1))
            h = out
        else:
            if h.is_cuda and hasattr(torch, "amp"):
                with torch.amp.autocast("cuda", enabled=False):
                    h = torch.sparse.mm(adj_t, h.float())
            elif h.is_cuda:
                with torch.cuda.amp.autocast(enabled=False):
                    h = torch.sparse.mm(adj_t, h.float())
            else:
                h = torch.sparse.mm(adj_t, h)
        states.append(h)
    return torch.stack(states, dim=0).mean(dim=0)
