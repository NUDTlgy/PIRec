from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from graph import InteractionGraph, lightgcn_propagate


class TimeAwareSelfAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_dim: int, dropout: float, time_bias_init: float) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"hidden_dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.time_bias = nn.Parameter(torch.full((num_heads,), float(time_bias_init)))

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None,
        causal_mask: torch.Tensor,
        log_interval_matrix: torch.Tensor,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        h = self.norm1(x)
        q = self.q_proj(h).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        time_bias = -self.time_bias.view(1, self.num_heads, 1, 1) * log_interval_matrix.unsqueeze(1)
        attn = attn + time_bias
        attn = attn.masked_fill(causal_mask.view(1, 1, seq_len, seq_len), float("-inf"))
        if padding_mask is not None:
            attn = attn.masked_fill(padding_mask.view(bsz, 1, 1, seq_len), float("-inf"))
        prob = torch.softmax(attn, dim=-1)
        prob = self.dropout(prob)
        out = torch.matmul(prob, v).transpose(1, 2).contiguous().view(bsz, seq_len, self.dim)
        x = x + self.dropout(self.out_proj(out))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class TimeAwareTransformerEncoder(nn.Module):
    def __init__(
        self,
        dim: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        time_bias_init: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TimeAwareSelfAttentionBlock(
                    dim=dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                    time_bias_init=time_bias_init,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self._causal_mask_cache: dict[tuple[int, str, int | None], torch.Tensor] = {}

    def causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        key = (int(length), device.type, device.index)
        mask = self._causal_mask_cache.get(key)
        if mask is None or mask.device != device:
            mask = torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1)
            self._causal_mask_cache[key] = mask
        return mask

    def readout(self, hidden: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        idx = (lengths - 1).clamp_min(0)
        h = hidden[torch.arange(hidden.size(0), device=hidden.device), idx]
        return self.dropout(self.norm(h))

    def forward(
        self,
        tokens: torch.Tensor,
        lengths: torch.Tensor,
        padding_mask: torch.Tensor | None,
        log_interval_matrix: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = tokens
        causal_mask = self.causal_mask(tokens.size(1), tokens.device)
        for layer in self.layers:
            hidden = layer(hidden, padding_mask, causal_mask, log_interval_matrix)
        return hidden, self.readout(hidden, lengths)


class DualScaleTransformerRec(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        hidden_dim: int,
        graph_layers: int,
        transformer_layers: int,
        transformer_heads: int,
        transformer_ffn_dim: int,
        dropout: float,
        time_bias_init: float,
        max_seq_len: int,
        max_phase_len: int,
        num_state_prototypes: int,
        boundary_temperature: float,
        boundary_hard: bool,
        boundary_prior: float,
    ) -> None:
        super().__init__()
        self.num_users = int(num_users)
        self.num_items = int(num_items)
        self.hidden_dim = int(hidden_dim)
        self.graph_layers = int(graph_layers)
        self.max_seq_len = int(max_seq_len)
        self.max_phase_len = int(max_phase_len)
        self.num_state_prototypes = int(num_state_prototypes)
        self.boundary_temperature = float(boundary_temperature)
        self.boundary_hard = bool(boundary_hard)
        self.boundary_prior = float(boundary_prior)

        self.user_emb = nn.Embedding(num_users, hidden_dim)
        self.item_emb = nn.Embedding(num_items, hidden_dim)
        self.pos_emb = nn.Embedding(max_seq_len + 2, hidden_dim)
        self.time_proj = nn.Linear(1, hidden_dim)

        self.fine_encoder = TimeAwareTransformerEncoder(
            dim=hidden_dim,
            num_layers=transformer_layers,
            num_heads=transformer_heads,
            ffn_dim=transformer_ffn_dim,
            dropout=dropout,
            time_bias_init=time_bias_init,
        )
        self.boundary_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3 + 1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.prototype_memory = nn.Parameter(torch.randn(self.num_state_prototypes, hidden_dim))
        nn.init.xavier_uniform_(self.prototype_memory)
        self.prototype_proj = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.score_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.latest_aux_losses: dict[str, torch.Tensor] = {}

    def node_embeddings(self) -> torch.Tensor:
        return torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)

    def encode_graph(self, graph: InteractionGraph) -> tuple[torch.Tensor, torch.Tensor]:
        all_states = lightgcn_propagate(self.node_embeddings(), graph, self.graph_layers)
        user_states = all_states[: self.num_users]
        item_states = all_states[self.num_users :]
        return user_states, item_states

    @staticmethod
    def _lengths_to_padding_mask(lengths: torch.Tensor, max_len: int, device: torch.device) -> torch.Tensor:
        pos = torch.arange(max_len, device=device).unsqueeze(0)
        return pos >= lengths.unsqueeze(1)

    def _sequence_tokens(
        self,
        prefix_items: torch.Tensor,
        prefix_times: torch.Tensor,
        prefix_lengths: torch.Tensor,
        item_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = prefix_items.device
        items = prefix_items[:, -self.max_seq_len :]
        times = prefix_times[:, -self.max_seq_len :]
        seq_len = items.size(1)
        lengths = prefix_lengths.clamp(min=1, max=seq_len)
        padding_mask = self._lengths_to_padding_mask(lengths, seq_len, device)
        valid_mask = ~padding_mask

        item_tokens = item_states[items]
        if seq_len <= 1:
            delta = torch.zeros((items.size(0), seq_len), dtype=torch.float32, device=device)
        else:
            delta = torch.diff(times, dim=1).clamp_min(0).to(torch.float32)
            delta = torch.cat([torch.zeros((items.size(0), 1), dtype=torch.float32, device=device), delta], dim=1)
            delta = delta.masked_fill(padding_mask, 0.0)
            delta[:, 0] = 0.0
        log_delta = torch.log1p(delta)
        time_tokens = self.time_proj(log_delta.to(item_tokens.dtype).unsqueeze(-1))
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(items.size(0), -1)
        tokens = item_tokens + time_tokens + self.pos_emb(pos_ids)
        tokens = tokens * valid_mask.unsqueeze(-1).to(tokens.dtype)

        time_matrix = torch.abs(times.unsqueeze(2) - times.unsqueeze(1)).to(torch.float32)
        log_interval_matrix = torch.log1p(time_matrix)
        log_interval_matrix = log_interval_matrix.masked_fill(
            (~valid_mask).unsqueeze(1) | (~valid_mask).unsqueeze(2),
            0.0,
        )
        return tokens, item_tokens, lengths, padding_mask, delta, log_interval_matrix

    def _boundary_samples(
        self,
        coarse_tokens: torch.Tensor,
        delta: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prev_tokens = torch.roll(coarse_tokens, shifts=1, dims=1)
        prev_tokens[:, 0] = coarse_tokens[:, 0]
        feat = torch.cat(
            [
                coarse_tokens,
                prev_tokens,
                coarse_tokens - prev_tokens,
                delta.unsqueeze(-1).to(coarse_tokens.dtype),
            ],
            dim=-1,
        )
        logits = self.boundary_mlp(feat).squeeze(-1)
        first_pos = torch.arange(logits.size(1), device=logits.device).unsqueeze(0) == 0
        logits = torch.where(first_pos, logits.new_full(logits.shape, 8.0), logits)
        logits = logits.masked_fill(~valid_mask, -8.0)
        if self.training:
            relaxed = F.gumbel_softmax(
                torch.stack([torch.zeros_like(logits), logits], dim=-1),
                tau=max(self.boundary_temperature, 1e-4),
                hard=self.boundary_hard,
                dim=-1,
            )[..., 1]
        else:
            relaxed = torch.sigmoid(logits)
        relaxed = torch.where(first_pos, torch.ones_like(relaxed), relaxed)
        relaxed = relaxed * valid_mask.to(relaxed.dtype)
        return logits, relaxed
    def _state_discovery(
        self,
        coarse_tokens: torch.Tensor,
        boundary_probs: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.training:
            return self._state_discovery_training(coarse_tokens, boundary_probs, lengths)

        batch_size = coarse_tokens.size(0)
        state_reprs = []
        consistency_terms = []
        eps = 1e-8
        for b in range(batch_size):
            L = int(lengths[b].item())
            h = coarse_tokens[b, :L]
            probs = boundary_probs[b, :L]
            if L == 0:
                state_reprs.append(coarse_tokens[b, :1].mean(dim=0))
                consistency_terms.append(coarse_tokens.new_zeros(()))
                continue

            probs = torch.cat([probs.new_ones(1), probs[1:]], dim=0)

            state_ids = torch.cumsum((probs > 0.5).to(torch.long), dim=0) - 1
            num_states = int(state_ids.max().item()) + 1
            pooled = []
            for sid in range(num_states):
                mask = state_ids == sid
                if int(mask.sum().item()) == 0:
                    continue
                pooled.append(h[mask].mean(dim=0))
            if not pooled:
                pooled = [h[-1]]
            state_tensor = torch.stack(pooled, dim=0)
            state_weight = state_tensor.new_full((state_tensor.size(0),), 1.0 / state_tensor.size(0))

            assign = torch.softmax(torch.matmul(self.prototype_proj(state_tensor), self.prototype_memory.t()), dim=-1)
            proto_state = torch.matmul(assign, self.prototype_memory)
            state_reprs.append(torch.sum(state_weight.unsqueeze(-1) * proto_state, dim=0))
            consistency_terms.append(torch.sum(state_weight * torch.mean((proto_state - state_tensor) ** 2, dim=-1)))
        return torch.stack(state_reprs, dim=0), torch.stack(consistency_terms).mean()

    def _state_discovery_training(
        self,
        coarse_tokens: torch.Tensor,
        boundary_probs: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = coarse_tokens.shape
        eps = 1e-8
        tau_s = max(self.boundary_temperature, 1e-4)
        num_slots = max(1, int(seq_len))
        device = coarse_tokens.device

        valid_mask = torch.arange(seq_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)
        probs = boundary_probs * valid_mask.to(boundary_probs.dtype)
        if seq_len > 0:
            probs = torch.cat([probs.new_ones(batch_size, 1), probs[:, 1:]], dim=1)
        soft_state_ids = torch.cumsum(probs, dim=1) - 1.0

        num_states = lengths.clamp(min=1, max=num_slots).to(coarse_tokens.dtype)
        soft_state_ids = soft_state_ids.clamp_min(0.0)
        soft_state_ids = torch.minimum(soft_state_ids, (num_states - 1.0).unsqueeze(1))

        slot_ids = torch.arange(num_slots, device=device, dtype=coarse_tokens.dtype)
        slot_mask = slot_ids.unsqueeze(0) < num_states.unsqueeze(1)
        membership_logits = -((soft_state_ids.unsqueeze(2) - slot_ids.view(1, 1, -1)) ** 2) / tau_s
        membership_logits = membership_logits.masked_fill(~slot_mask.unsqueeze(1), float("-inf"))
        membership = torch.softmax(membership_logits, dim=2)
        membership = membership * valid_mask.unsqueeze(-1).to(membership.dtype)

        state_mass = membership.sum(dim=1)
        state_tensor = torch.bmm(membership.transpose(1, 2), coarse_tokens)
        state_tensor = state_tensor / state_mass.clamp_min(eps).unsqueeze(-1)
        state_weight = state_mass / state_mass.sum(dim=1, keepdim=True).clamp_min(eps)

        flat_states = state_tensor.reshape(batch_size * num_slots, -1)
        assign = torch.softmax(torch.matmul(self.prototype_proj(flat_states), self.prototype_memory.t()), dim=-1)
        proto_state = torch.matmul(assign, self.prototype_memory).view(batch_size, num_slots, -1)
        state_repr = torch.sum(state_weight.unsqueeze(-1) * proto_state, dim=1)
        consistency = torch.sum(
            state_weight * torch.mean((proto_state - state_tensor) ** 2, dim=-1),
            dim=1,
        ).mean()
        return state_repr, consistency

    def encode_users(
        self,
        batch: dict,
        user_states: torch.Tensor,
        item_states: torch.Tensor,
        return_branches: bool = False,
    ):
        device = item_states.device
        prefix_items = batch["prefix_items_padded"].to(device=device, dtype=torch.long, non_blocking=True)
        prefix_times = batch["prefix_times_padded"].to(device=device, dtype=torch.long, non_blocking=True)
        prefix_lengths = batch["prefix_lengths"].to(device=device, dtype=torch.long, non_blocking=True)

        seq_tokens, item_tokens, seq_lengths, padding_mask, delta, log_interval_matrix = self._sequence_tokens(
            prefix_items,
            prefix_times,
            prefix_lengths,
            item_states,
        )
        fine_hidden, fine_h = self.fine_encoder(
            seq_tokens,
            seq_lengths,
            padding_mask,
            log_interval_matrix,
        )
        valid_mask = ~padding_mask
        _, boundary_probs = self._boundary_samples(item_tokens, delta, valid_mask)
        state_h, state_consistency = self._state_discovery(item_tokens, boundary_probs, seq_lengths)

        g = torch.sigmoid(self.gate(torch.cat([fine_h, state_h], dim=-1)))
        user_h = g * fine_h + (1.0 - g) * state_h
        user_h = user_h + user_states[batch["uid"].to(device=device, dtype=torch.long)]

        boundary_rate = (boundary_probs * valid_mask.to(boundary_probs.dtype)).sum() / valid_mask.sum().clamp_min(1)
        boundary_loss = (boundary_rate - self.boundary_prior) ** 2
        self.latest_aux_losses = {
            "boundary_loss": boundary_loss,
            "state_consistency_loss": state_consistency,
            "boundary_rate": boundary_rate.detach(),
        }
        if return_branches:
            return user_h, fine_h, state_h
        return user_h

    def score_targets(
        self,
        user_h: torch.Tensor,
        target_items: torch.Tensor,
        user_states: torch.Tensor,
        item_states: torch.Tensor,
        user_ids: torch.Tensor,
    ) -> torch.Tensor:
        del user_states, user_ids
        if user_h.dim() == 1:
            user_h = user_h.unsqueeze(0)
        if target_items.dim() == 2:
            return self.score_candidate_set(user_h, target_items, item_states)
        target_vec = item_states[target_items]
        if target_vec.dim() == 1:
            target_vec = target_vec.unsqueeze(0)
        return self._score_pairs(user_h, target_vec).squeeze(-1)

    def _score_pairs(self, user_vec: torch.Tensor, item_vec: torch.Tensor) -> torch.Tensor:
        first: nn.Linear = self.score_mlp[0]
        act = self.score_mlp[1]
        drop = self.score_mlp[2]
        last: nn.Linear = self.score_mlp[3]
        hidden = F.linear(user_vec, first.weight[:, : self.hidden_dim])
        hidden = hidden + F.linear(item_vec, first.weight[:, self.hidden_dim :], first.bias)
        hidden = act(hidden)
        hidden = drop(hidden)
        return last(hidden)

    def score_candidate_set(
        self,
        user_h: torch.Tensor,
        candidate_items: torch.Tensor,
        item_states: torch.Tensor,
    ) -> torch.Tensor:
        if user_h.dim() == 1:
            user_h = user_h.unsqueeze(0)
        if candidate_items.dim() == 1:
            candidate_items = candidate_items.unsqueeze(0)
        target_vec = item_states[candidate_items]
        user_expand = user_h.unsqueeze(1).expand(-1, candidate_items.size(1), -1)
        scores = self._score_pairs(user_expand, target_vec)
        return scores.squeeze(-1)

    def full_sort_scores(
        self,
        user_h: torch.Tensor,
        user_states: torch.Tensor,
        item_states: torch.Tensor,
        user_ids: torch.Tensor,
    ) -> torch.Tensor:
        del user_states, user_ids
        n_items = item_states.size(0)
        chunk_size = max(1, min(2048, n_items))
        scores = []
        for st in range(0, n_items, chunk_size):
            ed = min(st + chunk_size, n_items)
            chunk_items = torch.arange(st, ed, device=item_states.device, dtype=torch.long)
            chunk_scores = self.score_candidate_set(
                user_h,
                chunk_items.unsqueeze(0).expand(user_h.size(0), -1),
                item_states,
            )
            scores.append(chunk_scores)
        return torch.cat(scores, dim=1)




