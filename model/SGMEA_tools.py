
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import math
import re

from transformers.activations import ACT2FN
from transformers.pytorch_utils import apply_chunking_to_forward

from .layers import ProjectionHead
from .Tool_model import GAT, GCN
import pdb


class CDMRRouter(nn.Module):
    def __init__(self, args, modal_num, hidden_size):
        super().__init__()
        self.args = args
        self.modal_num = modal_num
        self.proj_dim = int(getattr(args, "cdmr_proj_dim", 32))
        self.type_emb_dim = int(getattr(args, "cdmr_type_emb_dim", 16))
        self.hidden_dim = int(getattr(args, "cdmr_hidden_dim", 64))
        type_count = int(getattr(args, "top_type_count", 0) or 0)
        if type_count <= 0:
            type_count = 6
        self.type_emb = nn.Embedding(type_count, self.type_emb_dim)
        self.modal_proj = nn.ModuleList([
            nn.Linear(hidden_size, self.proj_dim) for _ in range(modal_num)
        ])
        pair_count = modal_num * (modal_num - 1) // 2
        context_dim = modal_num * self.proj_dim + modal_num + 3 + pair_count + self.type_emb_dim
        self.context_norm = nn.LayerNorm(context_dim)
        self.mlp = nn.Sequential(
            nn.Linear(context_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, modal_num),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        beta_max = float(getattr(args, "cdmr_beta_max", 0.5))
        beta_init = float(getattr(args, "cdmr_beta_init", 0.05))
        beta_ratio = min(max(beta_init / max(beta_max, 1e-12), 1e-6), 1.0 - 1e-6)
        raw_beta_init = math.log(beta_ratio / (1.0 - beta_ratio))
        self.raw_beta = nn.Parameter(torch.full((modal_num,), raw_beta_init))
        self.last_raw_delta = None
        self.last_residual = None
        self.last_beta = None
        self.last_entropy = None
        self.last_pairwise_cos = None

    def _safe_type_ids(self, entity_type_ids, device):
        if entity_type_ids is None:
            return None
        return entity_type_ids.to(device).long().clamp(
            min=0,
            max=self.type_emb.num_embeddings - 1,
        )

    def _generic_mask(self, type_ids):
        if type_ids is None or not getattr(self.args, "freeze_type_modality_bias_entity", False):
            return None
        entity_type_id = int(getattr(self.args, "type_modality_bias_entity_type_id", -1))
        if entity_type_id < 0:
            return None
        return type_ids == entity_type_id

    def forward(self, embs, weight_norm, entity_type_ids=None):
        if entity_type_ids is None:
            return torch.zeros_like(weight_norm)
        type_ids = self._safe_type_ids(entity_type_ids, weight_norm.device)
        projected = []
        for idx, emb in enumerate(embs[:self.modal_num]):
            projected.append(self.modal_proj[idx](emb.detach()))
        weight_safe = weight_norm.detach().clamp_min(1e-12)
        entropy = -(weight_safe * torch.log(weight_safe)).sum(dim=-1, keepdim=True)
        max_weight = weight_safe.max(dim=-1, keepdim=True).values
        top2 = torch.topk(weight_safe, k=min(2, weight_safe.shape[-1]), dim=-1).values
        if top2.shape[-1] == 1:
            top1_gap = top2[:, :1]
        else:
            top1_gap = top2[:, :1] - top2[:, 1:2]
        pairwise = []
        for i in range(self.modal_num):
            for j in range(i + 1, self.modal_num):
                pairwise.append(F.cosine_similarity(
                    F.normalize(embs[i].detach(), dim=-1, eps=1e-8),
                    F.normalize(embs[j].detach(), dim=-1, eps=1e-8),
                    dim=-1,
                    eps=1e-8,
                ).unsqueeze(-1))
        pairwise_cos = torch.cat(pairwise, dim=-1) if pairwise else weight_safe.new_zeros((weight_safe.shape[0], 0))
        type_emb = self.type_emb(type_ids)
        context = torch.cat(projected + [weight_safe, entropy, max_weight, top1_gap, pairwise_cos, type_emb], dim=-1)
        raw_delta = self.mlp(self.context_norm(context))
        beta = float(getattr(self.args, "cdmr_beta_max", 0.5)) * torch.sigmoid(self.raw_beta)
        residual = beta.unsqueeze(0) * torch.tanh(raw_delta)
        residual_clamp = float(getattr(self.args, "cdmr_residual_clamp", 0.5))
        if residual_clamp > 0:
            residual = residual.clamp(min=-residual_clamp, max=residual_clamp)
        generic_mask = self._generic_mask(type_ids)
        if generic_mask is not None and generic_mask.any():
            residual = residual.masked_fill(generic_mask.unsqueeze(-1), 0.0)
        self.last_raw_delta = raw_delta.detach()
        self.last_residual = residual.detach()
        self.last_beta = beta.detach()
        self.last_entropy = entropy.detach()
        self.last_pairwise_cos = pairwise_cos.detach()
        return residual


class DEHRTypeScaleRouter(nn.Module):
    def __init__(self, args, modal_num, hidden_size):
        super().__init__()
        self.args = args
        self.modal_num = modal_num
        self.proj_dim = int(getattr(args, "dehr_proj_dim", 32))
        self.hidden_dim = int(getattr(args, "dehr_hidden_dim", 64))
        type_count = int(getattr(args, "top_type_count", 0) or 0)
        if type_count <= 0:
            type_count = 6
        self.type_count = type_count

        self.modal_proj = nn.ModuleList([
            nn.Linear(hidden_size, self.proj_dim) for _ in range(modal_num)
        ])
        token_in_dim = self.proj_dim + 7
        self.token_proj = nn.Linear(token_in_dim, self.hidden_dim)
        self.stat_proj = nn.Linear(7, self.hidden_dim)
        self.emb_token_proj = nn.Linear(self.proj_dim, self.hidden_dim)
        self.token_gate = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.emb_token_norm = nn.LayerNorm(self.hidden_dim)
        self.stat_token_norm = nn.LayerNorm(self.hidden_dim)
        self.type_token = nn.Embedding(type_count, self.hidden_dim)
        self.modal_token = nn.Parameter(torch.zeros(modal_num, self.hidden_dim))
        nn.init.normal_(self.modal_token, std=0.02)
        self.stat_feature_names = (
            "weight",
            "entropy",
            "max_weight",
            "top1_gap",
            "pairwise",
            "direction",
            "anchor",
        )
        self.register_buffer(
            "stat_feature_mask",
            self._build_stat_feature_mask(getattr(args, "dehr_stat_feature_mask", "")),
        )

        heads = int(getattr(args, "dehr_heads", 4))
        if self.hidden_dim % heads != 0:
            heads = 1
        layers = int(getattr(args, "dehr_layers", 1))
        if layers <= 0:
            self.transformer = nn.Identity()
        else:
            layer = nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=heads,
                dim_feedforward=max(self.hidden_dim * 2, 64),
                dropout=float(getattr(args, "dehr_dropout", 0.1)),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                layer,
                num_layers=layers,
            )
        self.evidence_head = nn.Linear(self.hidden_dim, modal_num)
        if getattr(args, "dehr_zero_init_head", False):
            nn.init.zeros_(self.evidence_head.weight)
            nn.init.zeros_(self.evidence_head.bias)
        self.learned_type_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(getattr(args, "dehr_dropout", 0.1))),
            nn.Linear(self.hidden_dim, modal_num),
        )
        self.contextual_modal_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 3),
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(getattr(args, "dehr_dropout", 0.1))),
            nn.Linear(self.hidden_dim, 1),
        )
        if getattr(args, "dehr_learned_zero_init", False):
            nn.init.zeros_(self.learned_type_head[-1].weight)
            nn.init.zeros_(self.learned_type_head[-1].bias)
            nn.init.zeros_(self.contextual_modal_head[-1].weight)
            nn.init.zeros_(self.contextual_modal_head[-1].bias)
        self.direction_gain = nn.Parameter(torch.ones(modal_num))
        direction_values = str(getattr(args, "dehr_direction_values", "") or "").strip()
        direction = []
        if direction_values:
            direction = [float(x) for x in direction_values.split(",") if x.strip()]
        if not direction:
            direction = [-0.35, 0.15, -0.10, 0.30]
        if len(direction) < modal_num:
            direction = direction + [0.0] * (modal_num - len(direction))
        direction_tensor = torch.tensor(direction[:modal_num], dtype=torch.float)
        direction_tensor = direction_tensor - direction_tensor.mean()
        self.register_buffer("reliability_direction", direction_tensor)
        self.type_direction_residual = nn.Embedding(type_count, modal_num)
        nn.init.zeros_(self.type_direction_residual.weight)

        rho_max = float(getattr(args, "dehr_rho_max", 3.0))
        rho_values = []
        rho_init_values = str(getattr(args, "dehr_rho_init_values", "") or "").strip()
        if rho_init_values:
            rho_values = [float(x) for x in rho_init_values.split(",") if x.strip()]
        if not rho_values:
            rho_values = [float(getattr(args, "dehr_rho_init", 1.0))]
        if len(rho_values) == 1:
            rho_values = rho_values * type_count
        if len(rho_values) < type_count:
            rho_values = rho_values + [rho_values[-1]] * (type_count - len(rho_values))
        rho_values = rho_values[:type_count]
        rho_tensor = torch.tensor(rho_values, dtype=torch.float)
        rho_ratio = (rho_tensor / max(rho_max, 1e-12)).clamp(min=1e-6, max=1.0 - 1e-6)
        raw_rho_init = torch.log(rho_ratio / (1.0 - rho_ratio))
        self.raw_rho = nn.Parameter(raw_rho_init)

        self.last_bias = None
        self.last_u = None
        self.last_alpha = None
        self.last_rho = None
        self.last_scale = None
        self.last_kl = None
        self.last_type_ids = None
        self.last_direction = None
        self.last_u_free = None
        self.last_residual = None
        self.last_residual_confidence = None
        self.last_residual_drive = None
        self.last_global_u = None
        self.last_type_u = None
        self.last_global_bias = None
        self.last_type_bias = None
        self.last_evidence_bias = None
        self.last_global_bias_abs_mean = None
        self.last_type_bias_abs_mean = None
        self.last_evidence_bias_abs_mean = None
        self.last_learned_type_u = None
        self.last_contextual_u = None
        self.last_contextual_mix = None

    def _build_stat_feature_mask(self, spec):
        spec = str(spec or "").strip()
        feature_count = len(self.stat_feature_names)
        if not spec:
            return torch.ones((1, feature_count), dtype=torch.float)
        compact = spec.replace(",", "").replace(" ", "").replace("_", "").replace("-", "")
        if set(compact) <= {"0", "1"} and len(compact) == feature_count:
            return torch.tensor([float(ch) for ch in compact], dtype=torch.float).view(1, -1)
        lowered = spec.lower()
        if lowered in {"none", "zero", "zeros", "off"}:
            return torch.zeros((1, feature_count), dtype=torch.float)
        aliases = {
            "base_weight": "weight",
            "w": "weight",
            "max": "max_weight",
            "maxw": "max_weight",
            "gap": "top1_gap",
            "top_gap": "top1_gap",
            "cos": "pairwise",
            "cosine": "pairwise",
            "pairwise_mean": "pairwise",
            "reliability": "direction",
            "prior": "direction",
            "anchor_direction": "anchor",
        }
        selected = set()
        for raw_name in re.split(r"[,;|\s]+", lowered):
            if not raw_name:
                continue
            name = aliases.get(raw_name, raw_name)
            if name == "all":
                selected.update(self.stat_feature_names)
            elif name in self.stat_feature_names:
                selected.add(name)
        return torch.tensor(
            [1.0 if name in selected else 0.0 for name in self.stat_feature_names],
            dtype=torch.float,
        ).view(1, -1)

    def _safe_type_ids(self, entity_type_ids, device):
        if entity_type_ids is None:
            return None
        return entity_type_ids.to(device).long().clamp(min=0, max=self.type_count - 1)

    def _generic_mask(self, type_ids):
        if type_ids is None or not getattr(self.args, "freeze_type_modality_bias_entity", False):
            return None
        entity_type_id = int(getattr(self.args, "type_modality_bias_entity_type_id", -1))
        if entity_type_id < 0:
            return None
        return type_ids == entity_type_id

    def _dirichlet_kl_to_uniform(self, alpha):
        alpha0 = alpha.sum(dim=-1, keepdim=True)
        k = alpha.shape[-1]
        log_b = torch.lgamma(alpha).sum(dim=-1) - torch.lgamma(alpha0.squeeze(-1))
        log_b_uniform = -torch.lgamma(alpha.new_tensor(float(k)))
        dig = torch.digamma(alpha) - torch.digamma(alpha0)
        kl = ((alpha - 1.0) * dig).sum(dim=-1) + log_b_uniform - log_b
        return kl.mean()

    def regularization_loss(self):
        loss = None
        if self.last_alpha is not None and float(getattr(self.args, "dehr_kl_weight", 0.0)) > 0:
            loss = float(getattr(self.args, "dehr_kl_weight", 0.0)) * self._dirichlet_kl_to_uniform(self.last_alpha)
        if self.last_residual is not None and float(getattr(self.args, "dehr_residual_l2_weight", 0.0)) > 0:
            residual_loss = self.last_residual.pow(2).mean()
            scaled = float(getattr(self.args, "dehr_residual_l2_weight", 0.0)) * residual_loss
            loss = scaled if loss is None else loss + scaled
        if float(getattr(self.args, "dehr_rho_weight", 0.0)) > 0:
            rho = self._rho()
            rho_loss = torch.log(rho.clamp_min(1e-12)).pow(2).mean()
            scaled = float(getattr(self.args, "dehr_rho_weight", 0.0)) * rho_loss
            loss = scaled if loss is None else loss + scaled
        residual_l2 = float(getattr(self.args, "dehr_type_direction_residual_l2_weight", 0.0))
        residual_scale = float(getattr(self.args, "dehr_type_direction_residual_scale", 0.0))
        if residual_l2 > 0 and residual_scale > 0:
            residual = self._type_direction_residual(torch.arange(self.type_count, device=self.raw_rho.device))
            scaled = residual_l2 * residual.pow(2).mean()
            loss = scaled if loss is None else loss + scaled
        learned_l2 = float(getattr(self.args, "dehr_learned_bias_l2_weight", 0.0))
        direction_mode = getattr(self.args, "dehr_direction_mode", "free")
        if learned_l2 > 0 and direction_mode in ("learned_type", "type_transformer") and self.last_residual is not None:
            scaled = learned_l2 * self.last_residual.pow(2).mean()
            loss = scaled if loss is None else loss + scaled
        if loss is None:
            return None
        return loss

    def _rho(self):
        rho_max = float(getattr(self.args, "dehr_rho_max", 3.0))
        return rho_max * torch.sigmoid(self.raw_rho)

    def set_reliability_direction(self, direction):
        direction = torch.as_tensor(direction, dtype=self.reliability_direction.dtype, device=self.reliability_direction.device)
        if direction.numel() < self.modal_num:
            pad = direction.new_zeros(self.modal_num - direction.numel())
            direction = torch.cat([direction.flatten(), pad], dim=0)
        direction = direction.flatten()[:self.modal_num]
        direction = direction - direction.mean()
        self.reliability_direction.copy_(direction)

    def set_type_direction_residual(self, residual_by_type):
        residual = torch.as_tensor(
            residual_by_type,
            dtype=self.type_direction_residual.weight.dtype,
            device=self.type_direction_residual.weight.device,
        )
        if residual.dim() != 2:
            raise ValueError(f"type direction residual must be 2-D, got shape={tuple(residual.shape)}")
        rows = min(residual.shape[0], self.type_direction_residual.weight.shape[0])
        cols = min(residual.shape[1], self.type_direction_residual.weight.shape[1])
        residual = residual[:rows, :cols]
        residual = residual - residual.mean(dim=-1, keepdim=True)
        scale = float(getattr(self.args, "dehr_type_direction_residual_scale", 0.0))
        with torch.no_grad():
            self.type_direction_residual.weight.zero_()
            if scale <= 0:
                return
            ratio = (residual / max(scale, 1e-12)).clamp(min=-0.999, max=0.999)
            raw = 0.5 * torch.log((1.0 + ratio) / (1.0 - ratio))
            self.type_direction_residual.weight[:rows, :cols].copy_(raw)

    def _global_direction(self, device):
        direction = self.reliability_direction.to(device)
        global_weight = float(getattr(self.args, "dehr_global_direction_weight", 1.0))
        return direction * global_weight

    def _type_direction_residual(self, type_ids):
        scale = float(getattr(self.args, "dehr_type_direction_residual_scale", 0.0))
        if scale <= 0:
            return self.reliability_direction.new_zeros((type_ids.shape[0], self.modal_num)).to(type_ids.device)
        residual = scale * torch.tanh(self.type_direction_residual(type_ids))
        clamp = float(getattr(self.args, "dehr_type_direction_residual_max", 0.35))
        if clamp > 0:
            residual = residual.clamp(min=-clamp, max=clamp)
        residual = residual - residual.mean(dim=-1, keepdim=True)
        return residual

    def forward(self, embs, weight_norm, entity_type_ids=None):
        if entity_type_ids is None:
            return torch.zeros_like(weight_norm)
        type_ids = self._safe_type_ids(entity_type_ids, weight_norm.device)
        projected = []
        for idx, emb in enumerate(embs[:self.modal_num]):
            projected_emb = self.modal_proj[idx](emb.detach())
            if getattr(self.args, "dehr_drop_projected_token", False):
                projected_emb = torch.zeros_like(projected_emb)
            projected.append(projected_emb)

        weight_safe = weight_norm.detach().clamp_min(1e-12)
        entropy = -(weight_safe * torch.log(weight_safe)).sum(dim=-1, keepdim=True)
        max_weight = weight_safe.max(dim=-1, keepdim=True).values
        top2 = torch.topk(weight_safe, k=min(2, weight_safe.shape[-1]), dim=-1).values
        if top2.shape[-1] == 1:
            top1_gap = top2[:, :1]
        else:
            top1_gap = top2[:, :1] - top2[:, 1:2]

        pairwise_mean = []
        for i in range(self.modal_num):
            sims = []
            for j in range(self.modal_num):
                if i == j:
                    continue
                sims.append(F.cosine_similarity(
                    F.normalize(embs[i].detach(), dim=-1, eps=1e-8),
                    F.normalize(embs[j].detach(), dim=-1, eps=1e-8),
                    dim=-1,
                    eps=1e-8,
                ).unsqueeze(-1))
            if sims:
                pairwise_mean.append(torch.cat(sims, dim=-1).mean(dim=-1, keepdim=True))
            else:
                pairwise_mean.append(weight_safe.new_zeros((weight_safe.shape[0], 1)))

        tokens = []
        for idx in range(self.modal_num):
            stats = torch.cat([
                weight_safe[:, idx:idx + 1],
                entropy,
                max_weight,
                top1_gap,
                pairwise_mean[idx],
                self.reliability_direction[idx].to(weight_norm.device).view(1, 1).expand(weight_safe.shape[0], 1),
                (
                    self.reliability_direction[idx].to(weight_norm.device)
                    * float(getattr(self.args, "dehr_anchor_scale", 1.0))
                ).view(1, 1).expand(weight_safe.shape[0], 1),
            ], dim=-1)
            stats = stats * self.stat_feature_mask.to(stats.device)
            token = torch.cat([projected[idx], stats], dim=-1)
            if getattr(self.args, "dehr_token_fusion", "concat") == "gated":
                emb_token = self.emb_token_norm(self.emb_token_proj(projected[idx]))
                stat_token = self.stat_token_norm(self.stat_proj(stats))
                gate = torch.sigmoid(self.token_gate(torch.cat([emb_token, stat_token], dim=-1)))
                fused_token = gate * stat_token + (1.0 - gate) * emb_token
                tokens.append(fused_token + self.modal_token[idx].unsqueeze(0))
            else:
                tokens.append(self.token_proj(token) + self.modal_token[idx].unsqueeze(0))
        modal_tokens = torch.stack(tokens, dim=1)
        type_token = self.type_token(type_ids).unsqueeze(1)
        transformer_input = torch.cat([type_token, modal_tokens], dim=1)
        hidden = self.transformer(transformer_input)
        raw = self.evidence_head(hidden[:, 0])
        evidence = F.softplus(raw * self.direction_gain.unsqueeze(0))
        alpha0 = float(getattr(self.args, "dehr_alpha0", 1.0))
        alpha = evidence + max(alpha0, 1e-6)
        digamma = torch.digamma(alpha)
        u_free = digamma - digamma.mean(dim=-1, keepdim=True)
        direction_mode = getattr(self.args, "dehr_direction_mode", "free")
        global_direction = self._global_direction(weight_norm.device).unsqueeze(0)
        type_direction = self._type_direction_residual(type_ids)
        learned_type_u = None
        contextual_u = None
        contextual_mix = 0.0
        if direction_mode in ("learned_type", "type_transformer"):
            learned_raw = self.learned_type_head(hidden[:, 0])
            learned_raw = learned_raw - learned_raw.mean(dim=-1, keepdim=True)
            learned_scale = float(getattr(self.args, "dehr_learned_bias_scale", 0.5))
            learned_type_u = learned_scale * torch.tanh(learned_raw)
            contextual_mix = float(getattr(self.args, "dehr_contextual_mix", 0.0))
            if contextual_mix > 0:
                contextual_mix = min(max(contextual_mix, 0.0), 1.0)
                type_hidden = hidden[:, 0:1].expand(-1, self.modal_num, -1)
                modal_hidden = hidden[:, 1:1 + self.modal_num]
                contextual_source = getattr(self.args, "dehr_contextual_source", "hidden")
                if contextual_source == "residual":
                    type_hidden = (hidden[:, 0:1] - transformer_input[:, 0:1]).expand(-1, self.modal_num, -1)
                    modal_hidden = hidden[:, 1:1 + self.modal_num] - transformer_input[:, 1:1 + self.modal_num]
                contextual_input = torch.cat(
                    [type_hidden, modal_hidden, type_hidden * modal_hidden],
                    dim=-1,
                )
                contextual_raw = self.contextual_modal_head(contextual_input).squeeze(-1)
                contextual_raw = contextual_raw - contextual_raw.mean(dim=-1, keepdim=True)
                contextual_scale = float(getattr(self.args, "dehr_contextual_scale", -1.0))
                if contextual_scale <= 0:
                    contextual_scale = learned_scale
                contextual_u = contextual_scale * torch.tanh(contextual_raw)
                u = (1.0 - contextual_mix) * learned_type_u + contextual_mix * contextual_u
            else:
                u = learned_type_u
            residual = learned_type_u
            confidence = None
            drive = learned_type_u
        elif direction_mode in ("constrained", "anchor_residual"):
            direction = global_direction + type_direction
            anchor_scale = float(getattr(self.args, "dehr_anchor_scale", 1.0))
            anchor = direction * anchor_scale
            if direction_mode == "anchor_residual":
                gamma = float(getattr(self.args, "dehr_residual_gamma", 0.0))
                if int(getattr(self.args, "dehr_residual_confidence", 1)) > 0:
                    concentration = alpha.sum(dim=-1, keepdim=True)
                    confidence = (1.0 - (float(self.modal_num) / concentration.clamp_min(float(self.modal_num)))).clamp(min=0.0, max=1.0)
                else:
                    confidence = torch.ones_like(u_free[:, :1])
                residual_form = getattr(self.args, "dehr_residual_form", "additive")
                if residual_form == "completion_confidence":
                    anchor_full = anchor.expand_as(u_free)
                    target_full = direction.expand_as(u_free)
                    init_evidence = F.softplus(raw.new_zeros(raw.shape))
                    init_alpha = init_evidence + max(alpha0, 1e-6)
                    init_concentration = init_alpha.sum(dim=-1, keepdim=True)
                    init_confidence = (1.0 - (float(self.modal_num) / init_concentration.clamp_min(float(self.modal_num)))).clamp(min=0.0, max=1.0)
                    centered_confidence = confidence - init_confidence.detach()
                    completion = (torch.sigmoid(gamma * centered_confidence) - 0.5).clamp(min=-0.5, max=0.5)
                    u = anchor_full + completion * (target_full - anchor_full)
                    residual = u - anchor_full
                    drive = completion
                elif residual_form == "confidence_amplitude":
                    dir_full = anchor.expand_as(u_free)
                    init_evidence = F.softplus(raw.new_zeros(raw.shape))
                    init_alpha = init_evidence + max(alpha0, 1e-6)
                    init_concentration = init_alpha.sum(dim=-1, keepdim=True)
                    init_confidence = (1.0 - (float(self.modal_num) / init_concentration.clamp_min(float(self.modal_num)))).clamp(min=0.0, max=1.0)
                    centered_confidence = confidence - init_confidence.detach()
                    amplitude = gamma * torch.tanh(centered_confidence)
                    u = dir_full * (1.0 + amplitude)
                    residual = u - dir_full
                    drive = centered_confidence
                elif residual_form in ("directional_amplitude", "directional_shrink"):
                    dir_full = anchor.expand_as(u_free)
                    dir_unit = dir_full / dir_full.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                    drive = (u_free * dir_unit).sum(dim=-1, keepdim=True)
                    if residual_form == "directional_shrink":
                        amplitude = -abs(gamma) * confidence * torch.tanh(drive.abs())
                    else:
                        amplitude = gamma * confidence * torch.tanh(drive)
                    u = dir_full * (1.0 + amplitude)
                    residual = u - dir_full
                elif residual_form == "amplitude":
                    amplitude = gamma * confidence * torch.tanh(u_free)
                    u = anchor.expand_as(u_free) * (1.0 + amplitude)
                    residual = u - anchor.expand_as(u_free)
                    drive = amplitude
                else:
                    residual = u_free - u_free.mean(dim=-1, keepdim=True)
                    residual = gamma * confidence * torch.tanh(residual)
                    u = anchor.expand_as(u_free) + residual
                    drive = residual
            else:
                evidence_mix = float(getattr(self.args, "dehr_evidence_mix", 0.0))
                if evidence_mix > 0:
                    confidence = torch.tanh(u_free.abs().mean(dim=-1, keepdim=True))
                    u = anchor * (1.0 + evidence_mix * confidence)
                    residual = u - anchor.expand_as(u_free)
                    drive = confidence
                else:
                    u = anchor.expand_as(u_free)
                    confidence = None
                    residual = torch.zeros_like(u_free)
                    drive = torch.zeros_like(u_free[:, :1])
            if direction_mode == "anchor_residual":
                pass
            elif evidence_mix > 0:
                pass
            else:
                confidence = None
        else:
            u = u_free
            residual = u_free.new_zeros(u_free.shape)
            confidence = None
            drive = torch.zeros_like(u_free[:, :1])
        rho = self._rho()[type_ids].unsqueeze(-1)
        bias = rho * u
        anchor_scale = float(getattr(self.args, "dehr_anchor_scale", 1.0))
        if direction_mode in ("learned_type", "type_transformer"):
            global_u = torch.zeros_like(u)
            type_u = learned_type_u
            global_bias = torch.zeros_like(bias)
            type_bias = bias
            evidence_bias = torch.zeros_like(bias)
        else:
            global_u = global_direction.expand_as(u) * anchor_scale
            type_u = type_direction * anchor_scale
            global_bias = rho * global_u
            type_bias = rho * type_u
            evidence_bias = bias - global_bias - type_bias
        bias_clamp = float(getattr(self.args, "dehr_bias_clamp", 0.0))
        if bias_clamp > 0:
            bias = bias.clamp(min=-bias_clamp, max=bias_clamp)
            global_bias = global_bias.clamp(min=-bias_clamp, max=bias_clamp)
            type_bias = type_bias.clamp(min=-bias_clamp, max=bias_clamp)
            evidence_bias = evidence_bias.clamp(min=-bias_clamp, max=bias_clamp)
        generic_mask = self._generic_mask(type_ids)
        if generic_mask is not None and generic_mask.any():
            bias = bias.masked_fill(generic_mask.unsqueeze(-1), 0.0)
            global_bias = global_bias.masked_fill(generic_mask.unsqueeze(-1), 0.0)
            type_bias = type_bias.masked_fill(generic_mask.unsqueeze(-1), 0.0)
            evidence_bias = evidence_bias.masked_fill(generic_mask.unsqueeze(-1), 0.0)
        self.last_bias = bias.detach()
        self.last_u = u.detach()
        self.last_u_free = u_free.detach()
        self.last_residual = residual.detach()
        self.last_residual_confidence = None if confidence is None else confidence.detach()
        self.last_residual_drive = drive.detach()
        self.last_direction = self.reliability_direction.detach()
        self.last_global_u = global_u.detach()
        self.last_type_u = type_u.detach()
        self.last_global_bias = global_bias.detach()
        self.last_type_bias = type_bias.detach()
        self.last_evidence_bias = evidence_bias.detach()
        self.last_learned_type_u = None if learned_type_u is None else learned_type_u.detach()
        self.last_contextual_u = None if contextual_u is None else contextual_u.detach()
        self.last_contextual_mix = contextual_mix
        self.last_global_bias_abs_mean = float(global_bias.detach().abs().mean().item())
        self.last_type_bias_abs_mean = float(type_bias.detach().abs().mean().item())
        self.last_evidence_bias_abs_mean = float(evidence_bias.detach().abs().mean().item())
        self.last_alpha = alpha
        rho_all = self._rho()
        self.last_rho = rho_all.detach()
        self.last_scale = torch.exp(bias.detach())
        self.last_type_ids = type_ids.detach()
        self.last_kl = self._dirichlet_kl_to_uniform(alpha.detach())
        return bias


class TypeModalityHyperGate(nn.Module):
    def __init__(self, args, modal_num):
        super().__init__()
        self.args = args
        self.modal_num = modal_num
        type_count = int(getattr(args, "top_type_count", 0) or 0)
        if type_count <= 0:
            type_count = 6
        self.type_count = type_count
        self.stat_dim = 8
        hidden_dim = int(getattr(args, "tmhg_hidden_dim", 64))
        self.hidden_dim = hidden_dim
        self.input_mode = getattr(args, "tmhg_input_mode", "stat")

        self.register_buffer("stats", torch.zeros(type_count, modal_num, self.stat_dim))
        self.register_buffer("type_count_buffer", torch.zeros(type_count))
        self.register_buffer("global_stats", torch.zeros(1, modal_num, self.stat_dim))
        self.prior_feature_dim = 4
        self.register_buffer("prior_logits", torch.zeros(type_count, modal_num))
        self.register_buffer("prior_q", torch.full((type_count, modal_num), 1.0 / float(modal_num)))
        self.register_buffer("prior_mask", torch.zeros(type_count, dtype=torch.bool))
        self.register_buffer("prior_features", torch.zeros(type_count, modal_num, self.prior_feature_dim))
        self.stats_ready = False
        self.prior_ready = False

        self.type_token = nn.Embedding(type_count, hidden_dim)
        self.modal_token = nn.Parameter(torch.zeros(modal_num, hidden_dim))
        nn.init.normal_(self.modal_token, std=0.02)
        self.availability_token = nn.Embedding(2, hidden_dim)
        nn.init.normal_(self.availability_token.weight, std=0.02)
        self.degree_buckets = max(int(getattr(args, "tmhg_degree_buckets", 8)), 2)
        self.degree_token = nn.Embedding(self.degree_buckets, hidden_dim)
        nn.init.normal_(self.degree_token.weight, std=0.02)
        self.prototype_role_token = nn.Embedding(2, hidden_dim)
        nn.init.normal_(self.prototype_role_token.weight, std=0.02)
        token_in_dim = int(getattr(args, "tmhg_token_proj_dim", 300))
        self.token_in_dim = token_in_dim
        self.register_buffer("type_modal_prototypes", torch.zeros(type_count, modal_num, token_in_dim))
        self.register_buffer("global_modal_prototype", torch.zeros(1, modal_num, token_in_dim))
        self.register_buffer("prototype_type_count", torch.zeros(type_count))
        self.prototype_ready = False
        self.stat_proj = nn.Sequential(
            nn.LayerNorm(self.stat_dim),
            nn.Linear(self.stat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(getattr(args, "tmhg_dropout", 0.1))),
        )
        self.prior_proj = nn.Sequential(
            nn.LayerNorm(self.prior_feature_dim),
            nn.Linear(self.prior_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(getattr(args, "tmhg_dropout", 0.1))),
        )
        self.prior_logit_proj = nn.Sequential(
            nn.LayerNorm(1),
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(getattr(args, "tmhg_dropout", 0.1))),
        )
        semantic_proj_layers = max(int(getattr(args, "tmhg_semantic_proj_layers", 2)), 1)
        semantic_layers = [nn.LayerNorm(token_in_dim)]
        in_dim = token_in_dim
        for layer_idx in range(semantic_proj_layers):
            semantic_layers.append(nn.Linear(in_dim, hidden_dim))
            if layer_idx != semantic_proj_layers - 1:
                semantic_layers.append(nn.GELU())
                semantic_layers.append(nn.Dropout(float(getattr(args, "tmhg_dropout", 0.1))))
            in_dim = hidden_dim
        self.semantic_proj = nn.Sequential(*semantic_layers)
        heads = int(getattr(args, "tmhg_heads", 4))
        if hidden_dim % heads != 0:
            heads = 1
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=max(hidden_dim * 2, 64),
            dropout=float(getattr(args, "tmhg_dropout", 0.1)),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=int(getattr(args, "tmhg_layers", 1)),
        )
        self.residual_head = nn.Linear(hidden_dim, 1)
        self.moe_gate = nn.Linear(hidden_dim, int(getattr(args, "tmhg_moe_experts", 3)))
        self.moe_experts = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            for _ in range(int(getattr(args, "tmhg_moe_experts", 3)))
        ])
        if getattr(args, "tmhg_zero_init_residual", False):
            nn.init.zeros_(self.residual_head.weight)
            nn.init.zeros_(self.residual_head.bias)
            for expert in self.moe_experts:
                nn.init.zeros_(expert[-1].weight)
                nn.init.zeros_(expert[-1].bias)

        self.raw_feature_gain = nn.Parameter(torch.zeros(self.stat_dim))
        self.raw_global_mix = nn.Parameter(torch.tensor(0.0))
        self.global_bias = nn.Parameter(torch.zeros(modal_num))
        self.type_bias = nn.Embedding(type_count, modal_num)
        nn.init.zeros_(self.type_bias.weight)
        strength_max = max(float(getattr(args, "tmhg_strength_max", 1.5)), 1e-6)
        strength_init = min(max(float(getattr(args, "tmhg_strength_init", 1.0)), 1e-6), strength_max * 0.999)
        strength_ratio = min(max(strength_init / strength_max, 1e-6), 1.0 - 1e-6)
        strength_raw = math.log(strength_ratio / (1.0 - strength_ratio))
        self.raw_type_strength = nn.Parameter(torch.full((type_count,), strength_raw))
        self.strength_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        if getattr(args, "tmhg_zero_init_residual", False):
            nn.init.zeros_(self.strength_head[-1].weight)
            nn.init.zeros_(self.strength_head[-1].bias)
        self.gamma_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(getattr(args, "tmhg_dropout", 0.1))),
            nn.Linear(hidden_dim, 1),
        )
        self.prototype_interaction_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.type_prototype_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        if getattr(args, "tmhg_zero_init_residual", False):
            nn.init.zeros_(self.gamma_head[-1].weight)
            nn.init.zeros_(self.gamma_head[-1].bias)
            nn.init.zeros_(self.prototype_interaction_head[-1].weight)
            nn.init.zeros_(self.prototype_interaction_head[-1].bias)
            nn.init.zeros_(self.type_prototype_head[-1].weight)
            nn.init.zeros_(self.type_prototype_head[-1].bias)

        self.last_bias = None
        self.last_q = None
        self.last_logits = None
        self.last_evidence = None
        self.last_residual = None
        self.last_alpha = None
        self.last_type_ids = None
        self.last_global_mix = None
        self.last_scale = None
        self.last_q_all = None
        self.last_prior_kl = None
        self.last_prior_cos = None
        self.last_type_strength = None
        self.last_gamma = None
        self.last_token_norm = None
        self.last_proto_interaction = None
        self._last_q_all_live = None
        self.register_buffer("entity_degree_bucket", torch.zeros(1, dtype=torch.long))

    def set_type_prototypes(self, prototypes, type_counts=None, global_prototype=None):
        prototypes = torch.as_tensor(
            prototypes,
            dtype=self.type_modal_prototypes.dtype,
            device=self.type_modal_prototypes.device,
        )
        rows = min(prototypes.shape[0], self.type_modal_prototypes.shape[0])
        cols = min(prototypes.shape[1], self.type_modal_prototypes.shape[1])
        dims = min(prototypes.shape[2], self.type_modal_prototypes.shape[2])
        with torch.no_grad():
            self.type_modal_prototypes.zero_()
            self.type_modal_prototypes[:rows, :cols, :dims].copy_(prototypes[:rows, :cols, :dims])
            if global_prototype is None:
                if type_counts is not None:
                    counts = torch.as_tensor(
                        type_counts,
                        dtype=self.prototype_type_count.dtype,
                        device=self.prototype_type_count.device,
                    ).flatten()
                    self.prototype_type_count.zero_()
                    self.prototype_type_count[:min(counts.numel(), self.prototype_type_count.numel())].copy_(
                        counts[:self.prototype_type_count.numel()]
                    )
                observed = self.prototype_type_count > 0
                if observed.any():
                    weights = self.prototype_type_count[observed].view(-1, 1, 1).clamp_min(1.0)
                    g = (self.type_modal_prototypes[observed] * weights).sum(dim=0, keepdim=True) / weights.sum()
                else:
                    g = self.type_modal_prototypes.mean(dim=0, keepdim=True)
            else:
                g = torch.as_tensor(
                    global_prototype,
                    dtype=self.global_modal_prototype.dtype,
                    device=self.global_modal_prototype.device,
                )
                if g.dim() == 2:
                    g = g.unsqueeze(0)
            self.global_modal_prototype.zero_()
            self.global_modal_prototype[:, :min(g.shape[1], self.modal_num), :min(g.shape[2], self.token_in_dim)].copy_(
                g[:, :self.modal_num, :self.token_in_dim]
            )
            if type_counts is not None:
                counts = torch.as_tensor(
                    type_counts,
                    dtype=self.prototype_type_count.dtype,
                    device=self.prototype_type_count.device,
                ).flatten()
                self.prototype_type_count.zero_()
                self.prototype_type_count[:min(counts.numel(), self.prototype_type_count.numel())].copy_(
                    counts[:self.prototype_type_count.numel()]
                )
            self.prototype_ready = True

    def _safe_type_ids(self, entity_type_ids, device):
        if entity_type_ids is None:
            return None
        return entity_type_ids.to(device).long().clamp(min=0, max=self.type_count - 1)

    def _generic_mask(self, type_ids):
        if type_ids is None or not getattr(self.args, "freeze_type_modality_bias_entity", False):
            return None
        entity_type_id = int(getattr(self.args, "type_modality_bias_entity_type_id", -1))
        if entity_type_id < 0:
            return None
        return type_ids == entity_type_id

    def set_stats(self, stats, type_counts=None, global_stats=None):
        stats = torch.as_tensor(stats, dtype=self.stats.dtype, device=self.stats.device)
        rows = min(stats.shape[0], self.stats.shape[0])
        cols = min(stats.shape[1], self.stats.shape[1])
        dims = min(stats.shape[2], self.stats.shape[2])
        with torch.no_grad():
            self.stats.zero_()
            self.stats[:rows, :cols, :dims].copy_(stats[:rows, :cols, :dims])
            if type_counts is not None:
                counts = torch.as_tensor(type_counts, dtype=self.type_count_buffer.dtype, device=self.type_count_buffer.device)
                self.type_count_buffer.zero_()
                self.type_count_buffer[:min(counts.numel(), self.type_count_buffer.numel())].copy_(
                    counts.flatten()[:self.type_count_buffer.numel()]
                )
            if global_stats is None:
                observed = self.type_count_buffer > 0
                if observed.any():
                    weights = self.type_count_buffer[observed].view(-1, 1, 1).clamp_min(1.0)
                    g = (self.stats[observed] * weights).sum(dim=0, keepdim=True) / weights.sum()
                else:
                    g = self.stats.mean(dim=0, keepdim=True)
            else:
                g = torch.as_tensor(global_stats, dtype=self.global_stats.dtype, device=self.global_stats.device)
                if g.dim() == 2:
                    g = g.unsqueeze(0)
            self.global_stats.zero_()
            self.global_stats[:, :min(g.shape[1], self.modal_num), :min(g.shape[2], self.stat_dim)].copy_(
                g[:, :self.modal_num, :self.stat_dim]
            )
            self.stats_ready = True

    def set_entity_degree_bucket(self, degree_bucket):
        degree_bucket = torch.as_tensor(degree_bucket, dtype=torch.long, device=self.entity_degree_bucket.device).flatten()
        if degree_bucket.numel() == 0:
            degree_bucket = torch.zeros(1, dtype=torch.long, device=self.entity_degree_bucket.device)
        with torch.no_grad():
            self.entity_degree_bucket = degree_bucket.clamp(min=0, max=self.degree_buckets - 1)

    def set_stat_prior(
        self,
        log_scales,
        observed_mask=None,
        quality=None,
        severe_quality=None,
        degree=None,
        init_type_bias=False,
    ):
        log_scales = torch.as_tensor(log_scales, dtype=self.prior_logits.dtype, device=self.prior_logits.device)
        rows = min(log_scales.shape[0], self.type_count)
        cols = min(log_scales.shape[1], self.modal_num)
        centered = torch.zeros_like(self.prior_logits)
        centered[:rows, :cols] = log_scales[:rows, :cols]
        centered = centered - centered.mean(dim=-1, keepdim=True)

        if observed_mask is None:
            observed = torch.zeros(self.type_count, dtype=torch.bool, device=self.prior_logits.device)
            observed[:rows] = True
        else:
            observed = torch.as_tensor(observed_mask, dtype=torch.bool, device=self.prior_logits.device).flatten()
            padded = torch.zeros(self.type_count, dtype=torch.bool, device=self.prior_logits.device)
            padded[:min(observed.numel(), self.type_count)] = observed[:self.type_count]
            observed = padded

        entity_type_id = int(getattr(self.args, "type_modality_bias_entity_type_id", -1))
        if getattr(self.args, "freeze_type_modality_bias_entity", False) and 0 <= entity_type_id < self.type_count:
            centered[entity_type_id].zero_()
            observed[entity_type_id] = False

        if observed.any():
            fallback = centered[observed].mean(dim=0)
            centered[~observed] = fallback
            if getattr(self.args, "freeze_type_modality_bias_entity", False) and 0 <= entity_type_id < self.type_count:
                centered[entity_type_id].zero_()

        quality = self._prior_vector_to_modal(quality, default=1.0)
        severe_quality = self._prior_vector_to_modal(severe_quality, default=1.0)
        degree = self._prior_vector_to_type(degree, default=0.0)
        quality_log = torch.log(quality.clamp_min(1e-6))
        quality_log = quality_log - quality_log.mean()
        severe_log = torch.log(severe_quality.clamp_min(1e-6))
        severe_log = severe_log - severe_log.mean()
        degree_centered = degree - degree[observed].mean() if observed.any() else degree - degree.mean()

        features = torch.zeros_like(self.prior_features)
        features[:, :, 0] = centered
        features[:, :, 1] = quality_log.view(1, -1).expand(self.type_count, -1)
        features[:, :, 2] = severe_log.view(1, -1).expand(self.type_count, -1)
        features[:, :, 3] = degree_centered.view(-1, 1).expand(-1, self.modal_num)

        with torch.no_grad():
            self.prior_logits.copy_(centered)
            self.prior_q.copy_(F.softmax(centered, dim=-1))
            self.prior_mask.copy_(observed)
            self.prior_features.copy_(features)
            if init_type_bias:
                self.type_bias.weight.copy_(centered)
                if getattr(self.args, "freeze_type_modality_bias_entity", False) and 0 <= entity_type_id < self.type_count:
                    self.type_bias.weight[entity_type_id].zero_()
            self.prior_ready = bool(observed.any())

    def _prior_vector_to_modal(self, values, default=0.0):
        out = torch.full((self.modal_num,), float(default), dtype=self.prior_logits.dtype, device=self.prior_logits.device)
        if values is None:
            return out
        tensor = torch.as_tensor(values, dtype=out.dtype, device=out.device).flatten()
        out[:min(tensor.numel(), self.modal_num)] = tensor[:self.modal_num]
        return out

    def _prior_vector_to_type(self, values, default=0.0):
        out = torch.full((self.type_count,), float(default), dtype=self.prior_logits.dtype, device=self.prior_logits.device)
        if values is None:
            return out
        tensor = torch.as_tensor(values, dtype=out.dtype, device=out.device).flatten()
        out[:min(tensor.numel(), self.type_count)] = tensor[:self.type_count]
        return out

    def _feature_gain(self):
        mode = getattr(self.args, "tmhg_evidence_mode", "free")
        if mode == "none":
            return self.raw_feature_gain.new_zeros(self.raw_feature_gain.shape)
        if mode == "monotonic":
            gain = F.softplus(self.raw_feature_gain)
            signs = gain.new_tensor([1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
            return signs * gain
        return self.raw_feature_gain

    def _compute_type_logits(self, device):
        stats = self.stats.to(device)
        global_stats = self.global_stats.to(device).expand_as(stats)
        count = self.type_count_buffer.to(device).view(-1, 1, 1)
        shrink = count / (count + F.softplus(self.raw_global_mix).to(device) + 1.0)
        stats = shrink * stats + (1.0 - shrink) * global_stats

        gain = self._feature_gain().to(device)
        evidence = (stats * gain.view(1, 1, -1)).sum(dim=-1)
        alpha0 = max(float(getattr(self.args, "tmhg_alpha0", 1.0)), 1e-6)
        alpha = F.softplus(evidence) + alpha0
        dirichlet_score = torch.digamma(alpha) - torch.digamma(alpha.sum(dim=-1, keepdim=True))
        dirichlet_score = dirichlet_score - dirichlet_score.mean(dim=-1, keepdim=True)

        prior_features = self.prior_features.to(device).clone()
        if not getattr(self.args, "tmhg_use_prior_logit_feature", False):
            prior_features[:, :, 0] = 0.0
        tokens = self.stat_proj(stats) + self.prior_proj(prior_features)
        if getattr(self.args, "tmhg_use_prior_logit_feature", False):
            tokens = tokens + self.prior_logit_proj(self.prior_logits.to(device).unsqueeze(-1))
        tokens = tokens + self.modal_token.to(device).unsqueeze(0)
        type_tokens = self.type_token(torch.arange(self.type_count, device=device)).unsqueeze(1)
        hidden = self.transformer(torch.cat([type_tokens, tokens], dim=1))[:, 1:]
        residual = self.residual_head(hidden).squeeze(-1)
        if getattr(self.args, "tmhg_use_moe", False):
            gate = F.softmax(self.moe_gate(hidden), dim=-1)
            expert_out = torch.stack([expert(hidden).squeeze(-1) for expert in self.moe_experts], dim=-1)
            residual = residual + (gate * expert_out).sum(dim=-1)
        residual = residual - residual.mean(dim=-1, keepdim=True)
        residual = float(getattr(self.args, "tmhg_residual_scale", 0.25)) * torch.tanh(residual)

        strength = self._type_strength(hidden[:, 0])
        prior_centered = self.prior_logits.to(device)
        prior_mode = getattr(self.args, "tmhg_prior_mode", "direct")
        learned_signal = dirichlet_score + residual
        if prior_mode == "direct":
            prior_term = strength * prior_centered
            learned_term = learned_signal
        else:
            prior_term = torch.zeros_like(prior_centered)
            learned_term = strength * learned_signal
        type_bias_term = self.type_bias.weight.to(device)
        if getattr(self.args, "tmhg_disable_type_bias", False):
            type_bias_term = torch.zeros_like(type_bias_term)
        logits = (
            prior_term
            + learned_term
            + self.global_bias.to(device).unsqueeze(0)
            + type_bias_term
        )
        logits = logits - logits.mean(dim=-1, keepdim=True)
        return logits, dirichlet_score, residual, alpha, strength

    def _compute_semantic_logits(self, modal_embs, entity_type_ids):
        device = modal_embs[0].device
        type_ids = self._safe_type_ids(entity_type_ids, device)
        if getattr(self.args, "tmhg_use_type_prototype_only", False) and self.prototype_ready:
            proto_raw_all = self.type_modal_prototypes.to(device)
            global_proto_all = self.global_modal_prototype.to(device).expand(self.type_count, -1, -1)
            min_count = max(int(getattr(self.args, "tmhg_prototype_min_count", 8)), 1)
            type_counts_all = self.prototype_type_count.to(device)
            use_global_all = (type_counts_all < float(min_count)).view(-1, 1, 1)
            proto_raw_all = torch.where(use_global_all, global_proto_all, proto_raw_all)
            proto_tokens_all = self.semantic_proj(proto_raw_all.reshape(-1, proto_raw_all.shape[-1])).view(
                self.type_count, self.modal_num, self.hidden_dim
            )
            proto_tokens_all = proto_tokens_all + self.modal_token.to(device).unsqueeze(0)
            type_token_all = self.type_token(torch.arange(self.type_count, device=device))
            proto_tokens_all = proto_tokens_all + type_token_all.unsqueeze(1)
            proto_tokens_all = proto_tokens_all + self.prototype_role_token.weight[1].view(1, 1, -1)
            proto_tokens_all = float(getattr(self.args, "tmhg_prototype_blend", 1.0)) * proto_tokens_all
            pooled_proto = proto_tokens_all.mean(dim=1, keepdim=True).expand(-1, self.modal_num, -1)
            type_feat = torch.cat([proto_tokens_all, pooled_proto], dim=-1)
            type_logits_all = self.type_prototype_head(type_feat).squeeze(-1)
            type_logits_all = type_logits_all - type_logits_all.mean(dim=-1, keepdim=True)
            gamma_max = max(float(getattr(self.args, "tmhg_gamma_max", 1.5)), 1e-6)
            gamma_all = gamma_max * torch.sigmoid(self.gamma_head(pooled_proto[:, 0])).squeeze(-1)
            logits = type_logits_all[type_ids]
            residual = logits
            alpha = F.softplus(logits) + 1.0
            gamma = gamma_all[type_ids]
            hidden_all = proto_tokens_all[type_ids]
            proto_interaction = logits
            return logits, residual, alpha, gamma, hidden_all, proto_interaction
        tokens = []
        avail_flags = []
        for emb in modal_embs[:self.modal_num]:
            emb_detached = emb.detach()
            avail = (emb_detached.abs().sum(dim=-1, keepdim=True) > 1e-8).long()
            avail_flags.append(avail)
            token = self.semantic_proj(emb_detached)
            tokens.append(token)
        token_tensor = torch.stack(tokens, dim=1)
        token_tensor = token_tensor + self.modal_token.to(device).unsqueeze(0)
        token_tensor = token_tensor + self.type_token(type_ids).unsqueeze(1)
        token_tensor = token_tensor + self.prototype_role_token.weight[0].view(1, 1, -1)
        if getattr(self.args, "tmhg_use_availability_mask", False):
            avail_tensor = torch.stack(avail_flags, dim=1).squeeze(-1)
            token_tensor = token_tensor + self.availability_token(avail_tensor)
        if getattr(self.args, "tmhg_use_degree_embedding", False) and self.entity_degree_bucket.numel() > 1:
            degree_bucket = self.entity_degree_bucket.to(device)
            degree_bucket = degree_bucket.clamp(min=0, max=self.degree_buckets - 1)
            if degree_bucket.shape[0] >= type_ids.shape[0]:
                degree_token = self.degree_token(degree_bucket[:type_ids.shape[0]])
            else:
                pad = degree_bucket.new_zeros(type_ids.shape[0])
                pad[:degree_bucket.shape[0]] = degree_bucket
                degree_token = self.degree_token(pad)
            token_tensor = token_tensor + degree_token.unsqueeze(1)
        seq_tokens = token_tensor
        proto_interaction = None
        if getattr(self.args, "tmhg_use_type_prototype", False) and self.prototype_ready:
            proto_raw = self.type_modal_prototypes.to(device)[type_ids]
            global_proto = self.global_modal_prototype.to(device).expand(type_ids.shape[0], -1, -1)
            min_count = max(int(getattr(self.args, "tmhg_prototype_min_count", 8)), 1)
            type_counts = self.prototype_type_count.to(device)[type_ids]
            use_global = (type_counts < float(min_count)).view(-1, 1, 1)
            proto_raw = torch.where(use_global, global_proto, proto_raw)
            proto_tokens = self.semantic_proj(proto_raw.reshape(-1, proto_raw.shape[-1])).view(
                type_ids.shape[0], self.modal_num, self.hidden_dim
            )
            proto_tokens = proto_tokens + self.modal_token.to(device).unsqueeze(0)
            proto_tokens = proto_tokens + self.type_token(type_ids).unsqueeze(1)
            proto_tokens = proto_tokens + self.prototype_role_token.weight[1].view(1, 1, -1)
            proto_tokens = float(getattr(self.args, "tmhg_prototype_blend", 1.0)) * proto_tokens
            if getattr(self.args, "tmhg_use_prototype_interaction", False):
                interaction_feat = torch.cat(
                    [
                        token_tensor,
                        proto_tokens,
                        token_tensor - proto_tokens,
                    ],
                    dim=-1,
                )
                proto_interaction = self.prototype_interaction_head(interaction_feat).squeeze(-1)
                proto_interaction = proto_interaction - proto_interaction.mean(dim=-1, keepdim=True)
            seq_tokens = torch.cat([token_tensor, proto_tokens], dim=1)
        hidden_all = self.transformer(seq_tokens)
        hidden = hidden_all[:, :self.modal_num]
        residual = self.residual_head(hidden).squeeze(-1)
        if getattr(self.args, "tmhg_use_moe", False):
            gate = F.softmax(self.moe_gate(hidden), dim=-1)
            expert_out = torch.stack([expert(hidden).squeeze(-1) for expert in self.moe_experts], dim=-1)
            residual = residual + (gate * expert_out).sum(dim=-1)
        residual = residual - residual.mean(dim=-1, keepdim=True)
        residual = float(getattr(self.args, "tmhg_residual_scale", 0.25)) * torch.tanh(residual)
        if proto_interaction is not None:
            residual = residual + float(getattr(self.args, "tmhg_prototype_interaction_scale", 0.25)) * torch.tanh(proto_interaction)
            residual = residual - residual.mean(dim=-1, keepdim=True)
        alpha = F.softplus(residual) + 1.0
        logits = torch.log(alpha.clamp_min(1e-12))
        logits = logits - logits.mean(dim=-1, keepdim=True)
        gamma_max = max(float(getattr(self.args, "tmhg_gamma_max", 1.5)), 1e-6)
        gamma = gamma_max * torch.sigmoid(self.gamma_head(hidden_all.mean(dim=1))).squeeze(-1)
        return logits, residual, alpha, gamma, hidden_all, proto_interaction

    def _type_strength(self, type_hidden):
        strength_max = max(float(getattr(self.args, "tmhg_strength_max", 1.5)), 1e-6)
        base = strength_max * torch.sigmoid(self.raw_type_strength)
        if not getattr(self.args, "tmhg_use_type_strength", False):
            return base.unsqueeze(-1)
        delta = torch.tanh(self.strength_head(type_hidden)).squeeze(-1)
        strength = base + 0.25 * delta
        strength = strength.clamp(min=0.0, max=strength_max)
        return strength.unsqueeze(-1)

    def regularization_loss(self):
        if self.last_bias is None:
            return None
        loss = None
        l2_weight = float(getattr(self.args, "tmhg_l2_weight", 0.0))
        if l2_weight > 0:
            scaled = l2_weight * self.last_bias.pow(2).mean()
            loss = scaled if loss is None else loss + scaled
        entropy_weight = float(getattr(self.args, "tmhg_entropy_weight", 0.0))
        if entropy_weight > 0 and self.last_q is not None:
            entropy = -(self.last_q * torch.log(self.last_q.clamp_min(1e-12))).sum(dim=-1).mean()
            scaled = -entropy_weight * entropy
            loss = scaled if loss is None else loss + scaled
        prior_kl_weight = float(getattr(self.args, "tmhg_prior_kl_weight", 0.0))
        if self.input_mode != "semantic" and prior_kl_weight > 0 and self.prior_ready and self._last_q_all_live is not None:
            q_all = self._last_q_all_live.clamp_min(1e-12)
            prior_q = self.prior_q.to(q_all.device).clamp_min(1e-12)
            kl = (prior_q * (torch.log(prior_q) - torch.log(q_all))).sum(dim=-1)
            mask = self.prior_mask.to(q_all.device)
            entity_type_id = int(getattr(self.args, "type_modality_bias_entity_type_id", -1))
            if getattr(self.args, "freeze_type_modality_bias_entity", False) and 0 <= entity_type_id < mask.numel():
                mask = mask.clone()
                mask[entity_type_id] = False
            if mask.any():
                kl_value = kl[mask].mean()
                scaled = prior_kl_weight * kl_value
                loss = scaled if loss is None else loss + scaled
                self.last_prior_kl = kl_value.detach()
        prior_logit_mse_weight = float(getattr(self.args, "tmhg_prior_logit_mse_weight", 0.0))
        if self.input_mode != "semantic" and prior_logit_mse_weight > 0 and self.prior_ready and self._last_q_all_live is not None:
            pred_logits = self._last_q_all_live.clamp_min(1e-12).log()
            pred_logits = pred_logits - pred_logits.mean(dim=-1, keepdim=True)
            target_logits = self.prior_logits.to(pred_logits.device)
            mask = self.prior_mask.to(pred_logits.device)
            entity_type_id = int(getattr(self.args, "type_modality_bias_entity_type_id", -1))
            if getattr(self.args, "freeze_type_modality_bias_entity", False) and 0 <= entity_type_id < mask.numel():
                mask = mask.clone()
                mask[entity_type_id] = False
            if mask.any():
                mse_value = F.mse_loss(pred_logits[mask], target_logits[mask])
                scaled = prior_logit_mse_weight * mse_value
                loss = scaled if loss is None else loss + scaled
        strength_l2 = float(getattr(self.args, "tmhg_strength_l2_weight", 0.0))
        if strength_l2 > 0 and self.last_type_strength is not None:
            target = float(getattr(self.args, "tmhg_strength_init", 1.0))
            scaled = strength_l2 * (self.last_type_strength - target).pow(2).mean()
            loss = scaled if loss is None else loss + scaled
        strength_reg = float(getattr(self.args, "tmhg_strength_reg_weight", 0.0))
        if strength_reg > 0 and self.last_gamma is not None:
            scaled = strength_reg * self.last_gamma.pow(2).mean()
            loss = scaled if loss is None else loss + scaled
        return loss

    def forward(self, weight_norm, entity_type_ids=None, modal_embs=None):
        if entity_type_ids is None:
            return torch.zeros_like(weight_norm)
        type_ids = self._safe_type_ids(entity_type_ids, weight_norm.device)
        if self.input_mode == "semantic":
            if modal_embs is None:
                return torch.zeros_like(weight_norm)
            logits, residual, alpha, gamma, hidden, proto_interaction = self._compute_semantic_logits(modal_embs, entity_type_ids)
            q = F.softmax(logits, dim=-1)
            bias_mode = getattr(self.args, "tmhg_bias_mode", "logit")
            if bias_mode == "logit":
                bias = gamma.unsqueeze(-1) * logits
            else:
                bias = gamma.unsqueeze(-1) * (
                    torch.log(q.clamp_min(1e-12)) - math.log(1.0 / float(self.modal_num))
                )
                bias = bias - bias.mean(dim=-1, keepdim=True)
            evidence = logits
            strength_all = gamma.unsqueeze(-1)
            q_all = q
        else:
            logits_all, evidence, residual, alpha, strength_all = self._compute_type_logits(weight_norm.device)
            q_all = F.softmax(logits_all, dim=-1)
            logits = logits_all[type_ids]
            q = q_all[type_ids]
            bias_mode = getattr(self.args, "tmhg_bias_mode", "logprob")
            if bias_mode == "logit":
                bias = float(getattr(self.args, "tmhg_gamma", 1.0)) * logits
            else:
                bias = float(getattr(self.args, "tmhg_gamma", 1.0)) * (
                    torch.log(q.clamp_min(1e-12)) - math.log(1.0 / float(self.modal_num))
                )
                bias = bias - bias.mean(dim=-1, keepdim=True)
            gamma = strength_all[type_ids].squeeze(-1)
        bias_clamp = float(getattr(self.args, "tmhg_bias_clamp", 0.6))
        if bias_clamp > 0:
            bias = bias.clamp(min=-bias_clamp, max=bias_clamp)
        generic_mask = self._generic_mask(type_ids)
        if generic_mask is not None and generic_mask.any():
            bias = bias.masked_fill(generic_mask.unsqueeze(-1), 0.0)
            q = torch.where(
                generic_mask.unsqueeze(-1),
                q.new_full(q.shape, 1.0 / float(self.modal_num)),
                q,
            )
        self.last_bias = bias.detach()
        self.last_q = q.detach()
        self.last_q_all = q_all.detach()
        self._last_q_all_live = q_all
        self.last_logits = logits.detach()
        self.last_evidence = evidence.detach()
        self.last_residual = residual.detach()
        self.last_alpha = alpha.detach()
        self.last_type_ids = type_ids.detach()
        self.last_type_strength = strength_all.detach().squeeze(-1)
        self.last_gamma = gamma.detach()
        if self.input_mode == "semantic":
            self.last_token_norm = hidden.detach().norm(dim=-1).mean(dim=-1)
            self.last_proto_interaction = None if proto_interaction is None else proto_interaction.detach()
        self.last_global_mix = F.softplus(self.raw_global_mix.detach()) + 1.0
        self.last_scale = torch.exp(bias.detach())
        if self.prior_ready and self.input_mode != "semantic":
            prior_q = self.prior_q.to(q_all.device).clamp_min(1e-12)
            kl = (prior_q * (torch.log(prior_q) - torch.log(q_all.clamp_min(1e-12)))).sum(dim=-1)
            mask = self.prior_mask.to(q_all.device)
            if mask.any():
                self.last_prior_kl = kl[mask].mean().detach()
                centered_q = q_all[mask] - q_all[mask].mean(dim=-1, keepdim=True)
                centered_prior = prior_q[mask] - prior_q[mask].mean(dim=-1, keepdim=True)
                cos = F.cosine_similarity(centered_q.flatten(), centered_prior.flatten(), dim=0, eps=1e-8)
                self.last_prior_cos = cos.detach()
        elif self.input_mode == "semantic":
            self.last_prior_kl = None
            self.last_prior_cos = None
        return bias


class MformerFusion(nn.Module):
    def __init__(self, args, modal_num, with_weight=1):
        super().__init__()
        self.args = args
        self.modal_num = modal_num
        self.fusion_layer = nn.ModuleList([BertLayer(args) for _ in range(args.num_hidden_layers)])
        self.type_id = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).cuda()  # 更新类型 ID 张量
        self.last_weight_norm = None
        self.last_weight_final = None
        self.last_cdmr_residual = None
        self.last_modal_embs = None
        self.last_hidden_states = None
        if getattr(args, "use_type_modality_bias", False):
            type_count = int(getattr(args, "top_type_count", 0) or 0)
            if type_count <= 0:
                type_count = 6
            self.type_modality_bias = nn.Embedding(type_count, modal_num)
            nn.init.zeros_(self.type_modality_bias.weight)
        else:
            self.type_modality_bias = None
        if getattr(args, "use_cdmr_router", False):
            self.cdmr_router = CDMRRouter(args, modal_num=modal_num, hidden_size=args.hidden_size)
        else:
            self.cdmr_router = None
        if getattr(args, "use_dehr_router", False):
            self.dehr_router = DEHRTypeScaleRouter(args, modal_num=modal_num, hidden_size=args.hidden_size)
        else:
            self.dehr_router = None
        if getattr(args, "use_tmhg_router", False):
            self.tmhg_router = TypeModalityHyperGate(args, modal_num=modal_num)
        else:
            self.tmhg_router = None

    def _apply_type_modality_bias(self, weight_norm, entity_type_ids, embs=None):
        if self.type_modality_bias is None and self.cdmr_router is None and self.dehr_router is None and self.tmhg_router is None:
            return weight_norm
        bias = torch.zeros_like(weight_norm)
        if self.type_modality_bias is not None and entity_type_ids is not None:
            type_ids = entity_type_ids.to(weight_norm.device).long().clamp(
                min=0,
                max=self.type_modality_bias.num_embeddings - 1,
            )
            if getattr(self.args, "freeze_type_modality_bias_entity", False):
                entity_type_id = int(getattr(self.args, "type_modality_bias_entity_type_id", -1))
                if 0 <= entity_type_id < self.type_modality_bias.num_embeddings:
                    with torch.no_grad():
                        self.type_modality_bias.weight[entity_type_id].zero_()
            if getattr(self.args, "type_modality_bias_shared", False):
                entity_type_id = int(getattr(self.args, "type_modality_bias_entity_type_id", -1))
                shared_type_ids = torch.zeros_like(type_ids)
                if getattr(self.args, "freeze_type_modality_bias_entity", False) and 0 <= entity_type_id < self.type_modality_bias.num_embeddings:
                    shared_type_ids = torch.where(type_ids == entity_type_id, type_ids, shared_type_ids)
                type_ids = shared_type_ids
            bias = self.type_modality_bias(type_ids)
        scale = float(getattr(self.args, "type_modality_bias_scale", 1.0))
        residual = torch.zeros_like(weight_norm)
        if self.cdmr_router is not None and embs is not None:
            residual = self.cdmr_router(embs, weight_norm, entity_type_ids=entity_type_ids)
        if self.dehr_router is not None and embs is not None:
            residual = residual + self.dehr_router(embs, weight_norm, entity_type_ids=entity_type_ids)
        if self.tmhg_router is not None:
            residual = residual + self.tmhg_router(weight_norm, entity_type_ids=entity_type_ids, modal_embs=embs)
        self.last_cdmr_residual = residual.detach()
        tau_route = float(getattr(self.args, "cdmr_tau_route", 1.0))
        if not self.training:
            eval_tau = float(getattr(self.args, "cdmr_eval_tau_route", -1.0))
            if eval_tau > 0:
                tau_route = eval_tau
        tau_route = max(tau_route, 1e-6)
        logits = (torch.log(weight_norm.clamp_min(1e-12)) + scale * bias + residual) / tau_route
        weight_final = F.softmax(logits, dim=-1)
        return weight_final

    def forward(self, embs, entity_type_ids=None):
        # 过滤掉 None 值，仅保留非空嵌入
        embs = [embs[idx] for idx in range(len(embs)) if embs[idx] is not None]
        # 计算有效模态数量
        modal_num = len(embs)

        # 将非空嵌入堆叠到一起，形成一个新的张量，维度为 [batch_size, modal_num, hidden_size]
        hidden_states = torch.stack(embs, dim=1)
        bs = hidden_states.shape[0]

        # 遍历每一层 BertLayer，对 hidden_states 进行处理
        for i, layer_module in enumerate(self.fusion_layer):
            layer_outputs = layer_module(hidden_states, output_attentions=True)
            hidden_states = layer_outputs[0]

        # 计算注意力权重
        attention_pro = torch.sum(layer_outputs[1], dim=-3)
        attention_pro_comb = torch.sum(attention_pro, dim=-2) / math.sqrt(modal_num * self.args.num_attention_heads)
        weight_norm = F.softmax(attention_pro_comb, dim=-1)
        weight_final = self._apply_type_modality_bias(weight_norm, entity_type_ids, embs=embs)
        self.last_weight_norm = weight_norm
        self.last_weight_final = weight_final
        self.last_modal_embs = [emb.detach() for emb in embs]
        self.last_hidden_states = hidden_states.detach()

        # 对每个模态嵌入进行加权并归一化
        embs = [weight_final[:, idx].unsqueeze(1) * F.normalize(embs[idx]) for idx in range(modal_num)]
        # 将加权后的嵌入拼接起来，形成联合嵌入
        joint_emb = torch.cat(embs, dim=1)

        return joint_emb, hidden_states, weight_final



class MultiModalEncoder(nn.Module):
    """
    entity embedding: (ent_num, input_dim)
    gcn layer: n_units

    """

    def __init__(self, args,
                 ent_num,
                 img_feature_dim,
                 char_feature_dim=None,
                 use_project_head=False,
                 attr_input_dim=1000):
        super(MultiModalEncoder, self).__init__()

        self.args = args
        attr_dim = self.args.attr_dim
        img_dim = self.args.img_dim
        name_dim = self.args.name_dim
        char_dim = self.args.char_dim
        dropout = self.args.dropout
        self.ENT_NUM = ent_num
        self.use_project_head = use_project_head

        self.n_units = [int(x) for x in self.args.hidden_units.strip().split(",")]
        self.n_heads = [int(x) for x in self.args.heads.strip().split(",")]
        self.input_dim = int(self.args.hidden_units.strip().split(",")[0])

        #########################
        # Entity Embedding
        #########################
        self.entity_emb = nn.Embedding(self.ENT_NUM, self.input_dim)
        nn.init.normal_(self.entity_emb.weight, std=1.0 / math.sqrt(self.ENT_NUM))
        self.entity_emb.requires_grad = True

        #########################
        # Modal Encoder
        #########################

        self.rel_fc = nn.Linear(1000, attr_dim)
        self.att_fc = nn.Linear(768, attr_dim)
        self.img_fc = nn.Linear(img_feature_dim, img_dim)
        self.name_fc = nn.Linear(300, char_dim)
        self.char_fc = nn.Linear(char_feature_dim, char_dim)
        # self.graph_fc = nn.Linear(self.input_dim, char_dim)

        # structure encoder
        if self.args.structure_encoder == "gcn":
            self.cross_graph_model = GCN(self.n_units[0], self.n_units[1], self.n_units[2],
                                         dropout=self.args.dropout)
        elif self.args.structure_encoder == "gat":
            self.cross_graph_model = GAT(n_units=self.n_units, n_heads=self.n_heads, dropout=args.dropout,
                                         attn_dropout=args.attn_dropout,
                                         instance_normalization=self.args.instance_normalization, diag=True)




        # 定义 GAT 层
        self.gat_rel = GAT(n_units=[attr_dim, attr_dim], n_heads=[1, 1], dropout=args.dropout,
                           attn_dropout=args.attn_dropout, instance_normalization=args.instance_normalization,
                           diag=True)
        self.gat_att = GAT(n_units=[attr_dim, attr_dim], n_heads=[1, 1], dropout=args.dropout,
                           attn_dropout=args.attn_dropout, instance_normalization=args.instance_normalization,
                           diag=True)
        self.gat_img = GAT(n_units=[img_dim, img_dim], n_heads=[1, 1], dropout=args.dropout,
                           attn_dropout=args.attn_dropout, instance_normalization=args.instance_normalization,
                           diag=True)
        self.gat_name = GAT(n_units=[char_dim, char_dim], n_heads=[1, 1], dropout=args.dropout,
                            attn_dropout=args.attn_dropout, instance_normalization=args.instance_normalization,
                            diag=True)
        self.gat_char = GAT(n_units=[char_dim, char_dim], n_heads=[1, 1], dropout=args.dropout,
                            attn_dropout=args.attn_dropout, instance_normalization=args.instance_normalization,
                            diag=True)







        #########################
        # Fusion Encoder
        #########################
        fusion_modal_num = int(bool(self.args.w_img)) + int(bool(self.args.w_attr)) + int(bool(self.args.w_rel)) + int(bool(self.args.w_gcn))
        if self.args.w_name:
            fusion_modal_num += 1
        if self.args.w_char:
            fusion_modal_num += 1
        use_sgmea_guidance = not getattr(self.args, "disable_sgmea_guidance", False)
        if use_sgmea_guidance and self.args.w_img:
            fusion_modal_num += 1
        if use_sgmea_guidance and self.args.w_attr:
            fusion_modal_num += 1
        self.entity_type_ids = None
        self.fusion = MformerFusion(args, modal_num=fusion_modal_num,
                                    with_weight=self.args.with_weight)

    def forward(self,
                input_idx,
                adj,
                img_features=None,
                rel_features=None,
                att_features=None,
                name_features=None,
                char_features=None,
                entity_type_ids=None):

        if self.args.w_gcn:
            gph_emb = self.cross_graph_model(self.entity_emb(input_idx), adj)
           
        else:
            gph_emb = None
        if self.args.w_img:
            img_emb = self.img_fc(img_features)
            if getattr(self.args, "disable_sgmea_guidance", False):
                gat_img_emb = None
            else:
                gat_img_emb = self.gat_img(img_emb, adj)
        else:
            img_emb = None
            gat_img_emb = None
        if self.args.w_rel:
            rel_emb = self.rel_fc(rel_features)
            #gat_rel_emb = self.gat_rel(rel_emb, adj)
            gat_rel_emb = None
        else:
            rel_emb = None
        if self.args.w_attr:
            att_emb = self.att_fc(att_features)
            if getattr(self.args, "disable_sgmea_guidance", False):
                gat_att_emb = None
            else:
                gat_att_emb = self.gat_att(att_emb, adj)
        else:
            att_emb = None
            gat_att_emb = None
        if self.args.w_name and name_features is not None:
            name_emb = self.name_fc(name_features)
            #gat_name_emb = self.gat_name(name_emb, adj)
            gat_name_emb = None
        else:
            name_emb = None
            gat_name_emb = None
        if self.args.w_char and char_features is not None:
            char_emb = self.char_fc(char_features)
            #gat_char_emb = self.gat_char(char_emb, adj)
            gat_char_emb = None
        else:
            char_emb = None
            gat_char_emb = None
        
        joint_emb, hidden_states, weight_norm = self.fusion(
            [img_emb, att_emb, rel_emb, gph_emb, name_emb, char_emb, gat_img_emb, gat_att_emb, gat_rel_emb, gat_name_emb, gat_char_emb],
            entity_type_ids=entity_type_ids,
        )
        return gph_emb, img_emb, rel_emb, att_emb, name_emb, char_emb, gat_img_emb, gat_att_emb, gat_rel_emb, gat_name_emb, gat_char_emb,joint_emb, hidden_states, weight_norm


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(0.1)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        # [8, 8, 3, 256]
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        # return x
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)
        # [8, 3, 8, 256]
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # # [8, 3, 8, 8]
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # [8, 3, 8, 8]
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        # [8, 8, 768]
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            output_attentions,
        )

        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        # attention: torch.Size([30355, 5, 4, 4])
        # 5: head
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN["gelu"]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.chunk_size_feed_forward = 0
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        if self.config.use_intermediate:
            self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states: torch.Tensor, output_attentions=False):
        self_attention_outputs = self.attention(
            hidden_states,
            output_attentions=output_attentions,
        )
        if not self.config.use_intermediate:
            return (self_attention_outputs[0], self_attention_outputs[1])

        attention_output = self_attention_outputs[0]
        # if decoder, the last output is tuple of self-attn cache
        outputs = self_attention_outputs[1]
        # present_key_value = self_attention_outputs[-1]
        # torch.Size([30355, 4, 300])
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output, outputs)
        # if decoder, return the attn key/values as the last output
        # outputs = outputs + (present_key_value,)
        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
        # return attention_output
