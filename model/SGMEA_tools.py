
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
        if getattr(self.args, "dehr_drop_type_token", False):
            type_token = torch.zeros_like(type_token)
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


class TypeAwareModalityRecoveryDenoiser(nn.Module):
    """Type-aware modality recovery and denoising (TMRD).

    The module is intentionally lightweight and conservative: it predicts an
    instance-level modality quality score, a semantic clean proxy from the
    contextual modality token, and optionally reads a type-conditioned clustered
    memory built from real modality features.
    """

    def __init__(self, args, modal_num, hidden_size):
        super().__init__()
        self.args = args
        self.modal_num = modal_num
        self.hidden_size = hidden_size
        type_count = int(getattr(args, "top_type_count", 0) or 0)
        if type_count <= 0:
            type_count = 6
        self.type_count = type_count
        self.memory_k = max(int(getattr(args, "tmrd_memory_k", 4)), 1)
        self.type_emb_dim = int(getattr(args, "tmrd_type_emb_dim", 16))
        self.hidden_dim = int(getattr(args, "tmrd_hidden_dim", 128))
        self.type_emb = nn.Embedding(type_count, self.type_emb_dim)
        self.modal_token = nn.Parameter(torch.zeros(modal_num, hidden_size))
        nn.init.normal_(self.modal_token, std=0.02)
        quality_input_dim = hidden_size * 5 + self.type_emb_dim + modal_num
        self.quality_head = nn.Sequential(
            nn.LayerNorm(quality_input_dim),
            nn.Linear(quality_input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )
        quality_init = min(max(float(getattr(args, "tmrd_quality_init", 0.9)), 1e-4), 1.0 - 1e-4)
        nn.init.zeros_(self.quality_head[-1].weight)
        nn.init.constant_(self.quality_head[-1].bias, math.log(quality_init / (1.0 - quality_init)))
        proxy_input_dim = hidden_size + self.type_emb_dim + modal_num
        self.proxy_head = nn.Sequential(
            nn.LayerNorm(proxy_input_dim),
            nn.Linear(proxy_input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, hidden_size),
        )
        suppress_init = min(max(float(getattr(args, "tmrd_learned_img_suppress_init", 0.08)), 1e-6), 10.0)
        self.raw_img_suppress = nn.Parameter(torch.tensor(math.log(math.exp(suppress_init) - 1.0)))
        self.recovery_query = nn.Linear(hidden_size + self.type_emb_dim, hidden_size)
        self.register_buffer("memory", torch.zeros(type_count, modal_num, self.memory_k, hidden_size))
        self.register_buffer("memory_counts", torch.zeros(type_count, modal_num))
        self.memory_ready = False
        self.last_quality = None
        self.last_quality_bias = None
        self.last_clean_delta = None
        self.last_proxy_cos = None
        self.last_memory_diversity = None
        self.last_corrupt_acc = None
        self.last_nce_loss = None
        self.last_corrupt_loss = None
        self.last_loss = None
        self.last_corrupt_pair_acc = None
        self.last_quality_gap = None
        self.last_missing_loss = None
        self.last_missing_acc = None
        self.last_missing_auc = None
        self.last_missing_gap = None
        self.last_missing_present_q = None
        self.last_missing_absent_q = None
        self.last_suppress_gate = None
        self.last_suppress_shortfall = None
        self.last_suppress_threshold = None
        self.last_missing_repair_rate = None
        self.last_present_repair_rate = None
        self.last_img_raw_coeff_missing = None
        self.last_img_repair_delta_norm = None
        self.last_learned_img_suppress = None

    def _safe_type_ids(self, entity_type_ids, device):
        if entity_type_ids is None:
            return None
        return entity_type_ids.to(device).long().clamp(min=0, max=self.type_count - 1)

    def _modal_one_hot(self, batch_size, device):
        eye = torch.eye(self.modal_num, device=device)
        return eye.view(1, self.modal_num, self.modal_num).expand(batch_size, -1, -1)

    def _context_without_self(self, tokens):
        if self.modal_num <= 1:
            return tokens
        pooled = tokens.sum(dim=1, keepdim=True) - tokens
        return pooled / float(self.modal_num - 1)

    def _predict_semantic_proxy(self, context, type_emb, modal_one_hot):
        """Predict a clean semantic proxy from other modalities only.

        This deliberately avoids an identity shortcut from the target modality.
        The residual is anchored on the cross-modal context, not on raw z_m.
        """
        proxy_in = torch.cat([context, type_emb, modal_one_hot], dim=-1)
        denoise_scale = float(getattr(self.args, "tmrd_denoise_scale", 0.25))
        proxy = context + denoise_scale * torch.tanh(self.proxy_head(proxy_in))
        return F.normalize(proxy, dim=-1, eps=1e-8)

    def _read_memory(self, context, type_ids):
        if (not self.memory_ready) or (not getattr(self.args, "tmrd_use_memory_recovery", False)):
            return None
        device = context.device
        type_emb = self.type_emb(type_ids).unsqueeze(1).expand(-1, self.modal_num, -1)
        query = self.recovery_query(torch.cat([context, type_emb], dim=-1))
        proto = self.memory.to(device)[type_ids]
        logits = torch.einsum("bmd,bmkd->bmk", F.normalize(query, dim=-1), F.normalize(proto, dim=-1))
        attn = F.softmax(logits, dim=-1)
        return torch.einsum("bmk,bmkd->bmd", attn, proto)

    def _quality_input(self, raw, proxy, memory_ref, type_emb, modal_one_hot):
        if memory_ref is None:
            memory_ref = proxy.detach()
        return torch.cat(
            [raw, proxy, torch.abs(raw - proxy), memory_ref, torch.abs(raw - memory_ref), type_emb, modal_one_hot],
            dim=-1,
        )

    def forward(self, embs, hidden_states, weight_norm, entity_type_ids=None, image_available=None):
        if entity_type_ids is None:
            return embs, torch.zeros_like(weight_norm)
        raw = torch.stack(embs[:self.modal_num], dim=1)
        device = raw.device
        type_ids = self._safe_type_ids(entity_type_ids, device)
        type_emb = self.type_emb(type_ids).unsqueeze(1).expand(-1, self.modal_num, -1)
        modal_one_hot = self._modal_one_hot(raw.shape[0], device)
        if getattr(self.args, "tmrd_context_source", "hidden") == "raw":
            context = self._context_without_self(raw)
        else:
            context = self._context_without_self(hidden_states[:, :self.modal_num])
        proxy = self._predict_semantic_proxy(context, type_emb, modal_one_hot)
        memory_proxy = self._read_memory(context, type_ids)
        if memory_proxy is not None:
            memory_mix = min(max(float(getattr(self.args, "tmrd_memory_mix", 0.0)), 0.0), 1.0)
            if memory_mix > 0:
                proxy = F.normalize((1.0 - memory_mix) * proxy + memory_mix * memory_proxy, dim=-1, eps=1e-8)
        quality_in = self._quality_input(raw, proxy, memory_proxy, type_emb, modal_one_hot)
        quality = torch.sigmoid(self.quality_head(quality_in)).squeeze(-1).clamp(min=1e-4, max=1.0)
        clean_candidate = quality.unsqueeze(-1) * raw + (1.0 - quality).unsqueeze(-1) * proxy
        clean_mix = min(max(float(getattr(self.args, "tmrd_clean_mix_scale", 0.0)), 0.0), 1.0)
        if clean_mix <= 0:
            clean = raw
        else:
            clean_gate = torch.full_like(quality, clean_mix)
            clean_modal_idx = int(getattr(self.args, "tmrd_clean_modal_idx", -1))
            if 0 <= clean_modal_idx < self.modal_num:
                modal_mask = torch.zeros_like(clean_gate)
                modal_mask[:, clean_modal_idx] = 1.0
                clean_gate = clean_gate * modal_mask
            quality_threshold = float(getattr(self.args, "tmrd_clean_quality_threshold", 0.0))
            if quality_threshold > 0:
                clean_gate = clean_gate * (quality < quality_threshold).float()
            clean = raw + clean_gate.unsqueeze(-1) * (clean_candidate - raw)
        img_repair_delta = None
        missing_repair_rate = None
        present_repair_rate = None
        img_raw_coeff_missing = None
        repair_mode = getattr(self.args, "tmrd_missing_repair_mode", "none")
        if repair_mode != "none" and self.modal_num > 0:
            img_idx = int(getattr(self.args, "tmrd_missing_modal_idx", 0))
            if 0 <= img_idx < self.modal_num:
                available = None
                if image_available is not None:
                    available = image_available.to(device).bool()
                if available is not None and available.shape[0] == raw.shape[0]:
                    proxy_img = proxy[:, img_idx, :]
                    memory_img = memory_proxy[:, img_idx, :] if memory_proxy is not None else proxy_img
                    if repair_mode == "proxy":
                        repair_img = proxy_img
                    elif repair_mode == "memory":
                        repair_img = memory_img
                    elif repair_mode == "zero":
                        repair_img = torch.zeros_like(proxy_img)
                    else:
                        mix = min(max(float(getattr(self.args, "tmrd_missing_repair_mix", 0.5)), 0.0), 1.0)
                        repair_img = F.normalize(mix * proxy_img + (1.0 - mix) * memory_img, dim=-1, eps=1e-8)
                    repaired_clean = clean.clone()
                    missing_mask = (~available).float().view(-1, 1)
                    present_mask = available.float().view(-1, 1)
                    img_before = repaired_clean[:, img_idx, :]
                    # Hard rule: unavailable image tokens have zero raw-image coefficient.
                    img_after = present_mask * img_before + missing_mask * repair_img
                    present_quantile = float(getattr(self.args, "tmrd_present_repair_quantile", 0.0))
                    present_gate = torch.zeros_like(missing_mask)
                    if present_quantile > 0 and available.any():
                        present_quantile = min(max(present_quantile, 0.0), 1.0)
                        q_img = quality[:, img_idx].detach()
                        present_q = q_img[available]
                        if present_q.numel() > 0:
                            threshold = torch.quantile(present_q, present_quantile)
                            low_present = (q_img <= threshold).float().view(-1, 1) * present_mask
                            present_mix = min(max(float(getattr(self.args, "tmrd_present_repair_mix", 0.5)), 0.0), 1.0)
                            present_gate = low_present * present_mix
                            img_after = (1.0 - present_gate) * img_after + present_gate * repair_img
                    repaired_clean[:, img_idx, :] = img_after
                    clean = repaired_clean
                    missing_repair_rate = (~available).float().mean()
                    present_repair_rate = (present_gate.squeeze(-1) > 0).float().mean()
                    img_raw_coeff_missing = torch.zeros((), device=device)
                    img_repair_delta = (img_after - raw[:, img_idx, :]).norm(dim=-1)
        quality_bias = torch.zeros_like(weight_norm)
        suppress_gate = None
        suppress_shortfall = None
        suppress_threshold_used = None
        if not getattr(self.args, "tmrd_disable_quality_bias", False):
            gamma = float(getattr(self.args, "tmrd_quality_gamma", 0.5))
            bias_mode = getattr(self.args, "tmrd_quality_bias_mode", "logq")
            if bias_mode == "centered_logit":
                # Entity-local reliability contrast: only reward/suppress a modality
                # when its clean odds deviate from the entity's average modality odds.
                odds = torch.logit(quality.clamp(min=1e-4, max=1.0 - 1e-4))
                quality_bias = gamma * (odds - odds.mean(dim=-1, keepdim=True))
            elif bias_mode == "learned_img_suppress":
                modal_idx = int(getattr(self.args, "tmrd_quality_suppress_modal_idx", 0))
                max_suppress = max(float(getattr(self.args, "tmrd_learned_img_suppress_max", 0.50)), 1e-6)
                suppress = F.softplus(self.raw_img_suppress).clamp(max=max_suppress)
                if 0 <= modal_idx < self.modal_num:
                    quality_bias[:, modal_idx] = -suppress
                self.last_learned_img_suppress = suppress.detach()
            elif bias_mode == "auto_img_suppress":
                modal_idx = int(getattr(self.args, "tmrd_quality_suppress_modal_idx", 0))
                max_suppress = max(float(getattr(self.args, "tmrd_learned_img_suppress_max", 0.50)), 1e-6)
                scale = max(float(getattr(self.args, "tmrd_auto_img_suppress_scale", 0.20)), 0.0)
                if 0 <= modal_idx < self.modal_num:
                    q_modal = quality[:, modal_idx]
                    suppress = (scale * (1.0 - q_modal.detach().mean())).clamp(max=max_suppress)
                    quality_bias[:, modal_idx] = -suppress
                    self.last_learned_img_suppress = suppress.detach()
            elif bias_mode in {"image_suppress", "image_suppress_gate", "image_suppress_targeted"}:
                modal_idx = int(getattr(self.args, "tmrd_quality_suppress_modal_idx", 0))
                threshold = float(getattr(self.args, "tmrd_quality_suppress_threshold", 0.52))
                temp = max(float(getattr(self.args, "tmrd_quality_suppress_temp", 0.10)), 1e-6)
                if 0 <= modal_idx < self.modal_num:
                    q_modal = quality[:, modal_idx]
                    quantile = float(getattr(self.args, "tmrd_quality_suppress_quantile", 0.0))
                    if quantile > 0:
                        quantile = min(max(quantile, 0.0), 1.0)
                        threshold = float(torch.quantile(q_modal.detach(), quantile).item())
                    suppress_threshold_used = torch.tensor(threshold, device=device)
                    suppress_shortfall = torch.relu(threshold - q_modal)
                    suppress_gate = torch.sigmoid((threshold - q_modal) / temp)
                    if bias_mode == "image_suppress_gate":
                        quality_bias[:, modal_idx] = -gamma * suppress_gate
                    elif bias_mode == "image_suppress_targeted":
                        quality_bias[:, modal_idx] = -gamma * suppress_gate * (suppress_shortfall > 0).float()
                    else:
                        quality_bias[:, modal_idx] = -gamma * suppress_gate * suppress_shortfall
            else:
                quality_bias = gamma * torch.log(quality.clamp_min(1e-6))
            bias_clip = float(getattr(self.args, "tmrd_quality_bias_clip", 1.0))
            if bias_clip > 0:
                if bias_mode == "centered_logit":
                    quality_bias = quality_bias.clamp(min=-bias_clip, max=bias_clip)
                else:
                    quality_bias = quality_bias.clamp(min=-bias_clip, max=0.0)
        self.last_quality = quality.detach()
        self.last_quality_bias = quality_bias.detach()
        if getattr(self.args, "tmrd_quality_bias_mode", "logq") not in {"learned_img_suppress", "auto_img_suppress"}:
            self.last_learned_img_suppress = None
        self.last_suppress_gate = suppress_gate.detach() if suppress_gate is not None else None
        self.last_suppress_shortfall = suppress_shortfall.detach() if suppress_shortfall is not None else None
        self.last_suppress_threshold = suppress_threshold_used.detach() if suppress_threshold_used is not None else None
        self.last_missing_repair_rate = missing_repair_rate.detach() if missing_repair_rate is not None else None
        self.last_present_repair_rate = present_repair_rate.detach() if present_repair_rate is not None else None
        self.last_img_raw_coeff_missing = img_raw_coeff_missing.detach() if img_raw_coeff_missing is not None else None
        self.last_img_repair_delta_norm = img_repair_delta.detach() if img_repair_delta is not None else None
        self.last_clean_delta = (clean - raw).detach()
        self.last_proxy_cos = F.cosine_similarity(
            F.normalize(proxy.detach(), dim=-1),
            F.normalize(raw.detach(), dim=-1),
            dim=-1,
            eps=1e-8,
        )
        clean_embs = [clean[:, idx, :] for idx in range(self.modal_num)]
        return clean_embs, quality_bias

    @torch.no_grad()
    def refresh_memory(self, modal_embs, entity_type_ids, entity_ids=None):
        if entity_type_ids is None:
            return {}
        device = self.memory.device
        embs = [emb.detach().to(device) for emb in modal_embs[:self.modal_num]]
        if entity_ids is None:
            entity_ids = torch.arange(embs[0].shape[0], dtype=torch.long, device=device)
        else:
            entity_ids = torch.as_tensor(entity_ids, dtype=torch.long, device=device)
        type_ids = entity_type_ids.to(device).long().clamp(min=0, max=self.type_count - 1)[entity_ids]
        iters = max(int(getattr(self.args, "tmrd_kmeans_iters", 8)), 1)
        new_memory = torch.zeros_like(self.memory)
        counts = torch.zeros_like(self.memory_counts)
        global_proto = []
        for m_idx, emb in enumerate(embs):
            values_all = emb[entity_ids]
            if values_all.shape[0] == 0:
                global_center = torch.zeros(self.memory_k, self.hidden_size, device=device, dtype=emb.dtype)
            else:
                seeds = values_all[torch.linspace(0, values_all.shape[0] - 1, steps=min(self.memory_k, values_all.shape[0]), device=device).long()]
                if seeds.shape[0] < self.memory_k:
                    seeds = torch.cat([seeds, seeds[-1:].expand(self.memory_k - seeds.shape[0], -1)], dim=0)
                global_center = seeds[:self.memory_k].clone()
            global_proto.append(global_center)
            for t_idx in range(self.type_count):
                mask = type_ids == t_idx
                values = values_all[mask]
                counts[t_idx, m_idx] = float(values.shape[0])
                if values.shape[0] == 0:
                    centers = global_center.clone()
                elif values.shape[0] < self.memory_k:
                    centers = values[torch.arange(self.memory_k, device=device) % values.shape[0]].clone()
                else:
                    init_idx = torch.linspace(0, values.shape[0] - 1, steps=self.memory_k, device=device).long()
                    centers = values[init_idx].clone()
                    for _ in range(iters):
                        sim = F.normalize(values, dim=-1) @ F.normalize(centers, dim=-1).t()
                        assign = sim.argmax(dim=1)
                        for k_idx in range(self.memory_k):
                            kmask = assign == k_idx
                            if kmask.any():
                                centers[k_idx] = values[kmask].mean(dim=0)
                new_memory[t_idx, m_idx] = centers
        self.memory.copy_(new_memory)
        self.memory_counts.copy_(counts)
        self.memory_ready = True
        diversity = self._memory_diversity().detach()
        self.last_memory_diversity = diversity
        return {
            "tmrd_memory_min_count": float(counts.min().item()) if counts.numel() else 0.0,
            "tmrd_memory_mean_count": float(counts.mean().item()) if counts.numel() else 0.0,
            "tmrd_memory_diversity": float(diversity.item()),
        }

    def _memory_diversity(self):
        if self.memory_k <= 1:
            return self.memory.new_tensor(0.0)
        proto = F.normalize(self.memory, dim=-1, eps=1e-8)
        sim = torch.matmul(proto, proto.transpose(-1, -2))
        eye = torch.eye(self.memory_k, device=sim.device, dtype=torch.bool).view(1, 1, self.memory_k, self.memory_k)
        off = sim.masked_select(~eye.expand_as(sim))
        if off.numel() == 0:
            return sim.new_tensor(0.0)
        return 1.0 - off.mean()

    @staticmethod
    def _binary_auc(scores, labels):
        pos = labels > 0.5
        neg = ~pos
        pos_count = pos.sum()
        neg_count = neg.sum()
        if pos_count == 0 or neg_count == 0:
            return None
        order = torch.argsort(scores)
        ranks = torch.empty_like(order, dtype=scores.dtype)
        ranks[order] = torch.arange(1, scores.numel() + 1, device=scores.device, dtype=scores.dtype)
        pos_rank_sum = ranks[pos].sum()
        auc = (pos_rank_sum - pos_count.to(scores.dtype) * (pos_count.to(scores.dtype) + 1.0) / 2.0) / (
            pos_count.to(scores.dtype) * neg_count.to(scores.dtype)
        )
        return auc

    def self_supervised_loss(self, embs, entity_type_ids, sample_size=2048, hidden_states=None, image_available=None):
        if entity_type_ids is None:
            return None, {}
        raw = torch.stack(embs[:self.modal_num], dim=1)
        hidden = None
        if hidden_states is not None:
            hidden = hidden_states[:, :self.modal_num]
        if getattr(self.args, "tmrd_detach_selfsup_inputs", True):
            raw = raw.detach()
            if hidden is not None:
                hidden = hidden.detach()
        device = raw.device
        type_ids_all = self._safe_type_ids(entity_type_ids, device)
        n = raw.shape[0]
        if n <= 1:
            return None, {}
        sample_size = min(max(int(sample_size), 2), n)
        idx = torch.randperm(n, device=device)[:sample_size]
        raw = raw[idx]
        if hidden is not None:
            hidden = hidden[idx]
        type_ids = type_ids_all[idx]
        available = None
        if image_available is not None:
            available = image_available.to(device).bool()[idx]
        batch_size = raw.shape[0]
        type_emb = self.type_emb(type_ids).unsqueeze(1).expand(-1, self.modal_num, -1)
        modal_one_hot = self._modal_one_hot(batch_size, device)
        if getattr(self.args, "tmrd_context_source", "hidden") == "raw" or hidden is None:
            context = self._context_without_self(raw)
        else:
            context = self._context_without_self(hidden)
        proxy = self._predict_semantic_proxy(context, type_emb, modal_one_hot)
        memory_proxy = self._read_memory(context, type_ids)
        quality_in = self._quality_input(raw, proxy, memory_proxy, type_emb, modal_one_hot)
        q_clean = torch.sigmoid(self.quality_head(quality_in)).squeeze(-1).clamp(min=1e-4, max=1.0)

        corrupt_raw = raw.clone()
        modal_choice = torch.randint(0, self.modal_num, (batch_size,), device=device)
        donor_idx = torch.empty(batch_size, dtype=torch.long, device=device)
        for row in range(batch_size):
            same = (type_ids == type_ids[row]).nonzero(as_tuple=False).flatten()
            same = same[same != row]
            if same.numel() == 0:
                donor_idx[row] = (row + 1) % batch_size
            else:
                donor_idx[row] = same[torch.randint(0, same.numel(), (1,), device=device)]
        corrupt_raw[torch.arange(batch_size, device=device), modal_choice] = raw[donor_idx, modal_choice]
        if getattr(self.args, "tmrd_context_source", "hidden") == "raw" or hidden is None:
            corrupt_context = self._context_without_self(corrupt_raw)
        else:
            corrupt_hidden = hidden.clone()
            corrupt_hidden[torch.arange(batch_size, device=device), modal_choice] = hidden[donor_idx, modal_choice]
            corrupt_context = self._context_without_self(corrupt_hidden)
        corrupt_proxy = self._predict_semantic_proxy(corrupt_context, type_emb, modal_one_hot)
        corrupt_memory_proxy = self._read_memory(corrupt_context, type_ids)
        corrupt_quality_in = self._quality_input(corrupt_raw, corrupt_proxy, corrupt_memory_proxy, type_emb, modal_one_hot)
        q_corrupt = torch.sigmoid(self.quality_head(corrupt_quality_in)).squeeze(-1).clamp(min=1e-4, max=1.0)
        chosen_clean = q_clean[torch.arange(batch_size, device=device), modal_choice]
        chosen_corrupt = q_corrupt[torch.arange(batch_size, device=device), modal_choice]
        clean_loss = F.binary_cross_entropy(chosen_clean, torch.ones_like(chosen_clean))
        corrupt_loss = F.binary_cross_entropy(chosen_corrupt, torch.zeros_like(chosen_corrupt))
        rank_margin = 0.20
        rank_loss = F.relu(rank_margin - chosen_clean + chosen_corrupt).mean()
        clean_anchor = F.mse_loss(q_clean, torch.full_like(q_clean, float(getattr(self.args, "tmrd_quality_init", 0.9))))
        clean_anchor_weight = float(getattr(self.args, "tmrd_clean_anchor_weight", 0.10))
        corrupt_loss = 0.35 * clean_loss + 0.45 * corrupt_loss + 0.20 * rank_loss + clean_anchor_weight * clean_anchor

        tau = max(float(getattr(self.args, "tmrd_nce_tau", 0.07)), 1e-6)
        nce_losses = []
        for m_idx in range(self.modal_num):
            logits = F.normalize(proxy[:, m_idx, :], dim=-1) @ F.normalize(raw[:, m_idx, :], dim=-1).t()
            logits = logits / tau
            target = torch.arange(batch_size, device=device)
            nce_losses.append(F.cross_entropy(logits, target))
        nce_loss = torch.stack(nce_losses).mean()
        total = float(getattr(self.args, "tmrd_corrupt_loss_weight", 1.0)) * corrupt_loss + float(getattr(self.args, "tmrd_nce_loss_weight", 1.0)) * nce_loss
        missing_loss = None
        missing_acc = None
        missing_auc = None
        missing_gap = None
        missing_present_q = None
        missing_absent_q = None
        missing_weight = float(getattr(self.args, "tmrd_missing_loss_weight", 0.0))
        missing_modal_idx = int(getattr(self.args, "tmrd_missing_modal_idx", 0))
        if missing_weight > 0 and available is not None and 0 <= missing_modal_idx < self.modal_num:
            img_q = q_clean[:, missing_modal_idx]
            labels = available.float()
            if labels.min() < labels.max():
                # Use real image availability as a weak diagnostic target. It trains
                # the quality head to notice distributional fake-image tokens, while
                # final inference still relies on learned quality scores rather than
                # directly masking a modality.
                missing_loss = F.binary_cross_entropy(img_q, labels)
                total = total + missing_weight * missing_loss
                missing_acc = ((img_q >= 0.5) == available).float().mean()
                missing_auc = self._binary_auc(img_q.detach(), labels.detach())
                present_q = img_q[available]
                absent_q = img_q[~available]
                missing_present_q = present_q.mean() if present_q.numel() > 0 else None
                missing_absent_q = absent_q.mean() if absent_q.numel() > 0 else None
                if missing_present_q is not None and missing_absent_q is not None:
                    missing_gap = missing_present_q - missing_absent_q
        pred_corrupt = q_corrupt[torch.arange(batch_size, device=device), modal_choice] < 0.5
        pred_clean = q_corrupt.clone()
        pred_clean[torch.arange(batch_size, device=device), modal_choice] = 1.0
        clean_ok = pred_clean > 0.5
        corrupt_acc = 0.5 * pred_corrupt.float().mean() + 0.5 * clean_ok.float().mean()
        corrupt_pair_acc = (chosen_clean > chosen_corrupt).float().mean()
        quality_gap = (chosen_clean - chosen_corrupt).mean()
        self.last_corrupt_loss = corrupt_loss.detach()
        self.last_nce_loss = nce_loss.detach()
        self.last_corrupt_acc = corrupt_acc.detach()
        self.last_corrupt_pair_acc = corrupt_pair_acc.detach()
        self.last_quality_gap = quality_gap.detach()
        self.last_missing_loss = missing_loss.detach() if missing_loss is not None else None
        self.last_missing_acc = missing_acc.detach() if missing_acc is not None else None
        self.last_missing_auc = missing_auc.detach() if missing_auc is not None else None
        self.last_missing_gap = missing_gap.detach() if missing_gap is not None else None
        self.last_missing_present_q = missing_present_q.detach() if missing_present_q is not None else None
        self.last_missing_absent_q = missing_absent_q.detach() if missing_absent_q is not None else None
        self.last_loss = total.detach()
        stats = {
            "tmrd_corrupt_loss": float(corrupt_loss.detach().item()),
            "tmrd_nce_loss": float(nce_loss.detach().item()),
            "tmrd_selfsup_loss": float(total.detach().item()),
            "tmrd_corrupt_acc_probe": float(corrupt_acc.detach().item()),
            "tmrd_corrupt_pair_acc_probe": float(corrupt_pair_acc.detach().item()),
            "tmrd_quality_gap_probe": float(quality_gap.detach().item()),
            "tmrd_q_clean_mean_probe": float(q_clean.detach().mean().item()),
            "tmrd_q_corrupt_mean_probe": float(q_corrupt.detach().mean().item()),
        }
        if missing_loss is not None:
            stats["tmrd_missing_loss"] = float(missing_loss.detach().item())
        if missing_acc is not None:
            stats["tmrd_missing_acc_probe"] = float(missing_acc.detach().item())
        if missing_auc is not None:
            stats["tmrd_missing_auc_probe"] = float(missing_auc.detach().item())
        if missing_gap is not None:
            stats["tmrd_missing_q_gap_probe"] = float(missing_gap.detach().item())
        if missing_present_q is not None:
            stats["tmrd_missing_present_q_probe"] = float(missing_present_q.detach().item())
        if missing_absent_q is not None:
            stats["tmrd_missing_absent_q_probe"] = float(missing_absent_q.detach().item())
        return total, stats

    def probe_stats(self):
        stats = {}
        if self.last_quality is not None:
            q = self.last_quality.detach()
            stats["tmrd_q_mean"] = float(q.mean().item())
            stats["tmrd_q_min"] = float(q.min().item())
            stats["tmrd_q_max"] = float(q.max().item())
            stats["tmrd_q_std"] = float(q.std().item())
        if self.last_quality_bias is not None:
            b = self.last_quality_bias.detach()
            stats["tmrd_quality_bias_abs_mean"] = float(b.abs().mean().item())
            suppress_modal_idx = int(getattr(self.args, "tmrd_quality_suppress_modal_idx", 0))
            if 0 <= suppress_modal_idx < b.shape[1]:
                stats["tmrd_quality_suppress_bias_mean"] = float(b[:, suppress_modal_idx].mean().item())
                stats["tmrd_quality_suppress_bias_abs_mean"] = float(b[:, suppress_modal_idx].abs().mean().item())
                stats["tmrd_quality_suppress_active_rate"] = float((b[:, suppress_modal_idx] < 0).float().mean().item())
        if self.last_suppress_gate is not None:
            gate = self.last_suppress_gate.detach()
            stats["tmrd_quality_suppress_gate_mean"] = float(gate.mean().item())
        if self.last_suppress_shortfall is not None:
            shortfall = self.last_suppress_shortfall.detach()
            stats["tmrd_quality_suppress_shortfall_mean"] = float(shortfall.mean().item())
        if self.last_suppress_threshold is not None:
            stats["tmrd_quality_suppress_threshold_used"] = float(self.last_suppress_threshold.detach().item())
        if self.last_clean_delta is not None:
            d = self.last_clean_delta.detach()
            stats["tmrd_delta_abs_mean"] = float(d.abs().mean().item())
            stats["tmrd_delta_norm_mean"] = float(d.norm(dim=-1).mean().item())
        if self.last_missing_repair_rate is not None:
            stats["tmrd_missing_repair_rate"] = float(self.last_missing_repair_rate.detach().item())
        if self.last_present_repair_rate is not None:
            stats["tmrd_present_repair_rate"] = float(self.last_present_repair_rate.detach().item())
        if self.last_img_raw_coeff_missing is not None:
            stats["tmrd_img_raw_coeff_missing"] = float(self.last_img_raw_coeff_missing.detach().item())
        if self.last_img_repair_delta_norm is not None:
            stats["tmrd_img_repair_delta_norm_mean"] = float(self.last_img_repair_delta_norm.detach().mean().item())
        if self.last_learned_img_suppress is not None:
            stats["tmrd_learned_img_suppress"] = float(self.last_learned_img_suppress.detach().item())
        if self.last_proxy_cos is not None:
            stats["tmrd_proxy_cos_mean"] = float(self.last_proxy_cos.detach().mean().item())
        if self.last_memory_diversity is not None:
            stats["tmrd_memory_diversity"] = float(self.last_memory_diversity.detach().item())
        if self.last_corrupt_acc is not None:
            stats["tmrd_corrupt_acc_probe"] = float(self.last_corrupt_acc.detach().item())
        if self.last_corrupt_pair_acc is not None:
            stats["tmrd_corrupt_pair_acc_probe"] = float(self.last_corrupt_pair_acc.detach().item())
        if self.last_quality_gap is not None:
            stats["tmrd_quality_gap_probe"] = float(self.last_quality_gap.detach().item())
        if self.last_nce_loss is not None:
            stats["tmrd_nce_loss_probe"] = float(self.last_nce_loss.detach().item())
        if self.last_corrupt_loss is not None:
            stats["tmrd_corrupt_loss_probe"] = float(self.last_corrupt_loss.detach().item())
        if self.last_missing_loss is not None:
            stats["tmrd_missing_loss"] = float(self.last_missing_loss.detach().item())
        if self.last_missing_acc is not None:
            stats["tmrd_missing_acc_probe"] = float(self.last_missing_acc.detach().item())
        if self.last_missing_auc is not None:
            stats["tmrd_missing_auc_probe"] = float(self.last_missing_auc.detach().item())
        if self.last_missing_gap is not None:
            stats["tmrd_missing_q_gap_probe"] = float(self.last_missing_gap.detach().item())
        if self.last_missing_present_q is not None:
            stats["tmrd_missing_present_q_probe"] = float(self.last_missing_present_q.detach().item())
        if self.last_missing_absent_q is not None:
            stats["tmrd_missing_absent_q_probe"] = float(self.last_missing_absent_q.detach().item())
        return stats


class TCMSFormerSanitizer(nn.Module):
    """Type-conditioned cross-modal visual sanitizer.

    TCMS does not output modality weights. It uses type-aware cross-modal
    interaction to repair the image token before the normal fusion stage.
    """

    def __init__(self, args, modal_num, hidden_size):
        super().__init__()
        self.args = args
        self.modal_num = modal_num
        self.hidden_size = hidden_size
        type_count = int(getattr(args, "top_type_count", 0) or 0)
        if type_count <= 0:
            type_count = 6
        self.type_count = type_count
        self.memory_k = max(int(getattr(args, "tcms_memory_k", 4)), 1)
        self.type_emb = nn.Embedding(type_count, hidden_size)
        self.avail_emb = nn.Embedding(2, hidden_size)
        self.modal_token = nn.Parameter(torch.zeros(modal_num, hidden_size))
        nn.init.normal_(self.modal_token, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=max(int(getattr(args, "tcms_heads", 2)), 1),
            dim_feedforward=hidden_size * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        tcms_layers = max(int(getattr(args, "tcms_layers", 1)), 0)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=tcms_layers) if tcms_layers > 0 else nn.Identity()
        self.anchor_norm = nn.LayerNorm(hidden_size)
        self.use_gate_match_features = not bool(getattr(args, "tcms_disable_gate_match_features", False))
        gate_dim = hidden_size * (6 if self.use_gate_match_features else 4)
        self.gate_head = nn.Sequential(
            nn.LayerNorm(gate_dim),
            nn.Linear(gate_dim, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
        )
        self.residual_head = nn.Sequential(
            nn.LayerNorm(gate_dim),
            nn.Linear(gate_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        nn.init.constant_(self.gate_head[-1].bias, float(getattr(args, "tcms_gate_init", -2.0)))
        if getattr(args, "tcms_zero_init_residual", False):
            nn.init.zeros_(self.residual_head[-1].weight)
            nn.init.zeros_(self.residual_head[-1].bias)
        self.register_buffer("memory", torch.zeros(type_count, self.memory_k, hidden_size))
        self.register_buffer("memory_counts", torch.zeros(type_count, self.memory_k))
        self.memory_ready = False
        self.last_gate = None
        self.last_clean_delta = None
        self.last_missing_rate = None
        self.last_anchor_cos = None
        self.last_sem_loss = None
        self.last_noise_loss = None
        self.last_sparse_loss = None
        self.last_loss = None
        self.last_noise_acc = None
        self.last_gate_clean = None
        self.last_gate_corrupt = None
        self.last_gate_gap = None
        self.last_cross_type_negative_rate = None
        self.last_memory_diversity = None
        self.current_epoch = 0

    def set_epoch(self, epoch):
        self.current_epoch = int(epoch)

    def _effective_beta(self):
        beta = float(getattr(self.args, "tcms_beta", 0.25))
        start = int(getattr(self.args, "tcms_beta_warmup_start", -1))
        end = int(getattr(self.args, "tcms_beta_warmup_end", -1))
        if start < 0:
            return beta
        epoch = int(getattr(self, "current_epoch", 0))
        if epoch < start:
            return 0.0
        if end <= start or epoch >= end:
            return beta
        return beta * float(epoch - start + 1) / float(max(end - start + 1, 1))

    def _safe_type_ids(self, entity_type_ids, device):
        return entity_type_ids.to(device).long().clamp(min=0, max=self.type_count - 1)

    def _encode(self, raw, type_ids, image_available=None):
        device = raw.device
        bs = raw.shape[0]
        type_token = self.type_emb(type_ids).unsqueeze(1)
        modal_tokens = raw + self.modal_token[: self.modal_num].view(1, self.modal_num, -1)
        if image_available is not None and self.modal_num > 0:
            avail_idx = image_available.to(device).long().clamp(min=0, max=1)
            modal_tokens[:, 0, :] = modal_tokens[:, 0, :] + self.avail_emb(avail_idx)
        tokens = torch.cat([type_token, modal_tokens], dim=1)
        hidden = self.encoder(tokens)
        h_type = hidden[:, 0, :]
        h_modal = hidden[:, 1:, :]
        if self.modal_num > 1:
            anchor = self.anchor_norm(h_modal[:, 1:, :].mean(dim=1) + h_type)
        else:
            anchor = self.anchor_norm(h_type)
        return hidden, h_type, h_modal, anchor

    def _read_memory(self, query, type_ids):
        proto = self.memory.to(query.device)[type_ids]
        if not self.memory_ready:
            return query
        logits = torch.einsum("bd,bkd->bk", F.normalize(query, dim=-1), F.normalize(proto, dim=-1))
        attn = F.softmax(logits, dim=-1)
        return torch.einsum("bk,bkd->bd", attn, proto)

    def _sanitize_from_parts(self, raw_img, h_img, h_type, anchor, proto, available=None):
        gate_parts = [h_img, h_type, anchor, proto]
        if self.use_gate_match_features:
            gate_parts.extend([h_img * anchor, torch.abs(h_img - anchor)])
        gate_input = torch.cat(gate_parts, dim=-1)
        gate = torch.sigmoid(self.gate_head(gate_input)).squeeze(-1)
        residual = torch.tanh(self.residual_head(gate_input))
        beta = self._effective_beta()
        clean_img = raw_img + beta * gate.unsqueeze(-1) * residual
        if not getattr(self.args, "tcms_prefusion", False):
            clean_img = F.normalize(clean_img, dim=-1, eps=1e-8)
        if available is not None:
            missing = (~available).to(raw_img.device).view(-1, 1)
            missing_mode = getattr(self.args, "tcms_missing_mode", "anchor_proto")
            if missing_mode in {"anchor", "anchor_blend"}:
                missing_fill = anchor if getattr(self.args, "tcms_prefusion", False) else F.normalize(anchor, dim=-1, eps=1e-8)
            elif missing_mode in {"proto", "proto_blend"}:
                missing_fill = proto if getattr(self.args, "tcms_prefusion", False) else F.normalize(proto, dim=-1, eps=1e-8)
            elif missing_mode == "zero":
                missing_fill = torch.zeros_like(raw_img)
            elif missing_mode == "keep":
                missing_fill = raw_img
            else:
                missing_fill = 0.5 * anchor + 0.5 * proto
                if not getattr(self.args, "tcms_prefusion", False):
                    missing_fill = F.normalize(missing_fill, dim=-1, eps=1e-8)
            if missing_mode.endswith("_blend"):
                # Conservative recovery avoids pushing explicitly-missing visual
                # tokens completely off the pretrained image-feature manifold.
                blend = min(max(float(getattr(self.args, "tcms_missing_blend", 0.25)), 0.0), 1.0)
                if float(getattr(self.args, "tcms_beta", 0.25)) > 0:
                    blend = blend * min(beta / max(float(getattr(self.args, "tcms_beta", 0.25)), 1e-12), 1.0)
                missing_fill = (1.0 - blend) * raw_img + blend * missing_fill
            clean_img = torch.where(missing, missing_fill, clean_img)
        return clean_img, gate

    def forward(self, embs, entity_type_ids=None, image_available=None):
        if entity_type_ids is None or self.modal_num <= 0:
            return embs
        raw = torch.stack(embs[: self.modal_num], dim=1)
        device = raw.device
        type_ids = self._safe_type_ids(entity_type_ids, device)
        available = image_available.to(device).bool() if image_available is not None else None
        _, h_type, h_modal, anchor = self._encode(raw, type_ids, available)
        h_img = h_modal[:, 0, :]
        proto = self._read_memory(anchor, type_ids)
        clean_img, gate = self._sanitize_from_parts(raw[:, 0, :], h_img, h_type, anchor, proto, available)
        clean = raw.clone()
        clean[:, 0, :] = clean_img
        self._ema_update(h_img.detach(), type_ids, gate.detach(), available)
        self.last_gate = gate.detach()
        self.last_clean_delta = (clean[:, 0, :] - raw[:, 0, :]).detach()
        self.last_missing_rate = ((~available).float().mean().detach() if available is not None else None)
        self.last_anchor_cos = F.cosine_similarity(clean_img.detach(), anchor.detach(), dim=-1).mean()
        return [clean[:, idx, :] for idx in range(self.modal_num)]

    @torch.no_grad()
    def _ema_update(self, h_img, type_ids, gate, available=None):
        if self.memory_k <= 0:
            return
        if available is None:
            available = torch.ones_like(gate, dtype=torch.bool)
        threshold = float(getattr(self.args, "tcms_gate_threshold", 0.35))
        valid = available.bool() & (gate < threshold)
        if not valid.any():
            return
        momentum = min(max(float(getattr(self.args, "tcms_ema_momentum", 0.99)), 0.0), 0.9999)
        for t_idx in torch.unique(type_ids[valid]):
            t_mask = valid & (type_ids == t_idx)
            values = F.normalize(h_img[t_mask], dim=-1, eps=1e-8)
            if values.numel() == 0:
                continue
            center = values.mean(dim=0)
            counts = self.memory_counts[t_idx]
            k_idx = int(torch.argmin(counts).item()) if (counts <= 0).any() else int(torch.argmin(F.normalize(self.memory[t_idx], dim=-1, eps=1e-8) @ center).item())
            if counts[k_idx] <= 0:
                self.memory[t_idx, k_idx].copy_(center)
            else:
                self.memory[t_idx, k_idx].mul_(momentum).add_(center, alpha=1.0 - momentum)
                self.memory[t_idx, k_idx].copy_(F.normalize(self.memory[t_idx, k_idx], dim=-1, eps=1e-8))
            self.memory_counts[t_idx, k_idx] += float(values.shape[0])
        self.memory_ready = bool((self.memory_counts > 0).any().item())
        self.last_memory_diversity = self._memory_diversity().detach()

    def _memory_diversity(self):
        if self.memory_k <= 1:
            return self.memory.new_tensor(0.0)
        proto = F.normalize(self.memory, dim=-1, eps=1e-8)
        sim = torch.matmul(proto, proto.transpose(-1, -2))
        eye = torch.eye(self.memory_k, device=sim.device, dtype=torch.bool).view(1, self.memory_k, self.memory_k)
        valid = self.memory_counts > 0
        pair_valid = valid.unsqueeze(-1) & valid.unsqueeze(-2) & (~eye)
        if not pair_valid.any():
            return sim.new_tensor(0.0)
        return 1.0 - sim.masked_select(pair_valid).mean()

    def _type_masked_nce(self, query, anchor, type_ids):
        tau = max(float(getattr(self.args, "tcms_nce_tau", 0.07)), 1e-6)
        logits = F.normalize(query, dim=-1) @ F.normalize(anchor.detach(), dim=-1).t()
        logits = logits / tau
        same_type = type_ids.view(-1, 1) == type_ids.view(1, -1)
        eye = torch.eye(type_ids.shape[0], dtype=torch.bool, device=type_ids.device)
        mask_out = same_type & (~eye)
        logits = logits.masked_fill(mask_out, -1e4)
        labels = torch.arange(type_ids.shape[0], device=type_ids.device)
        neg_rate = (~same_type).float().mean()
        self.last_cross_type_negative_rate = neg_rate.detach()
        return F.cross_entropy(logits, labels)

    def self_supervised_loss(self, embs, entity_type_ids, sample_size=2048, image_available=None):
        if entity_type_ids is None or self.modal_num <= 0:
            return None, {}
        raw_all = torch.stack(embs[: self.modal_num], dim=1)
        if getattr(self.args, "tmrd_detach_selfsup_inputs", True):
            raw_all = raw_all.detach()
        device = raw_all.device
        n = raw_all.shape[0]
        if n <= 1:
            return None, {}
        sample_size = min(max(int(sample_size), 2), n)
        idx = torch.randperm(n, device=device)[:sample_size]
        raw = raw_all[idx]
        type_ids_all = self._safe_type_ids(entity_type_ids, device)
        type_ids = type_ids_all[idx]
        available = image_available.to(device).bool()[idx] if image_available is not None else None
        _, h_type, h_modal, anchor = self._encode(raw, type_ids, available)
        h_img = h_modal[:, 0, :]
        proto = self._read_memory(anchor, type_ids)
        clean_img, gate_clean = self._sanitize_from_parts(raw[:, 0, :], h_img, h_type, anchor, proto, available)
        sem_loss = self._type_masked_nce(clean_img, anchor, type_ids)

        bsz = raw.shape[0]
        same_ratio = min(max(float(getattr(self.args, "tcms_noise_same_type_ratio", 0.70)), 0.0), 1.0)
        corrupt = raw.clone()
        donor_idx = torch.empty(bsz, dtype=torch.long, device=device)
        for row in range(bsz):
            use_same = torch.rand((), device=device).item() < same_ratio
            if use_same:
                candidates = (type_ids == type_ids[row]).nonzero(as_tuple=False).flatten()
                candidates = candidates[candidates != row]
            else:
                candidates = (type_ids != type_ids[row]).nonzero(as_tuple=False).flatten()
            if candidates.numel() == 0:
                candidates = torch.arange(bsz, device=device)
                candidates = candidates[candidates != row]
            donor_idx[row] = candidates[torch.randint(0, candidates.numel(), (1,), device=device)] if candidates.numel() > 0 else row
        corrupt[:, 0, :] = raw[donor_idx, 0, :]
        _, _ch_type, ch_modal, _c_anchor = self._encode(corrupt, type_ids, available)
        # Keep the semantic anchor fixed to the original non-visual context.
        # Otherwise the corrupted image can leak into self-attention and make
        # the clean/corrupt gate target ambiguous.
        _, gate_corrupt = self._sanitize_from_parts(corrupt[:, 0, :], ch_modal[:, 0, :], h_type, anchor, proto, available)
        noise_logits = torch.cat([gate_clean, gate_corrupt], dim=0)
        noise_labels = torch.cat([torch.zeros_like(gate_clean), torch.ones_like(gate_corrupt)], dim=0)
        noise_loss = F.binary_cross_entropy(noise_logits.clamp(min=1e-4, max=1.0 - 1e-4), noise_labels)
        sparse_loss = gate_clean.mean()
        total = sem_loss + float(getattr(self.args, "tcms_noise_loss_weight", 1.0)) * noise_loss + float(getattr(self.args, "tcms_sparse_weight", 0.01)) * sparse_loss
        noise_acc = ((noise_logits >= 0.5) == noise_labels.bool()).float().mean()
        self.last_sem_loss = sem_loss.detach()
        self.last_noise_loss = noise_loss.detach()
        self.last_sparse_loss = sparse_loss.detach()
        self.last_loss = total.detach()
        self.last_noise_acc = noise_acc.detach()
        self.last_gate_clean = gate_clean.detach().mean()
        self.last_gate_corrupt = gate_corrupt.detach().mean()
        self.last_gate_gap = (gate_corrupt.detach().mean() - gate_clean.detach().mean())
        self._ema_update(h_img.detach(), type_ids, gate_clean.detach(), available)
        stats = {
            "tcms_selfsup_loss": float(total.detach().item()),
            "tcms_sem_loss": float(sem_loss.detach().item()),
            "tcms_noise_loss": float(noise_loss.detach().item()),
            "tcms_sparse_loss": float(sparse_loss.detach().item()),
            "tcms_noise_acc_probe": float(noise_acc.detach().item()),
            "tcms_gate_clean_mean_probe": float(gate_clean.detach().mean().item()),
            "tcms_gate_corrupt_mean_probe": float(gate_corrupt.detach().mean().item()),
            "tcms_gate_gap_probe": float((gate_corrupt.detach().mean() - gate_clean.detach().mean()).item()),
            "tcms_cross_type_negative_rate": float(self.last_cross_type_negative_rate.detach().item()) if self.last_cross_type_negative_rate is not None else 0.0,
        }
        return total, stats

    def probe_stats(self):
        stats = {}
        if self.last_gate is not None:
            gate = self.last_gate.detach()
            stats["tcms_gate_mean"] = float(gate.mean().item())
            stats["tcms_gate_max"] = float(gate.max().item())
        if self.last_clean_delta is not None:
            delta = self.last_clean_delta.detach()
            stats["tcms_delta_norm_mean"] = float(delta.norm(dim=-1).mean().item())
            stats["tcms_delta_abs_mean"] = float(delta.abs().mean().item())
        if self.last_missing_rate is not None:
            stats["tcms_missing_rate"] = float(self.last_missing_rate.detach().item())
        if self.last_anchor_cos is not None:
            stats["tcms_anchor_cos_mean"] = float(self.last_anchor_cos.detach().item())
        if self.last_sem_loss is not None:
            stats["tcms_sem_loss_probe"] = float(self.last_sem_loss.detach().item())
        if self.last_noise_loss is not None:
            stats["tcms_noise_loss_probe"] = float(self.last_noise_loss.detach().item())
        if self.last_sparse_loss is not None:
            stats["tcms_sparse_loss_probe"] = float(self.last_sparse_loss.detach().item())
        if self.last_noise_acc is not None:
            stats["tcms_noise_acc_probe"] = float(self.last_noise_acc.detach().item())
        if self.last_gate_clean is not None:
            stats["tcms_gate_clean_mean_probe"] = float(self.last_gate_clean.detach().item())
        if self.last_gate_corrupt is not None:
            stats["tcms_gate_corrupt_mean_probe"] = float(self.last_gate_corrupt.detach().item())
        if self.last_gate_gap is not None:
            stats["tcms_gate_gap_probe"] = float(self.last_gate_gap.detach().item())
        if self.last_memory_diversity is not None:
            stats["tcms_memory_diversity"] = float(self.last_memory_diversity.detach().item())
        if self.last_loss is not None:
            stats["tcms_selfsup_loss_probe"] = float(self.last_loss.detach().item())
        return stats


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
        if getattr(args, "use_tmrd", False):
            self.tmrd = TypeAwareModalityRecoveryDenoiser(args, modal_num=modal_num, hidden_size=args.hidden_size)
        else:
            self.tmrd = None
        if getattr(args, "use_tcms", False):
            self.tcms = TCMSFormerSanitizer(args, modal_num=modal_num, hidden_size=args.hidden_size)
        else:
            self.tcms = None

    def _apply_type_modality_bias(self, weight_norm, entity_type_ids, embs=None, extra_bias=None):
        if self.type_modality_bias is None and self.cdmr_router is None and self.dehr_router is None and self.tmhg_router is None:
            if extra_bias is None:
                return weight_norm
            logits = torch.log(weight_norm.clamp_min(1e-12)) + extra_bias
            return F.softmax(logits, dim=-1)
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
        if extra_bias is not None:
            residual = residual + extra_bias
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

    def forward(self, embs, entity_type_ids=None, image_available=None):
        # 过滤掉 None 值，仅保留非空嵌入
        embs = [embs[idx] for idx in range(len(embs)) if embs[idx] is not None]
        # 计算有效模态数量
        modal_num = len(embs)

        self.last_tcms_raw_embs = [emb.detach() for emb in embs]
        if getattr(self, "tcms", None) is not None and getattr(self.args, "tcms_prefusion", False):
            embs = self.tcms(
                embs,
                entity_type_ids=entity_type_ids,
                image_available=image_available,
            )

        # 将非空嵌入堆叠到一起，形成一个新的张量，维度为 [batch_size, modal_num, hidden_size]
        hidden_states = torch.stack(embs, dim=1)
        bs = hidden_states.shape[0]

        # 遍历每一层 BertLayer，对 hidden_states 进行处理
        for i, layer_module in enumerate(self.fusion_layer):
            layer_outputs = layer_module(hidden_states, output_attentions=True)
            hidden_states = layer_outputs[0]
        if getattr(self, "tcms", None) is not None and not getattr(self.args, "tcms_prefusion", False):
            embs = self.tcms(
                embs,
                entity_type_ids=entity_type_ids,
                image_available=image_available,
            )

        # 计算注意力权重
        attention_pro = torch.sum(layer_outputs[1], dim=-3)
        attention_pro_comb = torch.sum(attention_pro, dim=-2) / math.sqrt(modal_num * self.args.num_attention_heads)
        weight_norm = F.softmax(attention_pro_comb, dim=-1)
        quality_bias = None
        if self.tmrd is not None:
            embs, quality_bias = self.tmrd(
                embs,
                hidden_states,
                weight_norm,
                entity_type_ids=entity_type_ids,
                image_available=image_available,
            )
        weight_final = self._apply_type_modality_bias(weight_norm, entity_type_ids, embs=embs, extra_bias=quality_bias)
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
                entity_type_ids=None,
                image_available=None):

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
            image_available=image_available,
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
