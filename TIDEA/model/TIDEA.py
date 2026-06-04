import types
import torch
import transformers
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import numpy as np
import pdb
import math
from .Tool_model import AutomaticWeightedLoss
from .TIDEA_tools import MultiModalEncoder
from .TIDEA_loss import CustomMultiLossLayer, icl_loss

from src.utils import pairwise_distances
import os.path as osp
import json
import math


class TIDEA(nn.Module):
    def __init__(self, kgs, args):
        super().__init__()
        self.kgs = kgs
        self.args = args
        self.img_features = F.normalize(torch.FloatTensor(kgs["images_list"])).cuda()
        self.image_available = kgs.get("image_available")
        self.input_idx = kgs["input_idx"].cuda()
        self.adj = kgs["adj"].cuda()
        self.rel_features = torch.Tensor(kgs["rel_features"]).cuda()
        self.att_features = torch.Tensor(kgs["att_features"]).cuda()
        self.left_entity_ids = kgs.get("left_entity_ids")
        self.right_entity_ids = kgs.get("right_entity_ids")
        self.entity_type_ids = kgs.get("entity_type_ids")
        self.entity_is_generic = kgs.get("entity_is_generic")
        if self.left_entity_ids is not None:
            self.left_entity_ids = self.left_entity_ids.cuda()
        if self.right_entity_ids is not None:
            self.right_entity_ids = self.right_entity_ids.cuda()
        if self.image_available is not None:
            self.image_available = self.image_available.cuda()
        if self.entity_type_ids is not None:
            self.entity_type_ids = self.entity_type_ids.cuda()
        if self.entity_is_generic is not None:
            self.entity_is_generic = self.entity_is_generic.cuda()
        self.entity_degree_bucket = None
        self.top_type_names = kgs.get("top_type_names", [])
        self.top_type_count = kgs.get("top_type_count", 0)
        self.name_features = None
        self.char_features = None
        if kgs["name_features"] is not None:
            self.name_features = kgs["name_features"].cuda()
            self.char_features = kgs["char_features"].cuda()

        img_dim = self._get_img_dim(kgs)

        char_dim = kgs["char_features"].shape[1] if self.char_features is not None else 100

        if self.top_type_count:
            self.args.top_type_count = int(self.top_type_count)
        if self.top_type_names:
            self.args.top_type_names = list(self.top_type_names)
            entity_type_name = getattr(self.args, "type_modality_bias_entity_type_name", "Entity")
            self.args.type_modality_bias_entity_type_id = self.top_type_names.index(entity_type_name) if entity_type_name in self.top_type_names else -1

        self.multimodal_encoder = MultiModalEncoder(args=self.args,
                                                    ent_num=kgs["ent_num"],
                                                    img_feature_dim=img_dim,
                                                    char_feature_dim=char_dim,
                                                    use_project_head=self.args.use_project_head,
                                                    attr_input_dim=kgs["att_features"].shape[1])
        self._init_type_modality_bias_from_json()

        self.multi_loss_layer = CustomMultiLossLayer(loss_num=11)
        self.criterion_cl = icl_loss(tau=self.args.tau, ab_weight=self.args.ab_weight, n_view=2)
        self.criterion_cl_joint = icl_loss(tau=self.args.tau, ab_weight=self.args.ab_weight, n_view=2, replay=self.args.replay, neg_cross_kg=self.args.neg_cross_kg)

        tmp = -1 * torch.ones(self.input_idx.shape[0], dtype=torch.int64).cuda()
        self.replay_matrix = torch.stack([self.input_idx, tmp], dim=1).cuda()
        self.replay_ready = 0
        self.idx_one = torch.ones(self.args.batch_size, dtype=torch.int64).cuda()
        self.idx_double = torch.cat([self.idx_one, self.idx_one]).cuda()
        self.last_num = 1000000000000
        # self.idx_one = np.ones(self.args.batch_size, dtype=np.int64)

    def set_epoch(self, epoch):
        self.current_epoch = int(epoch)
        fusion = getattr(getattr(self, "multimodal_encoder", None), "fusion", None)
        tcms = getattr(fusion, "tcms", None)
        if tcms is not None and hasattr(tcms, "set_epoch"):
            tcms.set_epoch(epoch)

    def _scheduled_tcms_weight(self, base_weight):
        base_weight = float(base_weight)
        if base_weight <= 0 or not hasattr(self, "current_epoch"):
            return base_weight
        start = int(getattr(self.args, "tcms_selfsup_start", -1))
        end = int(getattr(self.args, "tcms_selfsup_warmup_end", -1))
        if start < 0:
            return base_weight
        epoch = int(self.current_epoch)
        if epoch < start:
            return 0.0
        if end <= start or epoch >= end:
            return base_weight
        return base_weight * float(epoch - start + 1) / float(max(end - start + 1, 1))

    def _init_type_modality_bias_from_json(self):
        init_json = getattr(self.args, "type_modality_bias_init_json", "")
        if not init_json:
            return
        fusion = getattr(self.multimodal_encoder, "fusion", None)
        bias = getattr(fusion, "type_modality_bias", None)
        if bias is None:
            raise ValueError("--type_modality_bias_init_json requires --use_type_modality_bias")
        if not osp.exists(init_json):
            raise FileNotFoundError(f"type modality bias init json not found: {init_json}")
        with open(init_json, "r", encoding="utf-8") as fp:
            payload = json.load(fp)
        node = payload
        for key in getattr(self.args, "type_modality_bias_init_key", "combined_greedy.best.scales").split("."):
            if not key:
                continue
            node = node[key]

        modal_names = self._infer_type_modality_bias_modal_names()
        if len(modal_names) != bias.weight.shape[1]:
            raise ValueError(
                f"type-bias modal mismatch: modal_names={modal_names} "
                f"bias_shape={tuple(bias.weight.shape)}"
            )
        type_name_to_id = {name: idx for idx, name in enumerate(self.top_type_names)}
        with torch.no_grad():
            for type_name, scale_map in node.items():
                if type_name not in type_name_to_id:
                    continue
                type_id = type_name_to_id[type_name]
                if type_id >= bias.weight.shape[0]:
                    continue
                values = []
                for modal_name in modal_names:
                    scale = float(scale_map.get(modal_name, 1.0))
                    if scale <= 0:
                        raise ValueError(f"scale must be positive for {type_name}/{modal_name}: {scale}")
                    values.append(math.log(scale))
                bias.weight[type_id].copy_(torch.tensor(values, dtype=bias.weight.dtype, device=bias.weight.device))

            entity_type_id = int(getattr(self.args, "type_modality_bias_entity_type_id", -1))
            if getattr(self.args, "freeze_type_modality_bias_entity", False) and 0 <= entity_type_id < bias.weight.shape[0]:
                bias.weight[entity_type_id].zero_()

    def _infer_type_modality_bias_modal_names(self):
        names = []
        if self.args.w_img:
            names.append("img")
        if self.args.w_attr:
            names.append("attr")
        if self.args.w_rel:
            names.append("rel")
        if self.args.w_gcn:
            names.append("graph")
        if self.args.w_name:
            names.append("name")
        if self.args.w_char:
            names.append("char")
        if not getattr(self.args, "disable_tidea_guidance", False):
            if self.args.w_img:
                names.append("gat_img")
            if self.args.w_attr:
                names.append("gat_attr")
        return names

    def tcms_self_supervised_loss(self):
        fusion = getattr(self.multimodal_encoder, "fusion", None)
        tcms = getattr(fusion, "tcms", None)
        if tcms is None:
            return None, {}
        modal_embs = getattr(fusion, "last_tcms_raw_embs", None)
        if modal_embs is None:
            modal_embs = getattr(fusion, "last_modal_embs", None)
        if modal_embs is None:
            return None, {}
        return tcms.self_supervised_loss(
            modal_embs[:tcms.modal_num],
            self.entity_type_ids,
            sample_size=int(getattr(self.args, "tcms_loss_sample_size", 2048)),
            image_available=self.image_available,
        )

    def forward(self, batch):
        gph_emb, img_emb, rel_emb, att_emb, name_emb, char_emb, gat_img_emb, gat_att_emb, gat_rel_emb, gat_name_emb, gat_char_emb,joint_emb, hidden_states = self.joint_emb_generat(only_joint=False)


        gph_emb_hid, rel_emb_hid, att_emb_hid, img_emb_hid, name_emb_hid, char_emb_hid,gat_rel_emb_hid, gat_att_emb_hid, gat_img_emb_hid, gat_name_emb_hid, gat_char_emb_hid, joint_emb_hid = self.generate_hidden_emb(hidden_states)



        batch_tensor = batch if torch.is_tensor(batch) else torch.tensor(batch, dtype=torch.int64).cuda()

        if self.args.replay:
            batch = batch_tensor
            all_ent_batch = torch.cat([batch[:, 0], batch[:, 1]])
            if not self.replay_ready:
                loss_joi, l_neg, r_neg = self.criterion_cl_joint(
                    joint_emb,
                    batch,
                )
            else:
                neg_l = self.replay_matrix[batch[:, 0], self.idx_one[:batch.shape[0]]]
                neg_r = self.replay_matrix[batch[:, 1], self.idx_one[:batch.shape[0]]]
                neg_l_set = set(neg_l.tolist())
                neg_r_set = set(neg_r.tolist())
                all_ent_set = set(all_ent_batch.tolist())
                neg_l_list = list(neg_l_set - all_ent_set)
                neg_r_list = list(neg_r_set - all_ent_set)
                neg_l_ipt = torch.tensor(neg_l_list, dtype=torch.int64).cuda()
                neg_r_ipt = torch.tensor(neg_r_list, dtype=torch.int64).cuda()
                loss_joi, l_neg, r_neg = self.criterion_cl_joint(
                    joint_emb,
                    batch,
                    neg_l_ipt,
                    neg_r_ipt,
                )

            index = (
                all_ent_batch,
                self.idx_double[:batch.shape[0] * 2],
            )
            new_value = torch.cat([l_neg, r_neg]).cuda()

            self.replay_matrix = self.replay_matrix.index_put(index, new_value)
            if self.replay_ready == 0:
                num = torch.sum(self.replay_matrix < 0)
                if num == self.last_num:
                    self.replay_ready = 1
                    print("-----------------------------------------")
                    print("begin replay!")
                    print("-----------------------------------------")
                else:
                    self.last_num = num
        else:
            loss_joi = self.criterion_cl_joint(
                joint_emb,
                batch_tensor,
            )

        in_loss = self.inner_view_loss(gph_emb, rel_emb, att_emb, img_emb, name_emb, char_emb,  gat_rel_emb, gat_att_emb, gat_img_emb, gat_name_emb, gat_char_emb,batch_tensor)
        out_loss = self.inner_view_loss(gph_emb_hid, rel_emb_hid, att_emb_hid, img_emb_hid, name_emb_hid, gat_rel_emb_hid, gat_att_emb_hid, gat_img_emb_hid, gat_name_emb_hid, gat_char_emb_hid, char_emb_hid, batch_tensor)

        loss_all = loss_joi + in_loss + out_loss
        type_bias_l2 = None
        if getattr(self.args, "use_type_modality_bias", False) and getattr(self.args, "type_modality_bias_l2", 0.0) > 0:
            type_bias_l2 = self.type_modality_bias_regularization()
            loss_all = loss_all + float(self.args.type_modality_bias_l2) * type_bias_l2
        dehr_reg = self.dehr_router_regularization()
        if dehr_reg is not None:
            loss_all = loss_all + dehr_reg
        dehr_aux = self.dehr_router_alignment_aux_loss(joint_emb, batch_tensor)
        if dehr_aux is not None:
            loss_all = loss_all + dehr_aux
        tcms_loss = None
        tcms_stats = {}
        tcms_weight = self._scheduled_tcms_weight(getattr(self.args, "tcms_pretrain_loss_weight", 0.0))
        if getattr(self.args, "use_tcms", False) and tcms_weight > 0:
            tcms_loss, tcms_stats = self.tcms_self_supervised_loss()
            if tcms_loss is not None:
                loss_all = loss_all + tcms_weight * tcms_loss
        if getattr(self.args, "tcms_only_train", False):
            if getattr(self.args, "tcms_only_train_use_ea", False):
                pass
            else:
                if tcms_loss is None:
                    tcms_loss, tcms_stats = self.tcms_self_supervised_loss()
                if tcms_loss is None:
                    raise RuntimeError("--tcms_only_train requires a valid TCMS self-supervised loss")
                tcms_weight_eff = tcms_weight if tcms_weight > 0 else 1.0
                loss_all = tcms_weight_eff * tcms_loss
       # loss_all = loss_joi + in_loss
        loss_dic = {
            "joint_Intra_modal": loss_joi.item(),
            "Intra_modal": in_loss.item(),
            "Inter_modal": out_loss.item(),
        }
        if type_bias_l2 is not None:
            loss_dic["type_modality_bias_l2"] = float(type_bias_l2.detach().item())
            loss_dic["type_modality_bias_l2_scaled"] = float((float(self.args.type_modality_bias_l2) * type_bias_l2).detach().item())
        if dehr_reg is not None:
            loss_dic["dehr_regularization"] = float(dehr_reg.detach().item())
        if dehr_aux is not None:
            loss_dic["dehr_aux_alignment"] = float(dehr_aux.detach().item())
        if tcms_loss is not None:
            tcms_weight_eff = tcms_weight if tcms_weight > 0 else 1.0
            loss_dic["tcms_selfsup_scaled"] = float((tcms_weight_eff * tcms_loss).detach().item())
            loss_dic.update(tcms_stats)
        if getattr(self.args, "use_type_modality_bias", False) or getattr(self.args, "use_dehr_router", False) or getattr(self.args, "use_tcms", False):
            loss_dic.update(self.type_modality_bias_stats())
        output = {"loss_dic": loss_dic, "emb": joint_emb}
        return loss_all, output

    def dehr_router_regularization(self):
        fusion = getattr(self.multimodal_encoder, "fusion", None)
        router = getattr(fusion, "dehr_router", None)
        if router is None:
            return None
        return router.regularization_loss()

    def _candidate_position_map(self, side):
        attr = f"_{side}_candidate_position"
        cached = getattr(self, attr, None)
        ids = self.right_entity_ids if side == "right" else self.left_entity_ids
        if ids is None:
            return None
        if cached is None or cached.device != self.input_idx.device:
            pos = torch.full(
                (self.input_idx.shape[0],),
                -1,
                dtype=torch.long,
                device=self.input_idx.device,
            )
            pos[ids.long()] = torch.arange(ids.shape[0], dtype=torch.long, device=self.input_idx.device)
            setattr(self, attr, pos)
            cached = pos
        return cached

    def dehr_router_alignment_aux_loss(self, joint_emb, batch_tensor):
        weight = float(getattr(self.args, "dehr_aux_weight", 0.0))
        if weight <= 0 or not getattr(self.args, "use_dehr_router", False):
            return None
        scope = getattr(self.args, "dehr_aux_candidate_scope", "all_right")
        tau = max(float(getattr(self.args, "dehr_aux_tau", 0.05)), 1e-6)
        emb = F.normalize(joint_emb, dim=1)
        left = batch_tensor[:, 0].long()
        right = batch_tensor[:, 1].long()
        labels = torch.arange(left.shape[0], dtype=torch.long, device=emb.device)

        if scope == "batch":
            logits_lr = emb[left] @ emb[right].t() / tau
            logits_rl = emb[right] @ emb[left].t() / tau
            loss = 0.5 * (F.cross_entropy(logits_lr, labels) + F.cross_entropy(logits_rl, labels))
            return weight * loss

        if scope == "train_right":
            cand_right = right
            cand_left = left
            logits_lr = emb[left] @ emb[cand_right].t() / tau
            logits_rl = emb[right] @ emb[cand_left].t() / tau
            loss = 0.5 * (F.cross_entropy(logits_lr, labels) + F.cross_entropy(logits_rl, labels))
            return weight * loss

        if scope == "hard_modal":
            loss = self.dehr_router_hard_modal_aux_loss(emb, left, right, tau)
            return None if loss is None else weight * loss

        right_pos = self._candidate_position_map("right")
        left_pos = self._candidate_position_map("left")
        if right_pos is None or left_pos is None:
            return None
        target_r = right_pos[right]
        target_l = left_pos[left]
        valid_lr = target_r >= 0
        valid_rl = target_l >= 0
        losses = []
        if valid_lr.any():
            logits_lr = emb[left[valid_lr]] @ emb[self.right_entity_ids.long()].t() / tau
            losses.append(F.cross_entropy(logits_lr, target_r[valid_lr]))
        if valid_rl.any():
            logits_rl = emb[right[valid_rl]] @ emb[self.left_entity_ids.long()].t() / tau
            losses.append(F.cross_entropy(logits_rl, target_l[valid_rl]))
        if not losses:
            return None
        return weight * sum(losses) / len(losses)

    def _modal_aux_embeddings(self):
        with torch.no_grad():
            hidden_states = getattr(self.multimodal_encoder.fusion, "last_hidden_states", None)
            if hidden_states is not None and hidden_states.shape[1] >= 4:
                return [F.normalize(hidden_states[:, idx, :], dim=1) for idx in range(4)]
            if hasattr(self.multimodal_encoder.fusion, "last_modal_embs"):
                embs = getattr(self.multimodal_encoder.fusion, "last_modal_embs")
                if embs is not None:
                    return [F.normalize(emb.detach(), dim=1) for emb in embs[:4]]
            return None

    def dehr_router_hard_modal_aux_loss(self, joint_emb, left, right, tau):
        modal_embs = self._modal_aux_embeddings()
        if not modal_embs or self.left_entity_ids is None or self.right_entity_ids is None:
            return None
        k = max(int(getattr(self.args, "dehr_aux_hard_topk", 16)), 1)
        batch_size = int(left.shape[0])
        device = joint_emb.device
        losses = []
        hard_loss = getattr(self.args, "dehr_aux_hard_loss", "ce")
        target_margin = float(getattr(self.args, "dehr_aux_margin", 0.08))
        margin_temp = max(float(getattr(self.args, "dehr_aux_margin_temp", 0.02)), 1e-6)

        for modal_emb in modal_embs:
            right_bank = modal_emb[self.right_entity_ids.long()]
            left_bank = modal_emb[self.left_entity_ids.long()]
            pos_r = self._candidate_position_map("right")[right]
            pos_l = self._candidate_position_map("left")[left]

            sim_lr = modal_emb[left] @ right_bank.t()
            valid_lr = pos_r >= 0
            if valid_lr.any():
                sim_lr_valid = sim_lr[valid_lr].clone()
                row = torch.arange(sim_lr_valid.shape[0], device=device)
                sim_lr_valid[row, pos_r[valid_lr]] = -1e9
                hard_r = sim_lr_valid.topk(k=min(k, sim_lr_valid.shape[1]), dim=1).indices
                pos_ids = right[valid_lr].unsqueeze(1)
                neg_ids = self.right_entity_ids.long()[hard_r]
                cand_ids = torch.cat([pos_ids, neg_ids], dim=1)
                query_ids = left[valid_lr].unsqueeze(1)
                sim = (joint_emb[query_ids] * joint_emb[cand_ids]).sum(dim=-1)
                if hard_loss == "margin":
                    pos = sim[:, :1]
                    neg = sim[:, 1:]
                    losses.append(F.softplus((target_margin - (pos - neg)) / margin_temp).mean() * margin_temp)
                else:
                    logits = sim / tau
                    losses.append(F.cross_entropy(logits, torch.zeros(logits.shape[0], dtype=torch.long, device=device)))

            sim_rl = modal_emb[right] @ left_bank.t()
            valid_rl = pos_l >= 0
            if valid_rl.any():
                sim_rl_valid = sim_rl[valid_rl].clone()
                row = torch.arange(sim_rl_valid.shape[0], device=device)
                sim_rl_valid[row, pos_l[valid_rl]] = -1e9
                hard_l = sim_rl_valid.topk(k=min(k, sim_rl_valid.shape[1]), dim=1).indices
                pos_ids = left[valid_rl].unsqueeze(1)
                neg_ids = self.left_entity_ids.long()[hard_l]
                cand_ids = torch.cat([pos_ids, neg_ids], dim=1)
                query_ids = right[valid_rl].unsqueeze(1)
                sim = (joint_emb[query_ids] * joint_emb[cand_ids]).sum(dim=-1)
                if hard_loss == "margin":
                    pos = sim[:, :1]
                    neg = sim[:, 1:]
                    losses.append(F.softplus((target_margin - (pos - neg)) / margin_temp).mean() * margin_temp)
                else:
                    logits = sim / tau
                    losses.append(F.cross_entropy(logits, torch.zeros(logits.shape[0], dtype=torch.long, device=device)))

        if not losses:
            return None
        return sum(losses) / len(losses)

    def type_modality_bias_regularization(self):
        bias = getattr(self.multimodal_encoder.fusion, "type_modality_bias", None)
        if bias is None:
            return torch.tensor(0.0, device=self.input_idx.device)
        weight = bias.weight
        if getattr(self.args, "type_modality_bias_shared", False):
            weight = weight[:1]
            return torch.mean(weight.pow(2))
        entity_type_id = int(getattr(self.args, "type_modality_bias_entity_type_id", -1))
        if getattr(self.args, "freeze_type_modality_bias_entity", False) and 0 <= entity_type_id < weight.shape[0]:
            mask = torch.ones(weight.shape[0], dtype=torch.bool, device=weight.device)
            mask[entity_type_id] = False
            weight = weight[mask]
        return torch.mean(weight.pow(2))

    def type_modality_bias_stats(self):
        stats = {}
        fusion = self.multimodal_encoder.fusion
        bias = getattr(fusion, "type_modality_bias", None)
        with torch.no_grad():
            if bias is not None:
                weight = bias.weight.detach()
                stats["type_modality_bias_abs_mean"] = float(weight.abs().mean().item())
                stats["type_modality_bias_abs_max"] = float(weight.abs().max().item())
            weight_norm = getattr(fusion, "last_weight_norm", None)
            weight_final = getattr(fusion, "last_weight_final", None)
            if weight_norm is not None and weight_final is not None:
                delta = (weight_final - weight_norm).detach()
                stats["type_modality_weight_delta_abs_mean"] = float(delta.abs().mean().item())
                stats["type_modality_weight_delta_abs_max"] = float(delta.abs().max().item())
            dehr = getattr(fusion, "dehr_router", None)
            if dehr is not None:
                rho = getattr(dehr, "last_rho", None)
                bias_dehr = getattr(dehr, "last_bias", None)
                u = getattr(dehr, "last_u", None)
                u_free = getattr(dehr, "last_u_free", None)
                residual = getattr(dehr, "last_residual", None)
                confidence = getattr(dehr, "last_residual_confidence", None)
                drive = getattr(dehr, "last_residual_drive", None)
                alpha = getattr(dehr, "last_alpha", None)
                kl = getattr(dehr, "last_kl", None)
                if rho is not None:
                    stats["dehr_rho_mean"] = float(rho.mean().item())
                    stats["dehr_rho_max"] = float(rho.max().item())
                if bias_dehr is not None:
                    stats["dehr_bias_abs_mean"] = float(bias_dehr.abs().mean().item())
                    stats["dehr_bias_abs_max"] = float(bias_dehr.abs().max().item())
                global_bias = getattr(dehr, "last_global_bias", None)
                type_bias = getattr(dehr, "last_type_bias", None)
                if global_bias is not None:
                    stats["dehr_global_bias_abs_mean"] = float(global_bias.abs().mean().item())
                if type_bias is not None:
                    stats["dehr_type_bias_abs_mean"] = float(type_bias.abs().mean().item())
                evidence_bias = getattr(dehr, "last_evidence_bias", None)
                if evidence_bias is not None:
                    stats["dehr_evidence_bias_abs_mean"] = float(evidence_bias.abs().mean().item())
                if u is not None:
                    stats["dehr_u_abs_mean"] = float(u.abs().mean().item())
                learned_type_u = getattr(dehr, "last_learned_type_u", None)
                contextual_u = getattr(dehr, "last_contextual_u", None)
                contextual_mix = getattr(dehr, "last_contextual_mix", None)
                if learned_type_u is not None:
                    stats["dehr_learned_type_u_abs_mean"] = float(learned_type_u.abs().mean().item())
                if contextual_u is not None:
                    stats["dehr_contextual_u_abs_mean"] = float(contextual_u.abs().mean().item())
                if learned_type_u is not None and contextual_u is not None:
                    stats["dehr_contextual_delta_abs_mean"] = float((contextual_u - learned_type_u).abs().mean().item())
                if contextual_mix is not None:
                    stats["dehr_contextual_mix"] = float(contextual_mix)
                if u_free is not None:
                    stats["dehr_u_free_abs_mean"] = float(u_free.abs().mean().item())
                if residual is not None:
                    stats["dehr_residual_abs_mean"] = float(residual.abs().mean().item())
                    stats["dehr_residual_abs_max"] = float(residual.abs().max().item())
                if confidence is not None:
                    stats["dehr_residual_confidence_mean"] = float(confidence.mean().item())
                if drive is not None:
                    stats["dehr_residual_drive_mean"] = float(drive.mean().item())
                    stats["dehr_residual_drive_abs_mean"] = float(drive.abs().mean().item())
                if alpha is not None:
                    stats["dehr_alpha_mean"] = float(alpha.detach().mean().item())
                if kl is not None:
                    stats["dehr_kl_to_uniform"] = float(kl.detach().item())
            tcms = getattr(fusion, "tcms", None)
            if tcms is not None:
                stats.update(tcms.probe_stats())
        return stats

    def estimate_train_stat_dehr_direction(self, train_links, logger=None):
        fusion = getattr(self.multimodal_encoder, "fusion", None)
        router = getattr(fusion, "dehr_router", None)
        if router is None:
            return None
        metric = getattr(self.args, "dehr_stat_direction_metric", "hybrid")
        tau = float(getattr(self.args, "dehr_stat_margin_tau", 0.05))
        scale = float(getattr(self.args, "dehr_stat_direction_scale", 1.0))
        normalize = getattr(self.args, "dehr_stat_direction_normalize", "none")
        with torch.no_grad():
            (
                gph_emb,
                img_emb,
                rel_emb,
                att_emb,
                _name_emb,
                _char_emb,
                _gat_img_emb,
                _gat_att_emb,
                _gat_rel_emb,
                _gat_name_emb,
                _gat_char_emb,
                _joint_emb,
                _hidden_states,
            ) = self.joint_emb_generat(only_joint=False)
            modal_items = [
                ("img", img_emb),
                ("attr", att_emb),
                ("rel", rel_emb),
                ("graph", gph_emb),
            ]
            train_tensor = torch.as_tensor(train_links, dtype=torch.long, device=self.input_idx.device)
            left_ids = train_tensor[:, 0]
            right_ids = train_tensor[:, 1]
            scores = []
            rows = []
            for name, emb in modal_items[:router.modal_num]:
                if emb is None:
                    score = torch.tensor(0.0, device=self.input_idx.device)
                    hit = torch.tensor(0.0, device=self.input_idx.device)
                    margin_score = torch.tensor(0.0, device=self.input_idx.device)
                    margin = torch.tensor(0.0, device=self.input_idx.device)
                else:
                    emb = F.normalize(emb, dim=-1)
                    sim = torch.mm(emb[left_ids], emb[right_ids].t())
                    diag = sim.diag()
                    masked = sim.masked_fill(torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device), -1e9)
                    hardest = masked.max(dim=1).values
                    margin = diag - hardest
                    hit = (sim.argmax(dim=1) == torch.arange(sim.shape[0], device=sim.device)).float().mean()
                    margin_score = torch.tanh(margin / max(tau, 1e-6)).mean()
                    if metric == "hit":
                        score = hit
                    elif metric == "margin":
                        score = margin_score
                    else:
                        score = 0.5 * hit + 0.5 * margin_score
                scores.append(score)
                rows.append(
                    {
                        "modal": name,
                        "hit": float(hit.detach().item()),
                        "margin_mean": float(margin.detach().mean().item()),
                        "margin_score": float(margin_score.detach().item()),
                        "score": float(score.detach().item()),
                    }
                )
            score_tensor = torch.stack(scores)
            direction = score_tensor - score_tensor.mean()
            if normalize == "max_abs":
                direction = direction / direction.abs().max().clamp_min(1e-12)
            elif normalize == "mean_abs":
                direction = direction / direction.abs().mean().clamp_min(1e-12)
            direction = direction * scale
            router.set_reliability_direction(direction)
            self.args.dehr_direction_values = ",".join(f"{float(x):.6f}" for x in direction.detach().cpu().tolist())
            if logger is not None:
                stat_line = " || ".join(
                    (
                        f"{row['modal']}:hit={row['hit']:.4f},"
                        f"margin={row['margin_mean']:.4f},"
                        f"mscore={row['margin_score']:.4f},score={row['score']:.4f}"
                    )
                    for row in rows
                )
                logger.info(
                    "DEHRStatDirection | "
                    f"metric={metric} tau={tau} scale={scale} normalize={normalize} "
                    f"direction={self.args.dehr_direction_values} stats={stat_line}"
                )
            return direction.detach().cpu(), rows

    def init_dehr_direction_from_stat_prior_json(self, logger=None):
        fusion = getattr(self.multimodal_encoder, "fusion", None)
        router = getattr(fusion, "dehr_router", None)
        if router is None:
            return None
        prior_json = getattr(self.args, "dehr_stat_prior_json", "")
        if not prior_json:
            raise ValueError("--dehr_direction_source stat_prior_json requires --dehr_stat_prior_json")
        if not osp.exists(prior_json):
            raise FileNotFoundError(f"DEHR stat prior json not found: {prior_json}")
        with open(prior_json, "r", encoding="utf-8") as fp:
            payload = json.load(fp)
        node = payload
        for key in getattr(self.args, "dehr_stat_prior_key", "combined_greedy.best.scales").split("."):
            if key:
                node = node[key]

        modal_names = self._infer_type_modality_bias_modal_names()[:router.modal_num]
        type_name_to_id = {name: idx for idx, name in enumerate(self.top_type_names)}
        log_scales = torch.zeros(
            (router.type_count, router.modal_num),
            dtype=torch.float,
            device=self.input_idx.device,
        )
        observed = torch.zeros(router.type_count, dtype=torch.bool, device=self.input_idx.device)
        rows = []
        for type_name, scale_map in node.items():
            if type_name not in type_name_to_id:
                continue
            type_id = type_name_to_id[type_name]
            if type_id >= router.type_count:
                continue
            values = []
            for modal_name in modal_names:
                scale = float(scale_map.get(modal_name, 1.0))
                if scale <= 0:
                    raise ValueError(f"DEHR stat prior scale must be positive for {type_name}/{modal_name}: {scale}")
                values.append(math.log(scale))
            row = torch.tensor(values, dtype=torch.float, device=self.input_idx.device)
            row = row - row.mean()
            log_scales[type_id, : row.numel()] = row
            observed[type_id] = True
            rows.append((type_name, [float(x) for x in row.detach().cpu().tolist()]))

        if observed.any() and getattr(self.args, "dehr_stat_prior_global", "mean") == "mean":
            global_direction = log_scales[observed].mean(dim=0)
        else:
            global_direction = log_scales.new_zeros(router.modal_num)
        global_direction = global_direction - global_direction.mean()
        scale = float(getattr(self.args, "dehr_stat_direction_scale", 1.0))
        normalize = getattr(self.args, "dehr_stat_direction_normalize", "none")
        if normalize == "max_abs":
            global_direction = global_direction / global_direction.abs().max().clamp_min(1e-12)
        elif normalize == "mean_abs":
            global_direction = global_direction / global_direction.abs().mean().clamp_min(1e-12)
        global_direction = global_direction * scale
        router.set_reliability_direction(global_direction)
        self.args.dehr_direction_values = ",".join(f"{float(x):.6f}" for x in global_direction.detach().cpu().tolist())

        residual = log_scales - global_direction.unsqueeze(0)
        residual[~observed] = 0.0
        residual = residual - residual.mean(dim=-1, keepdim=True)
        if getattr(self.args, "dehr_stat_prior_init_type_residual", False):
            router.set_type_direction_residual(residual)

        if logger is not None:
            residual_mean = float(residual[observed].abs().mean().item()) if observed.any() else 0.0
            row_text = " || ".join(
                f"{type_name}=" + ",".join(f"{value:+.4f}" for value in values)
                for type_name, values in rows
            )
            logger.info(
                "DEHRStatPriorDirection | "
                f"json={prior_json} key={getattr(self.args, 'dehr_stat_prior_key', '')} "
                f"global={self.args.dehr_direction_values} residual_abs_mean={residual_mean:.4f} "
                f"init_type_residual={getattr(self.args, 'dehr_stat_prior_init_type_residual', False)} "
                f"rows={row_text}"
            )
        return global_direction.detach().cpu(), residual.detach().cpu()

    def generate_hidden_emb(self, hidden):


        gph_emb = F.normalize(hidden[:, 0, :].squeeze(1))
        rel_emb = F.normalize(hidden[:, 1, :].squeeze(1))
        att_emb = F.normalize(hidden[:, 2, :].squeeze(1))
        img_emb = F.normalize(hidden[:, 3, :].squeeze(1))
        #gat_rel_emb = F.normalize(hidden[:, 4, :].squeeze(1))
        #gat_att_emb = F.normalize(hidden[:, 5, :].squeeze(1))
        #gat_img_emb = F.normalize(hidden[:, 6, :].squeeze(1))
        gat_rel_emb, gat_att_emb,gat_img_emb =  None, None, None
        #gat_img_emb =  None
        if hidden.shape[1] >= 99:
            name_emb = F.normalize(hidden[:, 7, :].squeeze(1))
            char_emb = F.normalize(hidden[:, 8, :].squeeze(1))

            gat_name_emb = F.normalize(hidden[:, 9, :].squeeze(1))
            gat_char_emb = F.normalize(hidden[:, 10, :].squeeze(1))
            joint_emb = torch.cat([gph_emb, rel_emb, att_emb, img_emb, name_emb, char_emb, gat_rel_emb, gat_att_emb, gat_img_emb, gat_name_emb, gat_char_emb], dim=1)
        else:
            name_emb, char_emb = None, None
            gat_name_emb, gat_char_emb = None, None
            loss_name, loss_char = None, None
            joint_emb = torch.cat([gph_emb, rel_emb, att_emb, img_emb], dim=1)

        return gph_emb, rel_emb, att_emb, img_emb, name_emb, char_emb, gat_rel_emb, gat_att_emb, gat_img_emb, gat_name_emb, gat_char_emb,joint_emb

    def inner_view_loss(self, gph_emb, rel_emb, att_emb, img_emb, name_emb, char_emb , gat_rel_emb, gat_att_emb, gat_img_emb, gat_name_emb, gat_char_emb,train_ill):
        # pdb.set_trace()
        loss_GCN = self.criterion_cl(gph_emb, train_ill) if gph_emb is not None else 0
        loss_rel = self.criterion_cl(rel_emb, train_ill) if rel_emb is not None else 0
        loss_att = self.criterion_cl(att_emb, train_ill) if att_emb is not None else 0
        loss_img = self.criterion_cl(img_emb, train_ill) if img_emb is not None else 0
        loss_name = self.criterion_cl(name_emb, train_ill) if name_emb is not None else 0
        loss_char = self.criterion_cl(char_emb, train_ill) if char_emb is not None else 0

        # 计算新增的损失
        loss_gat_rel = self.criterion_cl(gat_rel_emb, train_ill) if gat_rel_emb is not None else 0
        loss_gat_att = self.criterion_cl(gat_att_emb, train_ill) if gat_att_emb is not None else 0
        loss_gat_img = self.criterion_cl(gat_img_emb, train_ill) if gat_img_emb is not None else 0
        loss_gat_name = self.criterion_cl(gat_name_emb, train_ill) if gat_name_emb is not None else 0
        loss_gat_char = self.criterion_cl(gat_char_emb, train_ill) if gat_char_emb is not None else 0

        total_loss = self.multi_loss_layer([
            loss_GCN, loss_rel, loss_att, loss_img, loss_name, loss_char,
            loss_gat_rel, loss_gat_att, loss_gat_img, loss_gat_name, loss_gat_char
        ])

        return total_loss

    # --------- necessary ---------------

    def joint_emb_generat(self, only_joint=True):
        gph_emb, img_emb, rel_emb, att_emb, \
            name_emb, char_emb, gat_img_emb, gat_att_emb, gat_rel_emb, gat_name_emb, gat_char_emb,joint_emb, hidden_states, weight_norm \
            = self.multimodal_encoder(self.input_idx,
                                                                                                self.adj,
                                                                                                self.img_features,
                                                                                                self.rel_features,
                                                                                                self.att_features,
                                                                                                self.name_features,
                                                                                                self.char_features,
                                                                                                entity_type_ids=self.entity_type_ids,
                                                                                                image_available=self.image_available)
        if only_joint:
            return joint_emb, weight_norm
        else:
            return gph_emb, img_emb, rel_emb, att_emb, name_emb, char_emb, gat_img_emb, gat_att_emb, gat_rel_emb, gat_name_emb, gat_char_emb, joint_emb, hidden_states

    # --------- share ---------------

    def _get_img_dim(self, kgs):
        if isinstance(kgs["images_list"], list):
            img_dim = kgs["images_list"][0].shape[1]
        elif isinstance(kgs["images_list"], np.ndarray) or torch.is_tensor(kgs["images_list"]):
            img_dim = kgs["images_list"].shape[1]
        return img_dim

    def Iter_new_links(self, epoch, left_non_train, final_emb, right_non_train, new_links=[]):
        if len(left_non_train) == 0 or len(right_non_train) == 0:
            return new_links
        distance_list = []
        for i in np.arange(0, len(left_non_train), 1000):
            d = pairwise_distances(final_emb[left_non_train[i:i + 1000]], final_emb[right_non_train])
            distance_list.append(d)
        distance = torch.cat(distance_list, dim=0)
        preds_l = torch.argmin(distance, dim=1).cpu().numpy().tolist()
        preds_r = torch.argmin(distance.t(), dim=1).cpu().numpy().tolist()
        del distance_list, distance, final_emb
        if (epoch + 1) % (self.args.semi_learn_step * 5) == self.args.semi_learn_step:
            new_links = [(left_non_train[i], right_non_train[p]) for i, p in enumerate(preds_l) if preds_r[p] == i]
        else:
            new_links = [(left_non_train[i], right_non_train[p]) for i, p in enumerate(preds_l) if (preds_r[p] == i) and ((left_non_train[i], right_non_train[p]) in new_links)]

        return new_links

    def data_refresh(self, logger, train_ill, test_ill_, left_non_train, right_non_train, new_links=[]):
        if len(new_links) != 0 and (len(left_non_train) != 0 and len(right_non_train) != 0):
            new_links_select = new_links
            train_ill = np.vstack((train_ill, np.array(new_links_select)))
            num_true = len([nl for nl in new_links_select if nl in test_ill_])
            # remove from left/right_non_train
            for nl in new_links_select:
                left_non_train.remove(nl[0])
                right_non_train.remove(nl[1])

            if self.args.rank == 0:
                logger.info(f"#new_links_select:{len(new_links_select)}")
                logger.info(f"train_ill.shape:{train_ill.shape}")
                logger.info(f"#true_links: {num_true}")
                logger.info(f"true link ratio: {(100 * num_true / len(new_links_select)):.1f}%")
                logger.info(f"#entity not in train set: {len(left_non_train)} (left) {len(right_non_train)} (right)")

            new_links = []
        else:
            logger.info("len(new_links) is 0")

        return left_non_train, right_non_train, train_ill, new_links
