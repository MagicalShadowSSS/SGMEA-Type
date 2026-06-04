import os
import os.path as osp
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler
from torch.cuda.amp import GradScaler, autocast
from datetime import datetime
from easydict import EasyDict as edict
from tqdm import tqdm
import json
import pickle
from collections import defaultdict

from config import cfg
from torchlight import initialize_exp, set_seed, get_dump_path
from src.data import load_data, Collator_base, EADataset
from src.utils import set_optim, Loss_log, pairwise_distances, csls_sim
from model import TIDEA
from src.distributed_utils import init_distributed_mode, dist_pdb, is_main_process, reduce_value, cleanup
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import torch.nn.functional as F
import scipy
import gc
import copy


class Runner:
    def __init__(self, args, writer=None, logger=None, rank=0):
        self.datapath = edict()
        self.datapath.log_dir = get_dump_path(args)
        self.datapath.model_dir = os.path.join(self.datapath.log_dir, 'model')
        self.rank = rank
        self.args = args
        self.writer = writer
        self.logger = logger
        self.scaler = GradScaler()
        self.model_list = []
        set_seed(args.random_seed)
        self.data_init()
        self.model_choise()
        set_seed(args.random_seed)

        if self.args.only_test:
            self.dataloader_init(test_set=self.test_set)
        else:
            self.dataloader_init(train_set=self.train_set, eval_set=self.eval_set, test_set=self.test_set)
            if self.args.dist:
                self.model_sync()
            else:
                self.model_list = [self.model]
            if self.args.il:
                assert self.args.il_start < self.args.epoch
                train_epoch_1_stage = self.args.il_start
            else:
                train_epoch_1_stage = self.args.epoch
            self.optim_init(self.args, total_epoch=train_epoch_1_stage)

    def _get_dehr_router(self):
        if self.args.model_name != "TIDEA":
            return None
        fusion = getattr(getattr(self.model, "multimodal_encoder", None), "fusion", None)
        if fusion is None:
            return None
        return getattr(fusion, "dehr_router", None)

    def model_sync(self):
        folder = osp.join(self.args.data_path, "tmp")
        if not os.path.exists(folder):
            os.makedirs(folder)
        checkpoint_path = osp.join(folder, "initial_weights.pt")
        if self.rank == 0:
            torch.save(self.model.state_dict(), checkpoint_path)
        dist.barrier()
        self.model = self._model_sync(self.model, checkpoint_path)

    def _model_sync(self, model, checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=self.args.device))
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(self.args.device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.args.gpu], find_unused_parameters=True)
        self.model_list.append(model)
        model = model.module
        return model
    def model_choise(self):
        assert self.args.model_name in ["EVA", "MCLEA", "MSNEA", "TIDEA"]
        if self.args.model_name == "TIDEA":
            self.model = TIDEA(self.KGs, self.args)

        self.model = self._load_model(self.model, model_name=self.args.model_name_save)
        if (
            self.args.model_name == "TIDEA"
            and getattr(self.args, "use_dehr_router", False)
            and getattr(self.args, "dehr_direction_source", "manual") == "stat_train"
        ):
            self.model.estimate_train_stat_dehr_direction(self.train_ill, logger=self.logger)
        if (
            self.args.model_name == "TIDEA"
            and getattr(self.args, "use_dehr_router", False)
            and getattr(self.args, "dehr_direction_source", "manual") == "stat_prior_json"
        ):
            self.model.init_dehr_direction_from_stat_prior_json(logger=self.logger)

        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"total params num: {total_params}")
    def optim_init(self, opt, total_step=None, total_epoch=None, accumulation_step=None):
        step_per_epoch = len(self.train_dataloader)
        if total_epoch is not None:
            opt.total_steps = int(step_per_epoch * total_epoch)
        else:
            opt.total_steps = int(step_per_epoch * opt.epoch) if total_step is None else int(total_step)
        opt.warmup_steps = int(opt.total_steps * 0.15)

        if self.rank == 0 and total_step is None:
            self.logger.info(f"warmup_steps: {opt.warmup_steps}")
            self.logger.info(f"total_steps: {opt.total_steps}")
            self.logger.info(f"weight_decay: {opt.weight_decay}")
        if getattr(opt, "tcms_only_train", False):
            self.optimizer, self.scheduler = self._optim_init_tcms_only(opt, accumulation_step)
            return
        if getattr(opt, "type_modality_bias_only_train", False):
            self.optimizer, self.scheduler = self._optim_init_type_modality_bias_only(opt, accumulation_step)
            return
        freeze_part = []

        self.optimizer, self.scheduler = set_optim(opt, self.model_list, freeze_part, accumulation_step)

    def _optim_init_tcms_only(self, opt, accumulation_step=None):
        for _, p in self.model.named_parameters():
            p.requires_grad = False
        fusion = getattr(getattr(self.model, "multimodal_encoder", None), "fusion", None)
        tcms = getattr(fusion, "tcms", None)
        if getattr(opt, "tcms_only_train", False) and tcms is None:
            raise ValueError("--tcms_only_train requires --use_tcms")
        train_params = []
        for p in tcms.parameters():
            p.requires_grad = True
            train_params.append(p)
        if getattr(opt, "tcms_train_fusion_layer", False):
            for p in fusion.fusion_layer.parameters():
                p.requires_grad = True
                train_params.append(p)
        params = [{"params": train_params, "lr": opt.lr, "weight_decay": opt.weight_decay}]
        optimizer = torch.optim.AdamW(params, lr=opt.lr, eps=opt.adam_epsilon)
        if accumulation_step is None:
            accumulation_step = opt.accumulation_steps
        if opt.scheduler == 'fixed':
            from src.utils import FixedScheduler
            scheduler = FixedScheduler(optimizer)
        elif opt.scheduler == 'linear':
            from transformers import get_linear_schedule_with_warmup
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(opt.warmup_steps / accumulation_step),
                num_training_steps=int(opt.total_steps / accumulation_step),
            )
        elif opt.scheduler == 'cos':
            from transformers import get_cosine_schedule_with_warmup
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(opt.warmup_steps / accumulation_step),
                num_training_steps=int(opt.total_steps / accumulation_step),
            )
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        if self.rank == 0:
            self.logger.info(
                f"TCMSOnlyTrain | trainable_params={trainable_params} "
                f"lr={opt.lr} weight_decay={opt.weight_decay} "
                f"tcms_pretrain_weight={getattr(opt, 'tcms_pretrain_loss_weight', 0.0)}"
            )
        return optimizer, scheduler

    def _optim_init_type_modality_bias_only(self, opt, accumulation_step=None):
        for _, p in self.model.named_parameters():
            p.requires_grad = False
        fusion = self.model.multimodal_encoder.fusion
        bias = getattr(self.model.multimodal_encoder.fusion, "type_modality_bias", None)
        dehr = getattr(fusion, "dehr_router", None)
        if bias is None and dehr is None:
            raise ValueError("--type_modality_bias_only_train requires a type/modality router")
        train_params = []
        entity_type_id = int(getattr(opt, "type_modality_bias_entity_type_id", -1))
        if bias is not None:
            bias.weight.requires_grad = True
            if getattr(opt, "freeze_type_modality_bias_entity", False) and 0 <= entity_type_id < bias.weight.shape[0]:
                def _zero_entity_grad(grad):
                    grad = grad.clone()
                    grad[entity_type_id].zero_()
                    return grad
                bias.weight.register_hook(_zero_entity_grad)
            if getattr(opt, "type_modality_bias_shared", False):
                def _shared_bias_grad(grad):
                    grad = grad.clone()
                    keep_rows = [0]
                    if getattr(opt, "freeze_type_modality_bias_entity", False) and 0 <= entity_type_id < grad.shape[0]:
                        keep_rows.append(entity_type_id)
                    mask = torch.ones(grad.shape[0], dtype=torch.bool, device=grad.device)
                    mask[torch.tensor(keep_rows, dtype=torch.long, device=grad.device)] = False
                    grad[mask] = 0
                    return grad
                bias.weight.register_hook(_shared_bias_grad)
            train_params.append(bias.weight)
        if dehr is not None:
            for p in dehr.parameters():
                p.requires_grad = True
                train_params.append(p)
        params = [{"params": train_params, "lr": opt.lr, "weight_decay": 0.0}]
        optimizer = torch.optim.AdamW(params, lr=opt.lr, eps=opt.adam_epsilon)
        if accumulation_step is None:
            accumulation_step = opt.accumulation_steps
        if opt.scheduler == 'fixed':
            from src.utils import FixedScheduler
            scheduler = FixedScheduler(optimizer)
        elif opt.scheduler == 'linear':
            from transformers import get_linear_schedule_with_warmup
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(opt.warmup_steps / accumulation_step),
                num_training_steps=int(opt.total_steps / accumulation_step),
            )
        elif opt.scheduler == 'cos':
            from transformers import get_cosine_schedule_with_warmup
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(opt.warmup_steps / accumulation_step),
                num_training_steps=int(opt.total_steps / accumulation_step),
            )
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        if self.rank == 0:
            bias_shape = tuple(bias.weight.shape) if bias is not None else None
            self.logger.info(
                f"TypeModalityBiasOnlyTrain | trainable_params={trainable_params} "
                f"bias_shape={bias_shape} lr={opt.lr} "
                f"shared={getattr(opt, 'type_modality_bias_shared', False)} "
                f"use_dehr={getattr(opt, 'use_dehr_router', False)} weight_decay=0.0"
            )
        return optimizer, scheduler

    def _type_modality_bias_modal_names(self):
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

    def _log_type_modality_bias_table(self):
        if self.rank != 0 or not (getattr(self.args, "use_type_modality_bias", False) or getattr(self.args, "use_dehr_router", False)):
            return
        fusion = getattr(self.model.multimodal_encoder, "fusion", None)
        bias = getattr(fusion, "type_modality_bias", None)
        modal_names = self._type_modality_bias_modal_names()
        type_names = list(getattr(self.model, "top_type_names", []) or getattr(self.args, "top_type_names", []) or [])
        if bias is not None:
            weight = bias.weight.detach().cpu()
            rows = []
            for idx in range(weight.shape[0]):
                type_name = type_names[idx] if idx < len(type_names) else str(idx)
                values = " ".join(
                    f"{modal}={float(weight[idx, j]):+.4f}"
                    for j, modal in enumerate(modal_names[:weight.shape[1]])
                )
                rows.append(f"{type_name}[{idx}] {values}")
            self.logger.info(
                "TypeModalityBiasTable | "
                f"epoch={self.epoch} shared={getattr(self.args, 'type_modality_bias_shared', False)} "
                + " || ".join(rows)
            )
        dehr = getattr(fusion, "dehr_router", None)
        if dehr is not None:
            parts = []
            rho = getattr(dehr, "last_rho", None)
            bias_dehr = getattr(dehr, "last_bias", None)
            u = getattr(dehr, "last_u", None)
            u_free = getattr(dehr, "last_u_free", None)
            residual = getattr(dehr, "last_residual", None)
            confidence = getattr(dehr, "last_residual_confidence", None)
            drive = getattr(dehr, "last_residual_drive", None)
            alpha = getattr(dehr, "last_alpha", None)
            direction = getattr(dehr, "last_direction", None)
            global_bias = getattr(dehr, "last_global_bias", None)
            type_bias = getattr(dehr, "last_type_bias", None)
            evidence_bias = getattr(dehr, "last_evidence_bias", None)
            if rho is not None:
                parts.append("rho=" + ",".join(f"{float(x):.4f}" for x in rho.detach().cpu().tolist()))
            if direction is not None:
                parts.append("direction=" + ",".join(f"{float(x):+.4f}" for x in direction.detach().cpu().tolist()))
            if bias_dehr is not None:
                bd = bias_dehr.detach().cpu()
                parts.append(f"bias_abs_mean={float(bd.abs().mean()):.4f}")
                parts.append(f"bias_abs_max={float(bd.abs().max()):.4f}")
            if global_bias is not None:
                gb = global_bias.detach().cpu()
                parts.append(f"global_bias_abs_mean={float(gb.abs().mean()):.4f}")
            if type_bias is not None:
                tb = type_bias.detach().cpu()
                parts.append(f"type_bias_abs_mean={float(tb.abs().mean()):.4f}")
                if global_bias is not None:
                    eb_mean = 0.0
                    if evidence_bias is not None:
                        eb_mean = float(evidence_bias.detach().abs().mean().item())
                    denom = float(global_bias.detach().abs().mean().item() + tb.abs().mean().item() + eb_mean + 1e-12)
                    parts.append(f"type_bias_fraction={float(tb.abs().mean().item()) / denom:.4f}")
            if evidence_bias is not None:
                eb = evidence_bias.detach().cpu()
                parts.append(f"evidence_bias_abs_mean={float(eb.abs().mean()):.4f}")
                if global_bias is not None:
                    tb_mean = 0.0
                    if type_bias is not None:
                        tb_mean = float(type_bias.detach().abs().mean().item())
                    denom = float(global_bias.detach().abs().mean().item() + tb_mean + eb.abs().mean().item() + 1e-12)
                    parts.append(f"evidence_bias_fraction={float(eb.abs().mean().item()) / denom:.4f}")
            if bias_dehr is not None:
                scale = getattr(dehr, "last_scale", None)
                type_ids = getattr(dehr, "last_type_ids", None)
                if scale is not None and type_ids is not None:
                    sc = scale.detach().cpu()
                    tids = type_ids.detach().cpu()
                    scale_rows = []
                    for idx in sorted(tids.unique().tolist()):
                        mask = tids == idx
                        if not mask.any():
                            continue
                        type_name = type_names[idx] if idx < len(type_names) else str(idx)
                        vals = sc[mask].mean(dim=0)
                        values = " ".join(
                            f"{modal}={float(vals[j]):.4f}"
                            for j, modal in enumerate(modal_names[:vals.numel()])
                        )
                        scale_rows.append(f"{type_name}[{idx}] {values}")
                    if scale_rows:
                        parts.append("scale_by_type=" + " || ".join(scale_rows))
            if u is not None:
                uu = u.detach().cpu()
                parts.append(f"u_abs_mean={float(uu.abs().mean()):.4f}")
            if u_free is not None:
                uf = u_free.detach().cpu()
                parts.append(f"u_free_abs_mean={float(uf.abs().mean()):.4f}")
            if residual is not None:
                rr = residual.detach().cpu()
                parts.append(f"res_abs_mean={float(rr.abs().mean()):.4f}")
                parts.append(f"res_abs_max={float(rr.abs().max()):.4f}")
            if confidence is not None:
                cc = confidence.detach().cpu()
                parts.append(f"res_conf_mean={float(cc.mean()):.4f}")
            if drive is not None:
                dd = drive.detach().cpu()
                parts.append(f"res_drive_mean={float(dd.mean()):.4f}")
                parts.append(f"res_drive_abs_mean={float(dd.abs().mean()):.4f}")
            if alpha is not None:
                aa = alpha.detach().cpu()
                parts.append(f"alpha_mean={float(aa.mean()):.4f}")
            contextual_mix = getattr(dehr, "last_contextual_mix", None)
            learned_type_u = getattr(dehr, "last_learned_type_u", None)
            contextual_u = getattr(dehr, "last_contextual_u", None)
            if contextual_mix is not None:
                parts.append(f"contextual_mix={float(contextual_mix):.4f}")
            if learned_type_u is not None:
                ltu = learned_type_u.detach().cpu()
                parts.append(f"learned_type_u_abs_mean={float(ltu.abs().mean()):.4f}")
            if contextual_u is not None:
                cu = contextual_u.detach().cpu()
                parts.append(f"contextual_u_abs_mean={float(cu.abs().mean()):.4f}")
                if learned_type_u is not None:
                    parts.append(f"contextual_delta_abs_mean={float((cu - learned_type_u.detach().cpu()).abs().mean()):.4f}")
            if parts:
                self.logger.info("DEHRRouterTable | epoch={} ".format(self.epoch) + " ".join(parts))
    def data_init(self):
        self.KGs, self.non_train, self.train_set, self.eval_set, self.test_set, self.test_ill_ = load_data(self.logger, self.args)
        self.train_ill = self.train_set.data
        self.eval_left = torch.LongTensor(self.eval_set[:, 0].squeeze()).cuda()
        self.eval_right = torch.LongTensor(self.eval_set[:, 1].squeeze()).cuda()
        if self.test_set is not None:
            self.test_left = torch.LongTensor(self.test_ill[:, 0].squeeze()).cuda()
            self.test_right = torch.LongTensor(self.test_ill[:, 1].squeeze()).cuda()

        self.eval_sampler = None
        if self.args.dist and not self.args.only_test:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_set)
            self.eval_sampler = torch.utils.data.distributed.DistributedSampler(self.eval_set)
            if self.test_set is not None:
                self.test_sampler = torch.utils.data.distributed.DistributedSampler(self.test_set)

    def dataloader_init(self, train_set=None, eval_set=None, test_set=None):
        bs = self.args.batch_size
        collator = Collator_base(self.args)
        if self.args.dist and not self.args.only_test:
            self.args.workers = min([os.cpu_count(), self.args.batch_size, self.args.workers])
            if train_set is not None:
                self.train_dataloader = self._dataloader_dist(train_set, self.train_sampler, bs, collator)
            if test_set is not None:
                self.test_dataloader = self._dataloader_dist(test_set, self.test_sampler, bs, collator)
            if eval_set is not None:
                self.eval_dataloader = self._dataloader_dist(eval_set, self.eval_sampler, bs, collator)
        else:
            self.args.workers = min([os.cpu_count(), self.args.batch_size, self.args.workers])
            if train_set is not None:
                self.train_dataloader = self._dataloader(train_set, bs, collator)
            if test_set is not None:
                self.test_dataloader = self._dataloader(test_set, bs, collator)
            if eval_set is not None:
                self.eval_dataloader = self._dataloader(eval_set, bs, collator)

    def _dataloader_dist(self, train_set, train_sampler, batch_size, collator):
        train_dataloader = DataLoader(
            train_set,
            sampler=train_sampler,
            pin_memory=True,
            num_workers=self.args.workers,
            persistent_workers=self.args.workers > 0,
            drop_last=True,
            batch_size=batch_size,
            collate_fn=collator
        )
        return train_dataloader

    def _dataloader(self, train_set, batch_size, collator):
        train_dataloader = DataLoader(
            train_set,
            num_workers=self.args.workers,
            persistent_workers=self.args.workers > 0,
            shuffle=(self.args.only_test == 0),
            # drop_last=(self.args.only_test == 0),
            drop_last=False,
            batch_size=batch_size,
            collate_fn=collator
        )
        return train_dataloader

    def run(self):
        self.loss_log = Loss_log()
        self.curr_loss = 0.
        self.lr = self.args.lr
        self.curr_loss_dic = defaultdict(float)
        self.curr_loss_dic_count = 0
        self.weight = [1, 1, 1, 1, 1, 1]
        self.loss_weight = [1, 1]
        self.loss_item = 99999.
        self.step = 1
        self.epoch = 0
        self.new_links = []
        self.best_model_wts = None

        self.best_mrr = 0

        self.non_iter_early_stop_patience = 50
        self.iter_early_stop_patience = 2000
        self.early_stop_init = self.non_iter_early_stop_patience
        self.early_stop_count = self.early_stop_init
        self.stage = 0

        self.dehr_calibration_pretrain()

        with tqdm(total=self.args.epoch) as _tqdm:
            for i in range(self.args.epoch):
                # _tqdm.set_description(f'Train | epoch {i} Loss {self.loss_log.get_loss():.5f} Acc {self.loss_log.get_acc()*100:.3f}%')
                if self.args.dist and not self.args.only_test:
                    self.train_sampler.set_epoch(i)
                # -------------------------------
                self.epoch = i
                if hasattr(self.model, "set_epoch"):
                    self.model.set_epoch(self.epoch)
                early_stop_triggered = self.early_stop_count <= 0 and self.epoch <= self.args.il_start
                if getattr(self.args, "disable_early_stop", False):
                    early_stop_triggered = False
                if self.args.il and ((self.epoch == self.args.il_start and self.stage == 0) or early_stop_triggered):
                    if self.early_stop_count <= 0:
                        logger.info(f"Early stop in epoch {self.epoch}... Begin iteration....")
                    self.stage = 1
                    self.early_stop_init = self.iter_early_stop_patience
                    self.early_stop_count = self.early_stop_init

                    self.eval_epoch = 1

                    self.step = 1
                    self.args.lr = self.args.lr / 5
                    self.optim_init(self.args, total_epoch=(self.args.epoch - self.args.il_start) * 3)
                    if self.best_model_wts is not None:
                        self.logger.info("load from the best model before IL... ")
                        self.model.load_state_dict(self.best_model_wts)
                    name = self._save_name_define()
                    self.test(save_name=f"{name}_test_ep{self.args.epoch}_no_iter")
                    if self.rank == 0:
                        if not self.args.only_test and self.args.save_model:
                            self._save_model(self.model, input_name=f"{name}_non_iter")

                if self.stage == 1 and (self.epoch + 1) % self.args.semi_learn_step == 0 and self.args.il:
                    self.il_for_ea()

                if self.stage == 1 and (self.epoch + 1) % (self.args.semi_learn_step * 10) == 0 and len(self.new_links) != 0 and self.args.il:
                    self.il_for_data_ref()

                self.train(_tqdm)
                self.loss_log.update(self.curr_loss)
                self.loss_item = self.loss_log.get_loss()
                _tqdm.set_description(f'Train | Ep [{self.epoch}/{self.args.epoch}] Step [{self.step}/{self.args.total_steps}] LR [{self.lr:.5f}] Loss {self.loss_log.get_loss():.5f} ')
                self.update_loss_log()
                if (i + 1) % self.args.eval_epoch == 0:
                    self.eval()
                _tqdm.update(1)
                if not getattr(self.args, "disable_early_stop", False) and not self.args.il and self.early_stop_count <= 0:
                    logger.info(f"Early stop in epoch {self.epoch} under non-iterative training.")
                    break
                if not getattr(self.args, "disable_early_stop", False) and self.stage == 1 and self.early_stop_count <= 0:
                    logger.info(f"Early stop in epoch {self.epoch}")
                    break

        name = self._save_name_define()
        if self.best_model_wts is not None:
            self.logger.info("load from the best model before final testing ... ")
            self.model.load_state_dict(self.best_model_wts)
        self.test(save_name=f"{name}_test_ep{self.args.epoch}")

        if self.rank == 0:
            self.logger.info(f"min loss {self.loss_log.get_min_loss()}")
            if not self.args.only_test and self.args.save_model:
                self._save_model(self.model, input_name=name)

    def _dehr_calib_router_params(self, router):
        return [p for p in router.parameters() if p.requires_grad]
    def _dehr_calib_compose_embedding(self, modal_embs, weight_norm, router, type_ids):
        bias = router(modal_embs, weight_norm, entity_type_ids=type_ids)
        self._dehr_calib_last_bias = bias
        logits = torch.log(weight_norm.detach().clamp_min(1e-12)) + bias
        tau_route = float(getattr(self.args, "dehr_tau_route", 1.0))
        tau_route = max(tau_route, 1e-6)
        weight_final = F.softmax(logits / tau_route, dim=-1)
        joint = torch.cat(
            [
                weight_final[:, idx:idx + 1] * F.normalize(modal_embs[idx], dim=1)
                for idx in range(len(modal_embs))
            ],
            dim=1,
        )
        return F.normalize(joint, dim=1)
    def _dehr_calib_loss(self, emb, links, left_bank, right_bank, left_pos, right_pos, tau, batch_size, train_mode=True):
        if not torch.is_tensor(links):
            if getattr(links, "ndim", 0) == 2 and links.shape[1] > 2:
                links = torch.as_tensor(links, dtype=torch.float32, device=emb.device)
            else:
                links = torch.as_tensor(links, dtype=torch.long, device=emb.device)
        sample_weights = None
        if links.ndim == 2 and links.shape[1] > 2:
            sample_weights = links[:, 2].float()
            links = links[:, :2].long()
        if train_mode:
            perm = torch.randperm(links.shape[0], device=emb.device)
            links = links[perm]
            if sample_weights is not None:
                sample_weights = sample_weights[perm]
        losses = []
        total = int(links.shape[0])
        for start in range(0, total, batch_size):
            batch = links[start:start + batch_size]
            batch_weights = None
            if sample_weights is not None:
                batch_weights = sample_weights[start:start + batch_size].clamp_min(1e-6)
            left = batch[:, 0].long()
            right = batch[:, 1].long()
            target_r = right_pos[right]
            target_l = left_pos[left]
            valid_lr = target_r >= 0
            valid_rl = target_l >= 0
            local_losses = []
            if valid_lr.any():
                logits_lr = emb[left[valid_lr]] @ emb[right_bank].t() / tau
                loss_lr = F.cross_entropy(logits_lr, target_r[valid_lr], reduction="none")
                if batch_weights is not None:
                    w = batch_weights[valid_lr]
                    loss_lr = (loss_lr * w).sum() / w.sum().clamp_min(1e-6)
                else:
                    loss_lr = loss_lr.mean()
                local_losses.append(loss_lr)
            if valid_rl.any():
                logits_rl = emb[right[valid_rl]] @ emb[left_bank].t() / tau
                loss_rl = F.cross_entropy(logits_rl, target_l[valid_rl], reduction="none")
                if batch_weights is not None:
                    w = batch_weights[valid_rl]
                    loss_rl = (loss_rl * w).sum() / w.sum().clamp_min(1e-6)
                else:
                    loss_rl = loss_rl.mean()
                local_losses.append(loss_rl)
            if local_losses:
                losses.append(sum(local_losses) / len(local_losses))
        if not losses:
            return None
        return sum(losses) / len(losses)

    @torch.no_grad()
    def _dehr_calib_hits1(self, emb, links, left_bank, right_bank, left_pos, right_pos, batch_size):
        if not torch.is_tensor(links):
            links = torch.as_tensor(links, dtype=torch.long, device=emb.device)
        total = int(links.shape[0])
        l_hit = 0
        r_hit = 0
        seen_l = 0
        seen_r = 0
        for start in range(0, total, batch_size):
            batch = links[start:start + batch_size]
            left = batch[:, 0].long()
            right = batch[:, 1].long()
            target_r = right_pos[right]
            target_l = left_pos[left]
            valid_lr = target_r >= 0
            valid_rl = target_l >= 0
            if valid_lr.any():
                logits_lr = emb[left[valid_lr]] @ emb[right_bank].t()
                l_hit += int((logits_lr.argmax(dim=1) == target_r[valid_lr]).sum().item())
                seen_l += int(valid_lr.sum().item())
            if valid_rl.any():
                logits_rl = emb[right[valid_rl]] @ emb[left_bank].t()
                r_hit += int((logits_rl.argmax(dim=1) == target_l[valid_rl]).sum().item())
                seen_r += int(valid_rl.sum().item())
        l_acc = float(l_hit / max(seen_l, 1))
        r_acc = float(r_hit / max(seen_r, 1))
        return l_acc, r_acc, 0.5 * (l_acc + r_acc)

    @torch.no_grad()
    def _mine_dehr_pseudo_pairs(self, emb):
        if self.non_train is None or self.model.left_entity_ids is None or self.model.right_entity_ids is None:
            return np.zeros((0, 2), dtype=np.int64), {"count": 0}
        pseudo_source = getattr(self.args, "dehr_calib_pseudo_source", "joint")
        if pseudo_source == "consensus":
            return self._mine_dehr_consensus_pseudo_pairs()
        if pseudo_source == "dropout_consensus":
            return self._mine_dehr_dropout_consensus_pseudo_pairs()
        left_ids = torch.as_tensor(self.non_train["left"], dtype=torch.long, device=emb.device)
        right_ids = torch.as_tensor(self.non_train["right"], dtype=torch.long, device=emb.device)
        if left_ids.numel() == 0 or right_ids.numel() == 0:
            return np.zeros((0, 2), dtype=np.int64), {"count": 0}
        sim = emb[left_ids] @ emb[right_ids].t()
        csls_k = max(int(getattr(self.args, "dehr_calib_pseudo_csls_k", 1)), 1)
        row_avg = torch.topk(sim, k=min(csls_k, sim.shape[1]), dim=1).values.mean(dim=1, keepdim=True)
        col_avg = torch.topk(sim, k=min(csls_k, sim.shape[0]), dim=0).values.mean(dim=0, keepdim=True)
        csls = 2.0 * sim - row_avg - col_avg
        topk = min(2, csls.shape[1])
        topv, topj = csls.topk(k=topk, dim=1)
        best_col = csls.argmax(dim=0)
        row_idx = torch.arange(csls.shape[0], device=emb.device)
        mutual = best_col[topj[:, 0]] == row_idx
        if topv.shape[1] == 1:
            margin = topv[:, 0]
        else:
            margin = topv[:, 0] - topv[:, 1]
        min_margin = float(getattr(self.args, "dehr_calib_pseudo_margin", 0.04))
        keep = mutual & (margin >= min_margin)
        if not keep.any():
            return np.zeros((0, 2), dtype=np.int64), {
                "count": 0,
                "margin_mean": 0.0,
                "margin_min": 0.0,
                "margin_max": 0.0,
            }
        kept_margin = margin[keep]
        pairs = torch.stack([left_ids[keep], right_ids[topj[:, 0][keep]]], dim=1)
        max_pairs = int(getattr(self.args, "dehr_calib_pseudo_max_pairs", 0))
        if max_pairs > 0 and pairs.shape[0] > max_pairs:
            order = torch.argsort(kept_margin, descending=True)
            order = order[:max_pairs]
            kept_margin = kept_margin[order]
            pairs = pairs[order]
        stats = {
            "count": int(pairs.shape[0]),
            "margin_mean": float(kept_margin.mean().item()),
            "margin_min": float(kept_margin.min().item()),
            "margin_max": float(kept_margin.max().item()),
        }
        return pairs.detach().cpu().numpy().astype(np.int64), stats

    @torch.no_grad()
    def _mine_dehr_dropout_consensus_pseudo_pairs(self):
        fusion = self.model.multimodal_encoder.fusion
        modal_embs = getattr(fusion, "last_modal_embs", None)
        weight_norm = getattr(fusion, "last_weight_norm", None)
        if modal_embs is None or weight_norm is None or self.non_train is None:
            return np.zeros((0, 2), dtype=np.int64), {"count": 0}
        device = weight_norm.device
        left_ids = torch.as_tensor(self.non_train["left"], dtype=torch.long, device=device)
        right_ids = torch.as_tensor(self.non_train["right"], dtype=torch.long, device=device)
        if left_ids.numel() == 0 or right_ids.numel() == 0:
            return np.zeros((0, 2), dtype=np.int64), {"count": 0}

        modal_embs = [F.normalize(emb.detach(), dim=1) for emb in modal_embs[:4]]
        weight_norm = weight_norm.detach()[:, :len(modal_embs)].clamp_min(1e-12)
        min_votes = max(int(getattr(self.args, "dehr_calib_consensus_min_votes", 2)), 1)
        type_match = bool(getattr(self.args, "dehr_calib_consensus_type_match", False))
        min_margin = float(getattr(self.args, "dehr_calib_pseudo_margin", 0.04))
        csls_k = max(int(getattr(self.args, "dehr_calib_pseudo_csls_k", 1)), 1)
        type_ids = self.model.entity_type_ids.to(device) if getattr(self.model, "entity_type_ids", None) is not None else None

        view_candidates = []
        view_margins = []
        for drop_idx in range(len(modal_embs)):
            mask = torch.ones((len(modal_embs),), dtype=weight_norm.dtype, device=device)
            mask[drop_idx] = 0.0
            view_w = weight_norm * mask.view(1, -1)
            view_w = view_w / view_w.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            emb = F.normalize(
                torch.cat(
                    [
                        view_w[:, idx:idx + 1] * modal_embs[idx]
                        for idx in range(len(modal_embs))
                    ],
                    dim=1,
                ),
                dim=1,
            )
            sim = emb[left_ids] @ emb[right_ids].t()
            row_avg = torch.topk(sim, k=min(csls_k, sim.shape[1]), dim=1).values.mean(dim=1, keepdim=True)
            col_avg = torch.topk(sim, k=min(csls_k, sim.shape[0]), dim=0).values.mean(dim=0, keepdim=True)
            csls = 2.0 * sim - row_avg - col_avg
            topk = min(2, csls.shape[1])
            topv, topj = csls.topk(k=topk, dim=1)
            best_col = csls.argmax(dim=0)
            row_idx = torch.arange(csls.shape[0], device=device)
            mutual = best_col[topj[:, 0]] == row_idx
            if topv.shape[1] == 1:
                margin = topv[:, 0]
            else:
                margin = topv[:, 0] - topv[:, 1]
            valid = mutual & (margin >= min_margin)
            if type_match and type_ids is not None:
                valid = valid & (type_ids[left_ids] == type_ids[right_ids[topj[:, 0]]])
            candidate = right_ids[topj[:, 0]]
            view_candidates.append(candidate)
            view_margins.append(torch.where(valid, margin, margin.new_full(margin.shape, -1e9)))

        stacked_candidates = torch.stack(view_candidates, dim=0)
        stacked_margins = torch.stack(view_margins, dim=0)
        keep_rows = []
        keep_cols = []
        keep_scores = []
        for idx in range(left_ids.shape[0]):
            valid_views = stacked_margins[:, idx] > -1e8
            if int(valid_views.sum().item()) < min_votes:
                continue
            cand = stacked_candidates[valid_views, idx]
            unique, counts = torch.unique(cand, return_counts=True)
            best_count, best_pos = counts.max(dim=0)
            if int(best_count.item()) < min_votes:
                continue
            chosen = unique[best_pos]
            vote_mask = valid_views & (stacked_candidates[:, idx] == chosen)
            score = stacked_margins[vote_mask, idx].mean()
            keep_rows.append(left_ids[idx])
            keep_cols.append(chosen)
            keep_scores.append(score)

        if not keep_rows:
            return np.zeros((0, 2), dtype=np.int64), {
                "count": 0,
                "margin_mean": 0.0,
                "margin_min": 0.0,
                "margin_max": 0.0,
                "source": "dropout_consensus",
            }
        keep_rows = torch.stack(keep_rows)
        keep_cols = torch.stack(keep_cols)
        keep_scores = torch.stack(keep_scores)
        max_pairs = int(getattr(self.args, "dehr_calib_pseudo_max_pairs", 0))
        if max_pairs > 0 and keep_rows.shape[0] > max_pairs:
            order = torch.argsort(keep_scores, descending=True)[:max_pairs]
            keep_rows = keep_rows[order]
            keep_cols = keep_cols[order]
            keep_scores = keep_scores[order]
        pairs = torch.stack([keep_rows, keep_cols], dim=1)
        stats = {
            "count": int(pairs.shape[0]),
            "margin_mean": float(keep_scores.mean().item()),
            "margin_min": float(keep_scores.min().item()),
            "margin_max": float(keep_scores.max().item()),
            "source": "dropout_consensus",
        }
        return pairs.detach().cpu().numpy().astype(np.int64), stats

    @torch.no_grad()
    def _mine_dehr_consensus_pseudo_pairs(self):
        fusion = self.model.multimodal_encoder.fusion
        modal_embs = getattr(fusion, "last_modal_embs", None)
        if modal_embs is None or self.non_train is None:
            return np.zeros((0, 2), dtype=np.int64), {"count": 0}
        device = modal_embs[0].device
        left_ids = torch.as_tensor(self.non_train["left"], dtype=torch.long, device=device)
        right_ids = torch.as_tensor(self.non_train["right"], dtype=torch.long, device=device)
        if left_ids.numel() == 0 or right_ids.numel() == 0:
            return np.zeros((0, 2), dtype=np.int64), {"count": 0}
        min_votes = max(int(getattr(self.args, "dehr_calib_consensus_min_votes", 2)), 1)
        min_modalities = max(int(getattr(self.args, "dehr_calib_consensus_min_modalities", 2)), 1)
        type_match = bool(getattr(self.args, "dehr_calib_consensus_type_match", False))
        modal_candidates = []
        modal_margins = []
        valid_counts = torch.zeros(left_ids.shape[0], dtype=torch.long, device=device)
        type_ids = self.model.entity_type_ids.to(device) if getattr(self.model, "entity_type_ids", None) is not None else None

        for modal_emb in modal_embs[:4]:
            emb = F.normalize(modal_emb.detach(), dim=1)
            sim = emb[left_ids] @ emb[right_ids].t()
            csls_k = max(int(getattr(self.args, "dehr_calib_pseudo_csls_k", 1)), 1)
            row_avg = torch.topk(sim, k=min(csls_k, sim.shape[1]), dim=1).values.mean(dim=1, keepdim=True)
            col_avg = torch.topk(sim, k=min(csls_k, sim.shape[0]), dim=0).values.mean(dim=0, keepdim=True)
            csls = 2.0 * sim - row_avg - col_avg
            topk = min(2, csls.shape[1])
            topv, topj = csls.topk(k=topk, dim=1)
            best_col = csls.argmax(dim=0)
            row_idx = torch.arange(csls.shape[0], device=device)
            mutual = best_col[topj[:, 0]] == row_idx
            if topv.shape[1] == 1:
                margin = topv[:, 0]
            else:
                margin = topv[:, 0] - topv[:, 1]
            modal_valid = mutual & (margin >= float(getattr(self.args, "dehr_calib_pseudo_margin", 0.04)))
            if type_match and type_ids is not None:
                left_type = type_ids[left_ids]
                right_type = type_ids[right_ids[topj[:, 0]]]
                modal_valid = modal_valid & (left_type == right_type)
            valid_counts += modal_valid.long()
            modal_candidates.append(right_ids[topj[:, 0]])
            modal_margins.append(torch.where(modal_valid, margin, margin.new_zeros(margin.shape)))

        stacked_candidates = torch.stack(modal_candidates, dim=0)  # M x N
        stacked_margins = torch.stack(modal_margins, dim=0)
        keep_rows = []
        keep_cols = []
        keep_scores = []
        for idx in range(left_ids.shape[0]):
            if valid_counts[idx].item() < min_modalities:
                continue
            cand = stacked_candidates[:, idx]
            unique, counts = torch.unique(cand, return_counts=True)
            best_count, best_pos = counts.max(dim=0)
            if int(best_count.item()) < min_votes:
                continue
            chosen = unique[best_pos]
            vote_mask = cand == chosen
            score = stacked_margins[vote_mask, idx].mean()
            keep_rows.append(left_ids[idx])
            keep_cols.append(chosen)
            keep_scores.append(score)
        if not keep_rows:
            return np.zeros((0, 2), dtype=np.int64), {
                "count": 0,
                "margin_mean": 0.0,
                "margin_min": 0.0,
                "margin_max": 0.0,
                "source": "consensus",
            }
        keep_rows = torch.stack(keep_rows)
        keep_cols = torch.stack(keep_cols)
        keep_scores = torch.stack(keep_scores)
        max_pairs = int(getattr(self.args, "dehr_calib_pseudo_max_pairs", 0))
        if max_pairs > 0 and keep_rows.shape[0] > max_pairs:
            order = torch.argsort(keep_scores, descending=True)[:max_pairs]
            keep_rows = keep_rows[order]
            keep_cols = keep_cols[order]
            keep_scores = keep_scores[order]
        pairs = torch.stack([keep_rows, keep_cols], dim=1)
        stats = {
            "count": int(pairs.shape[0]),
            "margin_mean": float(keep_scores.mean().item()),
            "margin_min": float(keep_scores.min().item()),
            "margin_max": float(keep_scores.max().item()),
            "source": "consensus",
        }
        return pairs.detach().cpu().numpy().astype(np.int64), stats

    @torch.no_grad()
    def _build_dehr_pseudo_agreement_weights(self, pseudo_links):
        if pseudo_links is None or pseudo_links.shape[0] == 0:
            return np.zeros((0,), dtype=np.float32), {"agreement_weight_mean": 0.0}
        fusion = self.model.multimodal_encoder.fusion
        modal_embs = getattr(fusion, "last_modal_embs", None)
        if modal_embs is None:
            ones = np.ones((pseudo_links.shape[0],), dtype=np.float32)
            return ones, {
                "agreement_weight_mean": 1.0,
                "agreement_vote_mean": 1.0,
                "agreement_margin_mean": 0.0,
            }

        device = modal_embs[0].device
        pairs = torch.as_tensor(pseudo_links, dtype=torch.long, device=device)
        left = pairs[:, 0]
        right = pairs[:, 1]
        right_pool = torch.as_tensor(self.non_train["right"], dtype=torch.long, device=device)
        csls_k = max(int(getattr(self.args, "dehr_calib_pseudo_csls_k", 1)), 1)
        min_w = float(getattr(self.args, "dehr_calib_pseudo_weight_min", 0.35))
        vote_scale = float(getattr(self.args, "dehr_calib_pseudo_weight_vote_scale", 0.65))

        votes = torch.zeros((pairs.shape[0],), dtype=torch.float32, device=device)
        margins = torch.zeros((pairs.shape[0],), dtype=torch.float32, device=device)
        modal_count = 0
        for modal_emb in modal_embs[:4]:
            emb = F.normalize(modal_emb.detach(), dim=1)
            sim = emb[left] @ emb[right_pool].t()
            row_avg = torch.topk(sim, k=min(csls_k, sim.shape[1]), dim=1).values.mean(dim=1, keepdim=True)
            col_avg = torch.topk(sim, k=min(csls_k, sim.shape[0]), dim=0).values.mean(dim=0, keepdim=True)
            csls = 2.0 * sim - row_avg - col_avg
            topk = min(2, csls.shape[1])
            topv, topj = csls.topk(k=topk, dim=1)
            chosen = right_pool[topj[:, 0]]
            vote = (chosen == right).float()
            if topv.shape[1] == 1:
                margin = topv[:, 0]
            else:
                margin = (topv[:, 0] - topv[:, 1]).clamp_min(0.0)
            votes += vote
            margins += vote * margin
            modal_count += 1

        if modal_count <= 0:
            ones = np.ones((pseudo_links.shape[0],), dtype=np.float32)
            return ones, {
                "agreement_weight_mean": 1.0,
                "agreement_vote_mean": 1.0,
                "agreement_margin_mean": 0.0,
            }

        vote_ratio = votes / float(modal_count)
        support_margin = margins / votes.clamp_min(1.0)
        margin_term = torch.tanh(5.0 * support_margin)
        weights = min_w + vote_scale * (0.7 * vote_ratio + 0.3 * margin_term)
        weights = weights.clamp_(min_w, 1.0)
        stats = {
            "agreement_weight_mean": float(weights.mean().item()),
            "agreement_vote_mean": float(vote_ratio.mean().item()),
            "agreement_margin_mean": float(support_margin.mean().item()),
        }
        return weights.detach().cpu().numpy().astype(np.float32), stats

    def dehr_calibration_pretrain(self):
        epochs = int(getattr(self.args, "dehr_calib_pretrain_epochs", 0))
        if epochs <= 0 or not getattr(self.args, "use_dehr_router", False):
            return
        router = self._get_dehr_router()
        if router is None:
            return
        if self.rank != 0:
            return
        train_links = np.asarray(self.train_ill, dtype=np.int64)
        if train_links.ndim != 2 or train_links.shape[0] < 4:
            return

        prev_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            _, weight_norm = self.model.joint_emb_generat()
            fusion = self.model.multimodal_encoder.fusion
            modal_embs = getattr(fusion, "last_modal_embs", None)
            if modal_embs is None:
                raise ValueError("DEHR calibration requires cached modal embeddings from fusion.")
            modal_embs = [emb.detach() for emb in modal_embs[:router.modal_num]]
            weight_norm = weight_norm.detach()
            type_ids = self.model.entity_type_ids.detach()
            base_emb = F.normalize(
                torch.cat(
                    [
                        weight_norm[:, idx:idx + 1] * F.normalize(modal_embs[idx], dim=1)
                        for idx in range(len(modal_embs))
                    ],
                    dim=1,
                ),
                dim=1,
            )

        rng = np.random.default_rng(int(getattr(self.args, "random_seed", 42)))
        pair_source = getattr(self.args, "dehr_calib_pair_source", "train")
        source_links = train_links
        source_note = f"train={train_links.shape[0]}"
        if pair_source in ("pseudo", "train_pseudo"):
            pseudo_links, pseudo_stats = self._mine_dehr_pseudo_pairs(base_emb)
            source_note += (
                f" pseudo={pseudo_stats.get('count', 0)}"
                f" pseudo_margin_mean={pseudo_stats.get('margin_mean', 0.0):.4f}"
            )
            pseudo_weights = None
            if (
                pseudo_links.shape[0] > 0
                and getattr(self.args, "dehr_calib_pseudo_weight_mode", "none") == "agreement"
                and getattr(self.args, "dehr_calib_pseudo_source", "joint") == "joint"
            ):
                pseudo_weights, weight_stats = self._build_dehr_pseudo_agreement_weights(pseudo_links)
                source_note += (
                    f" pseudo_w_mean={weight_stats.get('agreement_weight_mean', 1.0):.4f}"
                    f" pseudo_vote_mean={weight_stats.get('agreement_vote_mean', 0.0):.4f}"
                )
            if pair_source == "pseudo":
                if pseudo_links.shape[0] > 0:
                    pseudo_loss_weight = float(getattr(self.args, "dehr_calib_pseudo_loss_weight", 1.0))
                    if pseudo_weights is None and abs(pseudo_loss_weight - 1.0) > 1e-8:
                        pseudo_weights = np.full((pseudo_links.shape[0],), pseudo_loss_weight, dtype=np.float32)
                        source_note += f" pseudo_loss_w={pseudo_loss_weight:.3f}"
                    if pseudo_weights is not None:
                        source_links = np.concatenate(
                            [pseudo_links, pseudo_weights[:, None].astype(np.float32)],
                            axis=1,
                        )
                    else:
                        source_links = pseudo_links
                else:
                    source_links = train_links
                    source_note += " fallback=train"
            elif pair_source == "train_pseudo" and pseudo_links.shape[0] > 0:
                pseudo_loss_weight = float(getattr(self.args, "dehr_calib_pseudo_loss_weight", 1.0))
                if pseudo_weights is None and abs(pseudo_loss_weight - 1.0) > 1e-8:
                    pseudo_weights = np.full((pseudo_links.shape[0],), pseudo_loss_weight, dtype=np.float32)
                    source_note += f" pseudo_loss_w={pseudo_loss_weight:.3f}"
                if pseudo_weights is not None:
                    train_weight = np.ones((train_links.shape[0], 1), dtype=np.float32)
                    pseudo_weight = pseudo_weights[:, None].astype(np.float32)
                    train_weighted = np.concatenate([train_links.astype(np.float32), train_weight], axis=1)
                    pseudo_weighted = np.concatenate([pseudo_links.astype(np.float32), pseudo_weight], axis=1)
                    source_links = np.concatenate([train_weighted, pseudo_weighted], axis=0)
                else:
                    source_links = np.concatenate([train_links, pseudo_links], axis=0)
        perm = rng.permutation(source_links.shape[0])
        val_ratio = min(max(float(getattr(self.args, "dehr_calib_pretrain_val_ratio", 0.2)), 0.05), 0.5)
        val_count = max(1, int(round(source_links.shape[0] * val_ratio)))
        calib_val = source_links[perm[:val_count]]
        calib_train = source_links[perm[val_count:]]
        if calib_train.shape[0] == 0:
            calib_train = calib_val

        device = weight_norm.device
        candidate_left = self.model.left_entity_ids.long().to(device)
        candidate_right = self.model.right_entity_ids.long().to(device)
        left_pos = torch.full((self.model.input_idx.shape[0],), -1, dtype=torch.long, device=device)
        right_pos = torch.full((self.model.input_idx.shape[0],), -1, dtype=torch.long, device=device)
        left_pos[candidate_left] = torch.arange(candidate_left.shape[0], dtype=torch.long, device=device)
        right_pos[candidate_right] = torch.arange(candidate_right.shape[0], dtype=torch.long, device=device)

        previous_requires_grad = {p: p.requires_grad for p in self.model.parameters()}
        for p in self.model.parameters():
            p.requires_grad = False
        for p in router.parameters():
            p.requires_grad = True
        params = self._dehr_calib_router_params(router)
        optimizer = torch.optim.AdamW(
            params,
            lr=float(getattr(self.args, "dehr_calib_pretrain_lr", 1e-3)),
            weight_decay=float(getattr(self.args, "dehr_calib_pretrain_weight_decay", 1e-4)),
        )
        tau = max(float(getattr(self.args, "dehr_calib_pretrain_tau", 0.05)), 1e-6)
        batch_size = max(int(getattr(self.args, "dehr_calib_pretrain_batch_size", 512)), 1)
        patience = max(int(getattr(self.args, "dehr_calib_pretrain_patience", 8)), 1)
        bias_l2 = max(float(getattr(self.args, "dehr_calib_bias_l2", 0.0)), 0.0)
        best_metric = -1.0
        best_state = copy.deepcopy(router.state_dict())
        stale = 0
        self.logger.info(
            "DEHRCalibPretrain | "
            f"epochs={epochs} train={calib_train.shape[0]} val={calib_val.shape[0]} "
            f"lr={float(getattr(self.args, 'dehr_calib_pretrain_lr', 1e-3))} "
            f"tau={tau} batch={batch_size} val_ratio={val_ratio:.3f} "
            f"bias_l2={bias_l2} pair_source={pair_source} {source_note}"
        )
        selection = getattr(self.args, "dehr_calib_selection", "best_val")
        best_loss = float("inf")
        final_state = None
        for ep in range(epochs):
            router.train()
            emb = self._dehr_calib_compose_embedding(modal_embs, weight_norm, router, type_ids)
            loss = self._dehr_calib_loss(
                emb,
                calib_train,
                candidate_left,
                candidate_right,
                left_pos,
                right_pos,
                tau=tau,
                batch_size=batch_size,
                train_mode=True,
            )
            if loss is None:
                break
            if bias_l2 > 0 and getattr(self, "_dehr_calib_last_bias", None) is not None:
                loss = loss + bias_l2 * self._dehr_calib_last_bias.pow(2).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, float(getattr(self.args, "clip", 1.0)))
            optimizer.step()

            router.eval()
            with torch.no_grad():
                emb_eval = self._dehr_calib_compose_embedding(modal_embs, weight_norm, router, type_ids)
                val_l, val_r, val_avg = self._dehr_calib_hits1(
                    emb_eval,
                    calib_val,
                    candidate_left,
                    candidate_right,
                    left_pos,
                    right_pos,
                    batch_size=batch_size,
                )
                train_l, train_r, train_avg = self._dehr_calib_hits1(
                    emb_eval,
                    calib_train,
                    candidate_left,
                    candidate_right,
                    left_pos,
                    right_pos,
                    batch_size=batch_size,
                )
            if val_avg > best_metric:
                best_metric = val_avg
                best_state = copy.deepcopy(router.state_dict())
                stale = 0
            else:
                stale += 1
            if float(loss.detach().item()) < best_loss:
                best_loss = float(loss.detach().item())
                best_loss_state = copy.deepcopy(router.state_dict())
            final_state = copy.deepcopy(router.state_dict())
            if ep == 0 or (ep + 1) % max(1, min(5, epochs)) == 0 or ep == epochs - 1:
                stats = self.model.type_modality_bias_stats()
                stat_items = " ".join(
                    f"{key}={stats[key]:.4f}"
                    for key in (
                        "type_modality_weight_delta_abs_mean",
                        "dehr_bias_abs_mean",
                        "dehr_global_bias_abs_mean",
                        "dehr_type_bias_abs_mean",
                        "dehr_evidence_bias_abs_mean",
                        "dehr_contextual_mix",
                        "dehr_contextual_u_abs_mean",
                        "dehr_contextual_delta_abs_mean",
                    )
                    if key in stats
                )
                self.logger.info(
                    "DEHRCalibPretrain | "
                    f"epoch={ep} loss={float(loss.detach().item()):.4f} "
                    f"train_avg={train_avg:.4f} val_avg={val_avg:.4f} "
                    f"val_l2r={val_l:.4f} val_r2l={val_r:.4f} "
                    f"best_val={best_metric:.4f} best_loss={best_loss:.4f} {stat_items}"
                )
                self._log_type_modality_bias_table()
            if selection == "best_val" and stale >= patience:
                self.logger.info(
                    f"DEHRCalibPretrain | early_stop epoch={ep} best_val_avg={best_metric:.4f}"
                )
                break

        load_state = best_state
        if selection == "best_loss" and 'best_loss_state' in locals():
            load_state = best_loss_state
        elif selection == "final" and final_state is not None:
            load_state = final_state
        router.load_state_dict(load_state)
        router.eval()
        with torch.no_grad():
            emb_eval = self._dehr_calib_compose_embedding(modal_embs, weight_norm, router, type_ids)
            val_l, val_r, val_avg = self._dehr_calib_hits1(
                emb_eval,
                calib_val,
                candidate_left,
                candidate_right,
                left_pos,
                right_pos,
                batch_size=batch_size,
            )
        self.logger.info(
            f"DEHRCalibPretrain | loaded_{selection} val_avg={val_avg:.4f} val_l2r={val_l:.4f} val_r2l={val_r:.4f}"
        )
        self._log_type_modality_bias_table()
        for p, requires_grad in previous_requires_grad.items():
            p.requires_grad = requires_grad
        self.model.train(prev_training)
    def il_for_ea(self):
        with torch.no_grad():
            if self.args.model_name in ["TIDEA"]:
                final_emb, weight_norm = self.model.joint_emb_generat()
            else:
                final_emb = self.model.joint_emb_generat()
            final_emb = F.normalize(final_emb)
            self.new_links = self.model.Iter_new_links(self.epoch, self.non_train["left"], final_emb, self.non_train["right"], new_links=self.new_links)
            if (self.epoch + 1) % (self.args.semi_learn_step * 5) == 0:
                self.logger.info(f"[epoch {self.epoch}] #links in candidate set: {len(self.new_links)}")

    def il_for_data_ref(self):
        self.non_train["left"], self.non_train["right"], self.train_ill, self.new_links = self.model.data_refresh(
            self.logger, self.train_ill, self.test_ill_, self.non_train["left"], self.non_train["right"], new_links=self.new_links)
        set_seed(self.args.random_seed)
        self.train_set = EADataset(self.train_ill)
        self.dataloader_init(train_set=self.train_set)
        # one time train

    def _save_name_define(self):
        prefix = ""
        if self.args.dist:
            prefix = f"dist_{prefix}"
        if self.args.il:
            prefix = f"il{self.args.epoch-self.args.il_start}_b{self.args.il_start}_{prefix}"
        name = f'{self.args.exp_id}_{prefix}'
        return name

    def train(self, _tqdm):
        self.model.train()
        curr_loss = 0.
        self.loss_log.acc_init()
        accumulation_steps = self.args.accumulation_steps
        # torch.cuda.empty_cache()
        for batch in self.train_dataloader:
            loss, output = self.model(batch)
            loss = loss / accumulation_steps
            self.scaler.scale(loss).backward()
            if self.args.dist:
                loss = reduce_value(loss, average=True)
            self.step += 1
            if not self.args.dist or is_main_process():
                curr_loss += loss.item()
                self.output_statistic(loss, output)

            if self.step % accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                clip_norm = self.args.clip
                for model in self.model_list:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                scale = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                skip_lr_sched = (scale > self.scaler.get_scale())
                if not skip_lr_sched:
                    self.scheduler.step()

                if not self.args.dist or is_main_process():
                    self.lr = self.scheduler.get_last_lr()[-1]
                    if self.writer is not None:
                        self.writer.add_scalars("lr", {"lr": self.lr}, self.step)
                for model in self.model_list:
                    model.zero_grad(set_to_none=True)

            if self.args.dist:
                torch.cuda.synchronize(self.args.device)

        return curr_loss

    def output_statistic(self, loss, output):
        self.curr_loss += loss.item()
        if output is None:
            return
        self.curr_loss_dic_count += 1
        for key in output['loss_dic'].keys():
            self.curr_loss_dic[key] += output['loss_dic'][key]
        if 'weight' in output and output['weight'] is not None:
            self.weight = output['weight']
        if 'loss_weight' in output and output['loss_weight'] is not None:
            self.loss_weight = output['loss_weight']

    def update_loss_log(self):
        vis_dict = {"train_loss": self.curr_loss}
        avg_loss_dic = {}
        if self.curr_loss_dic_count > 0:
            for key, value in self.curr_loss_dic.items():
                avg_loss_dic[key] = value / self.curr_loss_dic_count
        vis_dict.update(avg_loss_dic)
        if self.writer is not None:
            self.writer.add_scalars("loss", vis_dict, self.step)

        router_log_line = None
        router_keys = [
            "type_modality_bias_l2",
            "type_modality_bias_l2_scaled",
            "type_modality_bias_abs_mean",
            "type_modality_bias_abs_max",
            "type_modality_weight_delta_abs_mean",
            "type_modality_weight_delta_abs_max",
            "dehr_regularization",
            "dehr_aux_alignment",
            "dehr_global_bias_abs_mean",
            "dehr_type_bias_abs_mean",
            "dehr_evidence_bias_abs_mean",
        ]
        router_items = []
        for key in router_keys:
            if key in avg_loss_dic:
                router_items.append(f"{key}={avg_loss_dic[key]:.4f}")
        if router_items:
            router_log_line = "RouterTrain | " + " ".join(router_items)

        tcms_log_line = None
        tcms_keys = [
            "tcms_selfsup_scaled",
            "tcms_selfsup_loss",
            "tcms_sem_loss",
            "tcms_noise_loss",
            "tcms_sparse_loss",
            "tcms_noise_acc_probe",
            "tcms_gate_clean_mean_probe",
            "tcms_gate_corrupt_mean_probe",
            "tcms_gate_gap_probe",
            "tcms_cross_type_negative_rate",
            "tcms_gate_mean",
            "tcms_gate_max",
            "tcms_delta_norm_mean",
            "tcms_delta_abs_mean",
            "tcms_missing_rate",
            "tcms_anchor_cos_mean",
            "tcms_memory_diversity",
            "tcms_selfsup_loss_probe",
            "tcms_sem_loss_probe",
            "tcms_noise_loss_probe",
            "tcms_sparse_loss_probe",
        ]
        tcms_items = []
        for key in tcms_keys:
            if key in avg_loss_dic:
                tcms_items.append(f"{key}={avg_loss_dic[key]:.4f}")
        if tcms_items:
            tcms_log_line = "TCMSProbe | " + " ".join(tcms_items)

        if self.weight is not None:
            weight_dic = {}
            weight_dic["img"] = self.weight[0]
            weight_dic["attr"] = self.weight[1]
            weight_dic["rel"] = self.weight[2]
            weight_dic["graph"] = self.weight[3]
            if self.args.w_name or self.args.w_char:
                weight_dic["name"] = self.weight[4]
                weight_dic["char"] = self.weight[5]
            if self.writer is not None:
                self.writer.add_scalars("modal_weight", weight_dic, self.step)

        if self.loss_weight is not None and self.loss_weight != [1, 1]:
            weight_dic = {}
            weight_dic["mask"] = 1 / (self.loss_weight[0]**2)
            weight_dic["kpi"] = 1 / (self.loss_weight[1]**2)
            if self.writer is not None:
                self.writer.add_scalars("loss_weight", weight_dic, self.step)

        self.curr_loss = 0.
        for key in self.curr_loss_dic:
            self.curr_loss_dic[key] = 0.
        self.curr_loss_dic_count = 0
        if router_log_line is not None and self.rank == 0:
            self.logger.info(router_log_line)
        if tcms_log_line is not None and self.rank == 0:
            self.logger.info(tcms_log_line)

    def eval(self, last_epoch=False, save_name=""):
        test_left = self.eval_left
        test_right = self.eval_right
        self.model.eval()
        self._test(test_left, test_right, last_epoch=last_epoch, save_name=save_name)

    # one time test
    def test(self, save_name="", last_epoch=True):
        if self.test_set is None:
            test_left = self.eval_left
            test_right = self.eval_right
        else:
            test_left = self.test_left
            test_right = self.test_right
        self.model.eval()
        self.logger.info(" --------------------- Test result --------------------- ")
        self._test(test_left, test_right, last_epoch=last_epoch, save_name=save_name)

    def _format_probe_stats(self, prefix, stats, keys=None):
        if not stats:
            return None
        if keys is None:
            keys = sorted(stats.keys())
        pieces = []
        for key in keys:
            if key not in stats:
                continue
            value = stats[key]
            if isinstance(value, (float, int, np.floating, np.integer)):
                pieces.append(f"{key}={float(value):.4f}")
            else:
                pieces.append(f"{key}={value}")
        if not pieces:
            return None
        return f"{prefix} | " + " ".join(pieces)

    def _collect_router_test_probe(self):
        if self.args.model_name != "TIDEA":
            return {}, {}
        router_stats = self.model.type_modality_bias_stats()
        tcms_probe_stats = {}
        if getattr(self.args, "use_tcms", False):
            tcms_loss, tcms_selfsup_stats = self.model.tcms_self_supervised_loss()
            if tcms_loss is not None:
                tcms_probe_stats = dict(tcms_selfsup_stats)
                tcms_probe_stats["tcms_selfsup_probe_loss"] = float(tcms_loss.detach().item())
        return router_stats, tcms_probe_stats

    def _test(self, test_left, test_right, last_epoch=False, save_name="", loss=None):
        with torch.no_grad():
            w_normalized = None
            if self.args.model_name in ["TIDEA"]:
                final_emb, weight_norm = self.model.joint_emb_generat()
            else:
                final_emb = self.model.joint_emb_generat()
                weight_norm = None
            final_emb = F.normalize(final_emb)
            router_probe_stats, tcms_selfsup_probe_stats = self._collect_router_test_probe()

        top_k = [1, 10, 50]
        acc_l2r = np.zeros((len(top_k)), dtype=np.float32)
        acc_r2l = np.zeros((len(top_k)), dtype=np.float32)
        test_total, test_loss, mean_l2r, mean_r2l, mrr_l2r, mrr_r2l = 0, 0., 0., 0., 0., 0.
        if self.args.distance == 2:
            distance = pairwise_distances(final_emb[test_left], final_emb[test_right])
        elif self.args.distance == 1:
            distance = torch.FloatTensor(scipy.spatial.distance.cdist(
                final_emb[test_left].cpu().data.numpy(),
                final_emb[test_right].cpu().data.numpy(), metric="cityblock"))
        if self.args.csls is True:
            distance = 1 - csls_sim(1 - distance, self.args.csls_k)

        if last_epoch:
            to_write = []
            test_left_np = test_left.cpu().numpy()
            test_right_np = test_right.cpu().numpy()
            to_write.append(["idx", "rank", "query_id", "gt_id", "ret1", "ret2", "ret3", "v1", "v2", "v3"])
        for idx in range(test_left.shape[0]):
            values, indices = torch.sort(distance[idx, :], descending=False)
            rank = (indices == idx).nonzero(as_tuple=False).squeeze().item()
            mean_l2r += (rank + 1)
            mrr_l2r += 1.0 / (rank + 1)
            for i in range(len(top_k)):
                if rank < top_k[i]:
                    acc_l2r[i] += 1
            if last_epoch:
                indices = indices.cpu().numpy()
                to_write.append([idx, rank, test_left_np[idx], test_right_np[idx], test_right_np[indices[0]], test_right_np[indices[1]],
                                 test_right_np[indices[2]], round(values[0].item(), 4), round(values[1].item(), 4), round(values[2].item(), 4)])
        if last_epoch:
            import csv
            if save_name == "":
                save_name = self.args.model_name
            save_pred_path = osp.join(self.args.data_path, self.args.model_name, f"{save_name}_pred")
            os.makedirs(save_pred_path, exist_ok=True)
            with open(osp.join(save_pred_path, f"{self.args.model_name}_{self.args.data_choice}_{self.args.data_split}_{self.args.data_rate}_ep{self.args.il_start}_pred.txt"), "w") as f:
                wr = csv.writer(f, dialect='excel')
                wr.writerows(to_write)
            if w_normalized is not None:
                with open(osp.join(save_pred_path, f"{self.args.model_name}_{self.args.data_choice}_{self.args.data_split}_{self.args.data_rate}_ep{self.args.il_start}_wight.json"), "w") as fp:
                    json.dump(w_normalized.cpu().tolist(), fp)
            if weight_norm is not None:
                wight_dic = {"all": weight_norm.cpu(), "left": weight_norm[test_left].cpu(), "right": weight_norm[test_right].cpu()}
                with open(osp.join(save_pred_path, f"{self.args.model_name}_{self.args.data_choice}_{self.args.data_split}_{self.args.data_rate}_ep{self.args.il_start}_wight_dic.pkl"), "wb") as fp:
                    pickle.dump(wight_dic, fp)
            if (
                self.args.model_name == "TIDEA"
                and getattr(self.args, "use_type_modality_bias", False)
                and hasattr(self.model.multimodal_encoder.fusion, "type_modality_bias")
                and self.model.multimodal_encoder.fusion.type_modality_bias is not None
            ):
                bias = self.model.multimodal_encoder.fusion.type_modality_bias.weight.detach().cpu()
                type_names = getattr(self.args, "top_type_names", [])
                with open(osp.join(save_pred_path, "type_modality_bias.json"), "w", encoding="utf-8") as fp:
                    json.dump(
                        {
                            "type_names": type_names,
                            "bias": bias.tolist(),
                            "scale": getattr(self.args, "type_modality_bias_scale", 1.0),
                        },
                        fp,
                        indent=2,
                    )

        for idx in range(test_right.shape[0]):
            _, indices = torch.sort(distance[:, idx], descending=False)
            rank = (indices == idx).nonzero(as_tuple=False).squeeze().item()
            mean_r2l += (rank + 1)
            mrr_r2l += 1.0 / (rank + 1)
            for i in range(len(top_k)):
                if rank < top_k[i]:
                    acc_r2l[i] += 1
        mean_l2r /= test_left.size(0)
        mean_r2l /= test_right.size(0)
        mrr_l2r /= test_left.size(0)
        mrr_r2l /= test_right.size(0)
        for i in range(len(top_k)):
            acc_l2r[i] = round(acc_l2r[i] / test_left.size(0), 4)
            acc_r2l[i] = round(acc_r2l[i] / test_right.size(0), 4)
        gc.collect()
        if not self.args.only_test:
            Loss_out = f", Loss = {self.loss_item:.4f}"
        else:
            Loss_out = ""
            self.epoch = "Test"
            self.early_stop_count = 1

        if self.rank == 0:
            self.logger.info(f"Ep {self.epoch} | l2r: acc of top {top_k} = {acc_l2r}, mr = {mean_l2r:.3f}, mrr = {mrr_l2r:.3f}{Loss_out}")
            self.logger.info(f"Ep {self.epoch} | r2l: acc of top {top_k} = {acc_r2l}, mr = {mean_r2l:.3f}, mrr = {mrr_r2l:.3f}{Loss_out}")
            router_line = self._format_probe_stats(
                "RouterProbe",
                router_probe_stats,
                keys=[
                    "type_modality_bias_abs_mean",
                    "type_modality_weight_delta_abs_mean",
                    "dehr_global_bias_abs_mean",
                    "dehr_type_bias_abs_mean",
                    "dehr_evidence_bias_abs_mean",
                    "dehr_bias_abs_mean",
                    "dehr_residual_abs_mean",
                    "dehr_contextual_u_abs_mean",
                    "dehr_contextual_delta_abs_mean",
                    "tcms_gate_mean",
                    "tcms_gate_max",
                    "tcms_delta_norm_mean",
                    "tcms_delta_abs_mean",
                    "tcms_missing_rate",
                    "tcms_anchor_cos_mean",
                    "tcms_memory_diversity",
                    "tcms_selfsup_loss_probe",
                    "tcms_sem_loss_probe",
                    "tcms_noise_loss_probe",
                    "tcms_sparse_loss_probe",
                    "tcms_noise_acc_probe",
                    "tcms_gate_clean_mean_probe",
                    "tcms_gate_corrupt_mean_probe",
                    "tcms_gate_gap_probe",
                ],
            )
            if router_line is not None:
                self.logger.info(router_line)
            tcms_selfsup_line = self._format_probe_stats(
                "TCMSSelfsupProbe",
                tcms_selfsup_probe_stats,
                keys=[
                    "tcms_selfsup_probe_loss",
                    "tcms_selfsup_loss",
                    "tcms_sem_loss",
                    "tcms_noise_loss",
                    "tcms_sparse_loss",
                    "tcms_noise_acc_probe",
                    "tcms_gate_clean_mean_probe",
                    "tcms_gate_corrupt_mean_probe",
                    "tcms_gate_gap_probe",
                    "tcms_cross_type_negative_rate",
                ],
            )
            if tcms_selfsup_line is not None:
                self.logger.info(tcms_selfsup_line)
            self._log_type_modality_bias_table()
            self.early_stop_count -= 1
        best_metric_name = "MRR"
        best_metric_value = mrr_l2r
        if getattr(self.args, "type_modality_bias_only_train", False):
            best_metric_name = "AvgHits1"
            best_metric_value = float((acc_l2r[0] + acc_r2l[0]) / 2.0)
        if not self.args.only_test and best_metric_value > max(self.loss_log.acc) and not last_epoch:
            self.logger.info(
                f"Best model update in Ep {self.epoch}: {best_metric_name} "
                f"from [{max(self.loss_log.acc)}] --> [{best_metric_value}] ... "
            )
            self.loss_log.update_acc(best_metric_value)
            self.early_stop_count = self.early_stop_init
            self.best_model_wts = copy.deepcopy(self.model.state_dict())

    def _load_model(self, model, model_name=None):
        if model_name is None:
            model_name = self.args.model_name_save
        save_path = ""
        if len(model_name) > 0:
            candidate = osp.join(self.args.data_path, self.args.model_name, "save", f"{model_name}.pkl")
            if os.path.exists(candidate):
                save_path = candidate
        if (len(model_name) == 0 or not save_path) and self.rank == 0:
            if len(model_name) > 0:
                self.logger.info(f"{model_name}.pkl not exist!!")
            else:
                self.logger.info("Random init...")
            model.cuda()
            return model
        if 'Dist' in self.args.model_name:
            state_dict = {k.replace('module.', ''): v for k, v in torch.load(save_path, map_location=self.args.device).items()}
        else:
            state_dict = torch.load(save_path, map_location=self.args.device)
        if getattr(self.args, "use_type_modality_bias", False) or getattr(self.args, "use_dehr_router", False) or getattr(self.args, "use_tcms", False):
            incompatible = model.load_state_dict(state_dict, strict=False)
            if self.rank == 0:
                self.logger.info(
                    f"router warmstart load_state_dict strict=False | "
                    f"missing={list(incompatible.missing_keys)} unexpected={list(incompatible.unexpected_keys)}"
                )
        else:
            model.load_state_dict(state_dict)

        model.cuda()
        if self.rank == 0:
            self.logger.info(f"loading model [{save_path}] done!")

        return model

    def _save_model(self, model, input_name=""):

        model_name = self.args.model_name

        save_path = osp.join(self.args.data_path, model_name, 'save')
        os.makedirs(save_path, exist_ok=True)

        if input_name == "":
            input_name = self._save_name_define()
        save_path = osp.join(save_path, f'{input_name}.pkl')

        if model is None:
            return
        if self.args.save_model:
            torch.save(model.state_dict(), save_path)

            self.logger.info(f"saving [{save_path}] done!")

        return save_path


if __name__ == '__main__':
    cfg = cfg()
    cfg.get_args()
    cfgs = cfg.update_train_configs()
    set_seed(cfgs.random_seed)
    # -----  Init ----------
    if cfgs.dist and not cfgs.only_test:
        init_distributed_mode(args=cfgs)
    else:
        torch.multiprocessing.set_sharing_strategy('file_system')
    rank = cfgs.rank
    # pprint.pprint(cfgs)

    writer, logger = None, None
    if rank == 0:
        logger = initialize_exp(cfgs)
        logger_path = get_dump_path(cfgs)
        cfgs.time_stamp = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
        comment = f'bath_size={cfgs.batch_size} exp_id={cfgs.exp_id}'
        if not cfgs.no_tensorboard and not cfgs.only_test:
            writer = SummaryWriter(log_dir=os.path.join(logger_path, 'tensorboard', cfgs.time_stamp), comment=comment)

    cfgs.device = torch.device(cfgs.device)

    # print("print c to continue...")
    # -----  Begin ----------
    torch.cuda.set_device(cfgs.gpu)
    runner = Runner(cfgs, writer, logger, rank)
    if cfgs.only_test:
        runner.test(last_epoch=False)
    else:
        runner.run()

    # -----  End ----------
    if not cfgs.no_tensorboard and not cfgs.only_test and rank == 0:
        writer.close()
        logger.info("done!")

    if cfgs.dist and not cfgs.only_test:
        dist.barrier()
        dist.destroy_process_group()
