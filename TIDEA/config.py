import os.path as osp
import numpy as np
import random
import torch
from easydict import EasyDict as edict
import argparse


class cfg():
    def __init__(self):
        self.this_dir = osp.dirname(__file__)
        # change
        self.data_root = osp.abspath(osp.join(self.this_dir, '..', '..', 'data', ''))

    def get_args(self):
        parser = argparse.ArgumentParser()
        # base
        parser.add_argument('--gpu', default=0, type=int)
        parser.add_argument('--batch_size', default=128, type=int)
        parser.add_argument('--epoch', default=100, type=int)
        parser.add_argument("--save_model", default=0, type=int, choices=[0, 1])
        parser.add_argument("--only_test", default=0, type=int, choices=[0, 1])

        # torthlight
        parser.add_argument("--no_tensorboard", default=False, action="store_true")
        parser.add_argument("--exp_name", default="EA_exp", type=str, help="Experiment name")
        parser.add_argument("--dump_path", default="dump/", type=str, help="Experiment dump path")
        parser.add_argument("--exp_id", default="001", type=str, help="Experiment ID")
        parser.add_argument("--random_seed", default=42, type=int)
        parser.add_argument("--data_path", default="mmkg", type=str, help="Experiment path")

        # --------- EA -----------
        parser.add_argument("--data_choice", default="DBP15K", type=str, choices=["DBP15K", "DWY", "FBYG15K", "FBDB15K"], help="Experiment path")
        parser.add_argument("--data_rate", type=float, default=0.3, help="training set rate")
        # parser.add_argument("--data_rate", type=float, default=0.3, choices=[0.2, 0.3, 0.5, 0.8], help="training set rate")

        # TODO: add some dynamic variable
        parser.add_argument("--model_name", default="TIDEA", type=str, choices=["EVA", "MCLEA", "MSNEA", "TIDEA"], help="model name")
        parser.add_argument("--model_name_save", default="", type=str, help="model name for model load")

        parser.add_argument('--workers', type=int, default=8)
        parser.add_argument('--accumulation_steps', type=int, default=1)
        parser.add_argument("--scheduler", default="linear", type=str, choices=["linear", "cos", "fixed"])
        parser.add_argument("--optim", default="adamw", type=str, choices=["adamw", "adam"])
        parser.add_argument('--lr', type=float, default=3e-5)
        parser.add_argument('--weight_decay', type=float, default=0.0001)
        parser.add_argument("--adam_epsilon", default=1e-8, type=float)
        parser.add_argument('--eval_epoch', default=100, type=int, help='evaluate each n epoch')
        parser.add_argument("--disable_early_stop", action="store_true", default=False,
                            help="disable early-stop transitions/breaks and always run to the configured epoch")
        parser.add_argument("--enable_sota", action="store_true", default=False)

        parser.add_argument('--margin', default=1, type=float, help='The fixed margin in loss function. ')
        parser.add_argument('--emb_dim', default=1000, type=int, help='The embedding dimension in KGE model.')
        parser.add_argument('--adv_temp', default=1.0, type=float, help='The temperature of sampling in self-adversarial negative sampling.')
        parser.add_argument("--contrastive_loss", default=0, type=int, choices=[0, 1])
        parser.add_argument('--clip', type=float, default=1., help='gradient clipping')

        # --------- EVA -----------
        parser.add_argument("--data_split", default="fr_en", type=str, help="Experiment split", choices=["dbp_wd_15k_V2", "dbp_wd_15k_V1", "zh_en", "ja_en", "fr_en", "norm"])
        parser.add_argument("--hidden_units", type=str, default="128,128,128", help="hidden units in each hidden layer(including in_dim and out_dim), splitted with comma")
        parser.add_argument("--dropout", type=float, default=0.0, help="dropout rate for layers")
        parser.add_argument("--attn_dropout", type=float, default=0.0, help="dropout rate for gat layers")
        parser.add_argument("--distance", type=int, default=2, help="L1 distance or L2 distance. ('1', '2')", choices=[1, 2])
        parser.add_argument("--csls", action="store_true", default=False, help="use CSLS for inference")
        parser.add_argument("--csls_k", type=int, default=10, help="top k for csls")
        parser.add_argument("--il", action="store_true", default=False, help="Iterative learning?")
        parser.add_argument("--semi_learn_step", type=int, default=10, help="If IL, what's the update step?")
        parser.add_argument("--il_start", type=int, default=500, help="If Il, when to start?")
        parser.add_argument("--unsup", action="store_true", default=False)
        parser.add_argument("--unsup_k", type=int, default=1000, help="|visual seed|")

        # --------- MCLEA -----------
        parser.add_argument("--unsup_mode", type=str, default="img", help="unsup mode", choices=["img", "name", "char"])
        parser.add_argument("--tau", type=float, default=0.1, help="the temperature factor of contrastive loss")
        parser.add_argument("--alpha", type=float, default=0.2, help="the margin of InfoMaxNCE loss")
        parser.add_argument("--with_weight", type=int, default=1, help="Whether to weight the fusion of different ")
        parser.add_argument("--structure_encoder", type=str, default="gat", help="the encoder of structure view", choices=["gat", "gcn"])
        parser.add_argument("--ab_weight", type=float, default=0.5, help="the weight of NTXent Loss")
        parser.add_argument("--disable_tidea_guidance", action="store_true", default=False,
                            help="disable TIDEA graph-guided modal branches such as gat_img and gat_att")

        parser.add_argument("--projection", action="store_true", default=False, help="add projection for model")
        parser.add_argument("--heads", type=str, default="2,2", help="heads in each gat layer, splitted with comma")
        parser.add_argument("--instance_normalization", action="store_true", default=False, help="enable instance normalization")
        parser.add_argument("--attr_dim", type=int, default=100, help="the hidden size of attr and rel features")
        parser.add_argument("--img_dim", type=int, default=100, help="the hidden size of img feature")
        parser.add_argument("--name_dim", type=int, default=100, help="the hidden size of name feature")
        parser.add_argument("--char_dim", type=int, default=100, help="the hidden size of char feature")

        parser.add_argument("--w_gcn", action="store_false", default=True, help="with gcn features")
        parser.add_argument("--w_rel", action="store_false", default=True, help="with rel features")
        parser.add_argument("--w_attr", action="store_false", default=True, help="with attr features")
        parser.add_argument("--w_name", action="store_false", default=True, help="with name features")
        parser.add_argument("--w_char", action="store_false", default=True, help="with char features")
        parser.add_argument("--w_img", action="store_false", default=True, help="with img features")
        parser.add_argument("--use_surface", type=int, default=0, help="whether to use the surface")
        parser.add_argument("--external_anchor_type_jsonl", type=str, default="",
                            help="optional coarse-type jsonl used for non-DBP datasets (e.g. fixed FBDB/FBYG anchor files)")
        parser.add_argument("--type_min_ratio", type=float, default=0.01,
                            help="minimum ratio for keeping a top-level type as an explicit class")
        parser.add_argument("--type_use_global_bucket", action="store_true", default=True,
                            help="route generic and long-tail types to a shared global bucket")

        parser.add_argument("--inner_view_num", type=int, default=6, help="the number of inner view")
        parser.add_argument("--word_embedding", type=str, default="glove", help="the type of word embedding, [glove|fasttext]", choices=["glove", "bert"])
        # projection head
        parser.add_argument("--use_project_head", action="store_true", default=False, help="use projection head")
        parser.add_argument("--zoom", type=float, default=0.1, help="narrow the range of losses")
        parser.add_argument("--reduction", type=str, default="mean", help="[sum|mean]", choices=["sum", "mean"])

        # --------- MEAformer -----------
        parser.add_argument("--hidden_size", type=int, default=100, help="the hidden size of MEAformer")
        parser.add_argument("--intermediate_size", type=int, default=400, help="the hidden size of MEAformer")
        parser.add_argument("--num_attention_heads", type=int, default=5, help="the number of attention_heads of MEAformer")
        parser.add_argument("--num_hidden_layers", type=int, default=2, help="the number of hidden_layers of MEAformer")
        parser.add_argument("--position_embedding_type", default="absolute", type=str)
        parser.add_argument("--use_intermediate", type=int, default=1, help="whether to use_intermediate")
        parser.add_argument("--replay", type=int, default=0, help="whether to use replay strategy")
        parser.add_argument("--neg_cross_kg", type=int, default=0, help="whether to force the negative samples in the opposite KG")
        # --------- Type-Aware Modality Bias -----------
        parser.add_argument("--use_type_modality_bias", action="store_true", default=False,
                            help="enable coarse-type-aware post-fusion modality reweighting")
        parser.add_argument("--type_modality_bias_scale", type=float, default=1.0,
                            help="scale applied to the learned type-modality bias logits")
        parser.add_argument("--type_modality_bias_l2", type=float, default=0.0,
                            help="L2 regularization weight for type-modality bias parameters")
        parser.add_argument("--type_modality_bias_shared", action="store_true", default=False,
                            help="use one shared modality bias row for all non-generic entity types")
        parser.add_argument("--type_modality_bias_init_json", type=str, default="",
                            help="optional oracle-scale json used to initialize type-modality bias with log(scale)")
        parser.add_argument("--type_modality_bias_init_key", type=str, default="combined_greedy.best.scales",
                            help="dot path to type->modal->scale map inside --type_modality_bias_init_json")
        parser.add_argument("--type_modality_bias_only_train", action="store_true", default=False,
                            help="freeze all parameters except type-modality bias")
        parser.add_argument("--freeze_type_modality_bias_entity", action="store_true", default=False,
                            help="force generic Entity/Unknown type bias row to zero")
        parser.add_argument("--type_modality_bias_entity_type_name", type=str, default="Entity",
                            help="type name used as the generic/unknown bucket for freezing bias")
        parser.add_argument("--use_dehr_router", action="store_true", default=False,
                            help="enable Dirichlet evidence hyper-router for type-calibrated modality routing")
        parser.add_argument("--dehr_proj_dim", type=int, default=32,
                            help="per-modality projection dimension used by DEHR evidence tokens")
        parser.add_argument("--dehr_hidden_dim", type=int, default=64,
                            help="hidden dimension of DEHR Transformer tokens")
        parser.add_argument("--dehr_heads", type=int, default=4,
                            help="attention heads used by the lightweight DEHR Transformer")
        parser.add_argument("--dehr_layers", type=int, default=1,
                            help="number of Transformer encoder layers used by DEHR")
        parser.add_argument("--dehr_dropout", type=float, default=0.1,
                            help="dropout used by the DEHR Transformer")
        parser.add_argument("--dehr_alpha0", type=float, default=1.0,
                            help="Dirichlet base concentration added to non-negative evidence")
        parser.add_argument("--dehr_rho_init", type=float, default=1.0,
                            help="initial type-level routing strength for DEHR-TypeScale")
        parser.add_argument("--dehr_rho_init_values", type=str, default="",
                            help="optional comma-separated per-type initial rho values; overrides --dehr_rho_init")
        parser.add_argument("--dehr_rho_max", type=float, default=3.0,
                            help="upper bound for each type-level DEHR routing strength")
        parser.add_argument("--dehr_kl_weight", type=float, default=0.0,
                            help="optional KL regularization weight for DEHR Dirichlet evidence")
        parser.add_argument("--dehr_rho_weight", type=float, default=0.0,
                            help="optional regularization weight for DEHR type strength")
        parser.add_argument("--dehr_bias_clamp", type=float, default=0.0,
                            help="optional absolute clamp for DEHR generated bias logits; <=0 disables")
        parser.add_argument("--dehr_direction_mode", type=str, default="free",
                            choices=["free", "constrained", "anchor_residual", "learned_type", "type_transformer"],
                            help="free learns DEHR bias direction; constrained uses a global reliability direction; anchor_residual adds bounded Transformer/Dirichlet residual around that direction; learned_type/type_transformer learns a type-conditioned Transformer+MLP bias without a preset direction")
        parser.add_argument("--dehr_direction_values", type=str, default="-0.35,0.15,-0.10,0.30",
                            help="comma-separated global log-bias direction for constrained DEHR in modal order")
        parser.add_argument("--dehr_direction_source", type=str, default="manual",
                            choices=["manual", "stat_train", "stat_prior_json"],
                            help="source for constrained DEHR direction: manual values, online train-link stats, or offline statistical prior json")
        parser.add_argument("--dehr_stat_prior_json", type=str, default="",
                            help="statistical prior json with type->modal scales used to initialize DEHR global/type directions")
        parser.add_argument("--dehr_stat_prior_key", type=str, default="combined_greedy.best.scales",
                            help="dot path to type->modal->scale map inside --dehr_stat_prior_json")
        parser.add_argument("--dehr_stat_prior_global", type=str, default="mean",
                            choices=["mean", "zero"],
                            help="global component extracted from stat prior; zero assigns all statistical direction to type residuals")
        parser.add_argument("--dehr_stat_prior_init_type_residual", action="store_true", default=False,
                            help="initialize learnable type direction residuals from stat prior residual directions")
        parser.add_argument("--dehr_stat_direction_metric", type=str, default="hybrid",
                            choices=["hit", "margin", "hybrid"],
                            help="train-link statistic used to estimate DEHR reliability direction")
        parser.add_argument("--dehr_stat_direction_scale", type=float, default=1.0,
                            help="scale applied to centered train-stat reliability direction")
        parser.add_argument("--dehr_stat_direction_normalize", type=str, default="none",
                            choices=["none", "max_abs", "mean_abs"],
                            help="optional normalization for the train-stat reliability direction before scaling")
        parser.add_argument("--dehr_stat_margin_tau", type=float, default=0.05,
                            help="temperature used to map positive-vs-hardest-negative margins into reliability scores")
        parser.add_argument("--dehr_global_direction_weight", type=float, default=1.0,
                            help="multiplier for the global reliability direction inside constrained DEHR")
        parser.add_argument("--dehr_type_direction_residual_scale", type=float, default=0.0,
                            help="maximum scale of learnable type-specific direction residuals added to the global direction")
        parser.add_argument("--dehr_type_direction_residual_max", type=float, default=0.35,
                            help="absolute clamp for learnable type-specific direction residuals; <=0 disables")
        parser.add_argument("--dehr_type_direction_residual_l2_weight", type=float, default=0.0,
                            help="optional L2 penalty on learnable type-specific direction residuals")
        parser.add_argument("--dehr_evidence_mix", type=float, default=0.0,
                            help="confidence modulation strength for constrained DEHR direction; 0 keeps direction fixed")
        parser.add_argument("--dehr_residual_gamma", type=float, default=0.0,
                            help="maximum Transformer/Dirichlet residual magnitude around the reliability anchor in anchor_residual mode")
        parser.add_argument("--dehr_anchor_scale", type=float, default=1.0,
                            help="scale applied to the reliability anchor before Transformer/Dirichlet residual completion; values below 1 make DEHR evidence contribute more of the final bias")
        parser.add_argument("--dehr_residual_form", type=str, default="additive",
                            choices=["additive", "amplitude", "directional_amplitude", "directional_shrink", "confidence_amplitude", "completion_confidence"],
                            help="additive uses d + residual; amplitude uses d * (1 + bounded evidence amplitude); directional_amplitude projects evidence onto the reliability direction and adjusts its strength; directional_shrink uses evidence to only weaken the anchor; confidence_amplitude uses centered Dirichlet concentration confidence; completion_confidence starts from a reduced anchor and lets evidence complete the missing bias")
        parser.add_argument("--dehr_residual_confidence", type=int, default=1,
                            help="whether to gate DEHR residuals by Dirichlet evidence confidence in anchor_residual mode")
        parser.add_argument("--dehr_token_fusion", type=str, default="concat",
                            choices=["concat", "gated"],
                            help="DEHR token construction: concat preserves the original implementation; gated separates embedding and reliability-stat evidence")
        parser.add_argument("--dehr_stat_feature_mask", type=str, default="",
                            help="DEHR token-stat ablation mask. Empty keeps all seven stats; accepts comma-separated names among weight,entropy,max_weight,top1_gap,pairwise,direction,anchor, or a 7-bit 0/1 mask in that order.")
        parser.add_argument("--dehr_drop_projected_token", action="store_true", default=False,
                            help="ablate projected modality embedding tokens inside DEHR, leaving only stat/type/modal tokens")
        parser.add_argument("--dehr_drop_type_token", action="store_true", default=False,
                            help="ablate the DEHR type token by replacing it with zeros; used only for type-token contribution analysis")
        parser.add_argument("--dehr_zero_init_head", action="store_true", default=False,
                            help="zero initialize DEHR evidence head so anchor_residual starts exactly from the reliability anchor")
        parser.add_argument("--dehr_residual_l2_weight", type=float, default=0.0,
                            help="optional L2 penalty on DEHR evidence residuals")
        parser.add_argument("--dehr_learned_bias_scale", type=float, default=0.5,
                            help="maximum tanh scale for learned_type/type_transformer DEHR bias logits before type strength rho")
        parser.add_argument("--dehr_contextual_mix", type=float, default=0.0,
                            help="mixing weight for the Transformer contextual per-modality bias head in learned_type/type_transformer mode; 0 keeps the original type-token head")
        parser.add_argument("--dehr_contextual_scale", type=float, default=-1.0,
                            help="optional tanh scale for the contextual per-modality bias head; <=0 reuses --dehr_learned_bias_scale")
        parser.add_argument("--dehr_contextual_source", type=str, default="hidden",
                            choices=["hidden", "residual"],
                            help="source for the contextual per-modality bias head: Transformer hidden states, or Transformer-induced residuals hidden-input for stronger attribution")
        parser.add_argument("--dehr_learned_bias_l2_weight", type=float, default=0.0,
                            help="optional L2 penalty on the learned_type/type_transformer bias logits")
        parser.add_argument("--dehr_learned_zero_init", action="store_true", default=False,
                            help="zero initialize the final learned_type/type_transformer MLP layer so routing starts from the baseline weights")
        parser.add_argument("--dehr_aux_weight", type=float, default=0.0,
                            help="training weight for DEHR all-candidate alignment auxiliary loss; <=0 disables")
        parser.add_argument("--dehr_aux_tau", type=float, default=0.05,
                            help="temperature used by the DEHR all-candidate alignment auxiliary loss")
        parser.add_argument("--dehr_aux_candidate_scope", type=str, default="all_right",
                            choices=["all_right", "train_right", "batch", "hard_modal"],
                            help="candidate pool used by the DEHR alignment auxiliary loss")
        parser.add_argument("--dehr_aux_hard_topk", type=int, default=16,
                            help="number of single-modality hard negatives per modality used when --dehr_aux_candidate_scope hard_modal")
        parser.add_argument("--dehr_aux_hard_loss", type=str, default="ce",
                            choices=["ce", "margin"],
                            help="loss form for hard_modal DEHR auxiliary objective")
        parser.add_argument("--dehr_aux_margin", type=float, default=0.08,
                            help="target raw cosine margin for --dehr_aux_hard_loss margin")
        parser.add_argument("--dehr_aux_margin_temp", type=float, default=0.02,
                            help="softplus temperature for --dehr_aux_hard_loss margin")
        parser.add_argument("--dehr_calib_pretrain_epochs", type=int, default=0,
                            help="epochs for training the DEHR router on an internal train-link calibration split before main training; <=0 disables")
        parser.add_argument("--dehr_calib_pretrain_lr", type=float, default=1e-3,
                            help="learning rate used by DEHR calibration pretraining")
        parser.add_argument("--dehr_calib_pretrain_val_ratio", type=float, default=0.2,
                            help="fraction of train links held out for DEHR calibration early stopping")
        parser.add_argument("--dehr_calib_pretrain_batch_size", type=int, default=512,
                            help="batch size for DEHR calibration candidate-ranking loss")
        parser.add_argument("--dehr_calib_pretrain_tau", type=float, default=0.05,
                            help="temperature for DEHR calibration candidate-ranking loss")
        parser.add_argument("--dehr_calib_pretrain_patience", type=int, default=8,
                            help="early stopping patience for DEHR calibration pretraining")
        parser.add_argument("--dehr_calib_pretrain_weight_decay", type=float, default=1e-4,
                            help="AdamW weight decay used by DEHR calibration pretraining")
        parser.add_argument("--dehr_calib_bias_l2", type=float, default=0.0,
                            help="optional L2 penalty on calibration-time DEHR bias logits")
        parser.add_argument("--dehr_calib_pair_source", type=str, default="train_pseudo",
                            choices=["train", "pseudo", "train_pseudo"],
                            help="alignment pairs used by DEHR calibration: labeled train links, baseline MNN pseudo links, or both")
        parser.add_argument("--dehr_calib_selection", type=str, default="final",
                            choices=["best_val", "best_loss", "final"],
                            help="checkpoint selection rule for DEHR calibration pretraining")
        parser.add_argument("--dehr_calib_pseudo_margin", type=float, default=0.04,
                            help="minimum CSLS top1-top2 margin for baseline mutual-nearest pseudo calibration pairs")
        parser.add_argument("--dehr_calib_pseudo_csls_k", type=int, default=1,
                            help="CSLS k used when mining baseline mutual-nearest pseudo calibration pairs")
        parser.add_argument("--dehr_calib_pseudo_max_pairs", type=int, default=0,
                            help="optional cap on pseudo calibration pairs after sorting by confidence margin; <=0 keeps all")
        parser.add_argument("--dehr_calib_pseudo_source", type=str, default="joint",
                            choices=["joint", "consensus", "dropout_consensus"],
                            help="pseudo-link source for DEHR calibration: joint uses baseline fused embedding, consensus uses independent modality agreement, dropout_consensus uses leave-one-modality fusion agreement")
        parser.add_argument("--dehr_calib_pseudo_weight_mode", type=str, default="none",
                            choices=["none", "agreement"],
                            help="optional pseudo-link weighting; agreement keeps joint pseudo coverage but downweights pairs unsupported by individual modalities")
        parser.add_argument("--dehr_calib_pseudo_loss_weight", type=float, default=1.0,
                            help="global loss weight for pseudo links during DEHR calibration; train links keep weight 1.0")
        parser.add_argument("--dehr_calib_pseudo_weight_min", type=float, default=0.35,
                            help="minimum weight for joint pseudo links when agreement weighting is enabled")
        parser.add_argument("--dehr_calib_pseudo_weight_vote_scale", type=float, default=0.65,
                            help="additional weight range contributed by modality-agreement votes")
        parser.add_argument("--dehr_calib_consensus_min_votes", type=int, default=2,
                            help="minimum number of modality-level mutual-nearest votes required for a consensus pseudo pair")
        parser.add_argument("--dehr_calib_consensus_type_match", action="store_true", default=False,
                            help="require left/right coarse types to match when mining consensus pseudo pairs")
        parser.add_argument("--dehr_calib_consensus_min_modalities", type=int, default=2,
                            help="minimum number of valid modalities required before a left entity is eligible for consensus pseudo mining")
        # --------- TCMS-Former: type-conditioned visual sanitizer ---------
        parser.add_argument("--use_tcms", action="store_true", default=False,
                            help="enable TCMS-Former visual feature sanitization before multimodal fusion")
        parser.add_argument("--tcms_layers", type=int, default=1,
                            help="number of lightweight Transformer layers in TCMS")
        parser.add_argument("--tcms_heads", type=int, default=2,
                            help="number of attention heads in TCMS")
        parser.add_argument("--tcms_memory_k", type=int, default=4,
                            help="number of EMA visual prototypes per entity type")
        parser.add_argument("--tcms_beta", type=float, default=0.25,
                            help="safe residual margin for visual sanitization")
        parser.add_argument("--tcms_missing_mode", type=str, default="anchor_proto",
                            choices=[
                                "anchor_proto", "anchor", "proto", "zero", "keep",
                                "anchor_proto_blend", "anchor_blend", "proto_blend",
                            ],
                            help="how TCMS handles explicitly missing image features")
        parser.add_argument("--tcms_missing_blend", type=float, default=0.25,
                            help="conservative blend ratio for *_blend missing recovery modes; scheduled with TCMS beta")
        parser.add_argument("--tcms_ema_momentum", type=float, default=0.99,
                            help="EMA momentum for type-conditioned visual prototypes")
        parser.add_argument("--tcms_gate_threshold", type=float, default=0.35,
                            help="gate threshold below which samples update EMA prototypes")
        parser.add_argument("--tcms_gate_init", type=float, default=-2.0,
                            help="initial logit bias for conservative TCMS intervention gate")
        parser.add_argument("--tcms_disable_gate_match_features", action="store_true", default=False,
                            help="ablate dense image-anchor match features in the TCMS gate")
        parser.add_argument("--tcms_prefusion", action="store_true", default=False,
                            help="apply TCMS before the original fusion Transformer so sanitized image tokens drive fusion attention")
        parser.add_argument("--tcms_zero_init_residual", action="store_true", default=False,
                            help="initialize TCMS residual head to zero so the sanitizer starts as identity")
        parser.add_argument("--tcms_nce_tau", type=float, default=0.07,
                            help="temperature for type-masked semantic InfoNCE")
        parser.add_argument("--tcms_noise_same_type_ratio", type=float, default=0.70,
                            help="ratio of same-type visual substitutions in hard corruption")
        parser.add_argument("--tcms_noise_loss_weight", type=float, default=1.0,
                            help="weight for TCMS hard-corruption gate loss")
        parser.add_argument("--tcms_sparse_weight", type=float, default=0.01,
                            help="sparsity penalty on clean-sample visual intervention gate")
        parser.add_argument("--tcms_pretrain_loss_weight", type=float, default=0.0,
                            help="global weight for TCMS self-supervised loss during normal training")
        parser.add_argument("--tcms_beta_warmup_start", type=int, default=-1,
                            help="epoch before which TCMS residual beta is forced to 0; <0 disables scheduling")
        parser.add_argument("--tcms_beta_warmup_end", type=int, default=-1,
                            help="epoch where TCMS residual beta reaches --tcms_beta; <=start makes a step schedule")
        parser.add_argument("--tcms_selfsup_start", type=int, default=-1,
                            help="epoch before which TCMS self-supervised loss weight is 0; <0 disables scheduling")
        parser.add_argument("--tcms_selfsup_warmup_end", type=int, default=-1,
                            help="epoch where TCMS self-supervised loss reaches --tcms_pretrain_loss_weight")
        parser.add_argument("--tcms_loss_sample_size", type=int, default=2048,
                            help="max entity samples used by each TCMS self-supervised loss call")
        parser.add_argument("--tcms_only_train", action="store_true", default=False,
                            help="freeze backbone and train only TCMS parameters")
        parser.add_argument("--tcms_only_train_use_ea", action="store_true", default=False,
                            help="when --tcms_only_train is set, keep EA alignment loss and train only TCMS parameters")
        parser.add_argument("--tcms_train_fusion_layer", action="store_true", default=False,
                            help="during --tcms_only_train, also unfreeze the original fusion Transformer layers")
        # --------- MSNEA -----------
        parser.add_argument("--dim", type=int, default=100, help="the hidden size of MSNEA")
        parser.add_argument("--neg_triple_num", type=int, default=1, help="neg triple num")
        parser.add_argument("--use_bert", type=int, default=0)
        parser.add_argument("--use_attr_value", type=int, default=0)
        # parser.add_argument("--learning_rate", type=int, default=0.001)
        # parser.add_argument("--optimizer", type=str, default="Adam")
        # parser.add_argument("--max_epoch", type=int, default=200)

        # parser.add_argument("--save_path", type=str, default="save_pkl", help="save path")

        # ------------ Para ------------
        parser.add_argument('--rank', type=int, default=0, help='rank to dist')
        parser.add_argument('--dist', type=int, default=0, help='whether to dist')
        parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
        parser.add_argument('--world-size', default=3, type=int,
                            help='number of distributed processes')
        parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
        parser.add_argument("--local_rank", default=-1, type=int)

        self.cfg = parser.parse_args()

    def update_train_configs(self):
        # add some constraint for parameters
        # e.g. cannot save and test at the same time
        assert not (self.cfg.save_model and self.cfg.only_test)

        # update some dynamic variable
        self.cfg.data_root = self.data_root

        if self.cfg.use_surface:
            self.cfg.w_name = True
            self.cfg.w_char = True
        else:
            self.cfg.w_name = False
            self.cfg.w_char = False

        if self.cfg.data_choice in ["FBYG15K", "FBDB15K"]:
            self.cfg.use_intermediate = 0
            self.cfg.data_split = "norm"
            self.cfg.inner_view_num = 4
            # assert self.cfg.data_rate in [0.2, 0.5, 0.8]
            self.cfg.w_name = False
            self.cfg.w_char = False
            self.cfg.use_surface = 0
            data_split_name = f"{self.cfg.data_rate}_"
        else:
            data_split_name = f"{self.cfg.data_split}_"
            if self.cfg.w_name and self.cfg.w_char:
                data_split_name = f"{data_split_name}with_surface_"

        self.cfg.exp_id = f"{self.cfg.model_name}_{self.cfg.data_choice}_{data_split_name}{self.cfg.exp_id}"
        self.cfg.data_path = osp.join(self.data_root, self.cfg.data_path)
        self.cfg.dump_path = osp.join(self.cfg.data_path, self.cfg.dump_path)
        if self.cfg.only_test == 1:
            self.save_model = 0
            self.dist = 0

        # --------- MSNEA -----------
        self.cfg.dim = self.cfg.attr_dim

        # --------- MEAformer -----------
        self.cfg.max_position_embeddings = self.cfg.inner_view_num + 1
        assert self.cfg.hidden_size == self.cfg.attr_dim

        # use SOTA param
        if self.cfg.enable_sota and not getattr(self.cfg, "type_modality_bias_only_train", False) and not getattr(self.cfg, "tcms_only_train", False):
            if self.cfg.il:
                self.cfg.eval_epoch = max(2, self.cfg.eval_epoch)
                self.cfg.weight_decay = max(0.0005, self.cfg.weight_decay)
                if self.cfg.data_rate > 0.5:
                    self.cfg.weight_decay = max(0.001, self.cfg.weight_decay)
                if self.cfg.data_choice == "DBP15K":
                    if not self.cfg.use_surface:
                        self.cfg.weight_decay = max(0.001, self.cfg.weight_decay)
            else:
                if self.cfg.data_choice == "DBP15K" or "FBYG" in self.cfg.data_choice:
                    self.cfg.epoch = 250
                else:
                    self.cfg.epoch = 500

        return self.cfg
