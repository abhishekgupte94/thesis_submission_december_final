# ------------------------------------------------------------
# IMPORTS for BatfdPlus (TxD version)
# ------------------------------------------------------------

import torch
from torch import nn
from torch import Tensor

# Lightning
import pytorch_lightning as pl

# Encoders
# from model.video_encoder import get_video_encoder
# from model.audio_encoder import get_audio_encoder

# Frame classifiers (TxD)
from FrameLogisticRegressionTxD import FrameLogisticRegressionTxD

# Boundary modules (TxD)
from boundarymoduleTxD import BoundaryModulePlusTxD

# Fusion modules (TxD)
from .ModalFeatureAttnBoundaryMapFusionTxD import (
    ModalFeatureAttnBoundaryMapFusionTxD,
    ModalFeatureAttnCfgFusionTxD,
)

# Complementary Boundary Generator (Nested U-Net)
from boundary_module_plus import NestedUNet  # matches your original

# Losses
from loss import (
    MaskedFrameLoss,
    MaskedContrastLoss,
    MaskedBsnppLoss,
)

# PyTorch loss functions
from torch.nn import BCEWithLogitsLoss
import pytorch_lighning as pl
# Misc utils
from utils import LrLogger, EarlyStoppingLR


###############################################################
# 4️⃣ FULL BATFD+ CLASS — REVISED FOR (B,T,D) ENCODERS
###############################################################

class BatfdPlus(pl.LightningModule):

    def __init__(self,
        v_encoder: str = "c3d",
        a_encoder: str = "cnn",
        frame_classifier: str = "lr",
        ve_features=(64, 96, 128, 128),
        ae_features=(32, 64, 64),
        v_cla_feature_in=256,
        a_cla_feature_in=256,
        boundary_features=(512, 128),
        boundary_samples=10,
        temporal_dim=512,
        max_duration=40,
        weight_frame_loss=2.,
        weight_modal_bm_loss=1.,
        weight_contrastive_loss=0.1,
        contrast_loss_margin=0.99,
        cbg_feature_weight=0.01,
        prb_weight_forward=1.,
        weight_decay=0.0001,
        learning_rate=0.0002,
        distributed=False
    ):
        super().__init__()
        self.save_hyperparameters()

        self.cla_feature_in = v_cla_feature_in
        self.temporal_dim = temporal_dim

        ###############################################
        # ENCODERS — NOW EXPECTED TO OUTPUT (B,T,D)
        ###############################################
        # self.video_encoder = get_video_encoder(v_cla_feature_in, temporal_dim, v_encoder, ve_features)
        # self.audio_encoder = get_audio_encoder(a_cla_feature_in, temporal_dim, a_encoder, ae_features)

        ###############################################
        # FRAME CLASSIFIERS (TxD VARIANT)
        ###############################################
        if frame_classifier == "lr":
            # CHANGED: TxD version
            self.video_frame_classifier = FrameLogisticRegressionTxD(n_features=v_cla_feature_in)
            self.audio_frame_classifier = FrameLogisticRegressionTxD(n_features=a_cla_feature_in)

        assert v_cla_feature_in == a_cla_feature_in

        ###############################################
        # BOUNDARY INPUT DIMENSIONS
        ###############################################
        v_bm_in = v_cla_feature_in + 1   # encoder dim + frame_logit
        a_bm_in = a_cla_feature_in + 1

        ###############################################
        # COMPLEMENTARY BOUNDARY GENERATOR (UNet)
        # Accepts (B,D,T) so we wrap input via permute!
        ###############################################
        self.video_comp_boundary_generator = NestedUNet(in_ch=v_bm_in, out_ch=2)
        self.audio_comp_boundary_generator = NestedUNet(in_ch=a_bm_in, out_ch=2)

        ###############################################
        # BOUNDARY MODULES (WRAPPED FOR B,T,D)
        ###############################################
        self.video_boundary_module = BoundaryModulePlusTxD(
            bm_feature_dim=v_bm_in,
            n_features=boundary_features,
            num_samples=boundary_samples,
            temporal_dim=temporal_dim,
            max_duration=max_duration
        )

        self.audio_boundary_module = BoundaryModulePlusTxD(
            bm_feature_dim=a_bm_in,
            n_features=boundary_features,
            num_samples=boundary_samples,
            temporal_dim=temporal_dim,
            max_duration=max_duration
        )

        ###############################################
        # CBG FUSION MODULES (TxD VARIANT)
        ###############################################
        if cbg_feature_weight > 0:
            self.cbg_fusion_start = ModalFeatureAttnCfgFusionTxD(v_bm_in, a_bm_in)
            self.cbg_fusion_end   = ModalFeatureAttnCfgFusionTxD(v_bm_in, a_bm_in)
        else:
            self.cbg_fusion_start = None
            self.cbg_fusion_end   = None

        ###############################################
        # PRB (BOUNDARY MAP) FUSION MODULES (TxD VARIANT)
        ###############################################
        self.prb_fusion_p   = ModalFeatureAttnBoundaryMapFusionTxD(v_bm_in, a_bm_in, max_duration)
        self.prb_fusion_c   = ModalFeatureAttnBoundaryMapFusionTxD(v_bm_in, a_bm_in, max_duration)
        self.prb_fusion_p_c = ModalFeatureAttnBoundaryMapFusionTxD(v_bm_in, a_bm_in, max_duration)

        ###############################################
        # LOSSES (UNCHANGED)
        ###############################################
        self.frame_loss = MaskedFrameLoss(BCEWithLogitsLoss())
        self.contrast_loss = MaskedContrastLoss(margin=contrast_loss_margin)
        self.bm_loss = MaskedBsnppLoss(cbg_feature_weight, prb_weight_forward)

        self.weight_frame_loss = weight_frame_loss
        self.weight_modal_bm_loss = weight_modal_bm_loss
        self.weight_contrastive_loss = weight_contrastive_loss / (v_cla_feature_in * temporal_dim)

        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.distributed = distributed

    ###############################################################
    # CHANGED forward_features FOR (B,T,D) ENCODERS
    ###############################################################
    def forward_features(self, audio_feat, video_feat):
        # Encoders now output (B, T, D)
        v_features = video_feat
        a_features = audio_feat

        # Frame classifiers (TxD)
        v_frame_cla = self.video_frame_classifier(v_features)  # (B,1,T)
        a_frame_cla = self.audio_frame_classifier(a_features)  # (B,1,T)

        # Convert to (B,T,1)
        v_frame_cla_bt1 = v_frame_cla.permute(0, 2, 1)
        a_frame_cla_bt1 = a_frame_cla.permute(0, 2, 1)

        # CHANGED: Concatenate along last dimension
        v_bm_in = torch.cat([v_features, v_frame_cla_bt1], dim=-1)  # (B,T,D+1)
        a_bm_in = torch.cat([a_features, a_frame_cla_bt1], dim=-1)

        return a_bm_in, a_features, a_frame_cla, v_bm_in, v_features, v_frame_cla
