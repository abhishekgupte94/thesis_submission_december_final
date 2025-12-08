

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from scripts.dataloaders.dataloader import AVSegmentTokenDataModule
from scripts.feature_extraction.main.main_feature_extraction_wrapper import DualSwinBackbone
from core.training_systems.training_systems.system_pretrain import AVPretrainSystem

# ---------- 1) DataModule ---------- #


dm = AVSegmentTokenDataModule(
    index_json_path="preprocessed_index.json",
    batch_size=64,
    num_workers=16,
    root_dir="../../core",
)

# ---------- 2) Swin backbone ---------- #

audio_cfg_path = "configs/audio_swin.yaml"
video_cfg_path = "configs/video_swin.yaml"

swin_backbone = DualSwinBackbone(
    audio_cfg_path=audio_cfg_path,
    video_cfg_path=video_cfg_path,
    audio_opts=None,
    video_opts=None,
)

# ---------- 3) Determine d_swin (fused dim) ---------- #

# Grab one batch to probe shapes
dm.setup()
batch = next(iter(dm.train_dataloader()))
with torch.no_grad():
    feats_a, feats_v = swin_backbone(batch["audio_tokens"], batch["video_tokens"])

print("feats_a:", feats_a.shape)
print("feats_v:", feats_v.shape)

# Example: if feats_a: [B, T_a, Da], feats_v: [B, T_v, Dv]
# and _combine_av() in architectures_av.py does mean+concat,
# then d_swin = Da + Dv.
Da = feats_a.shape[-1]
Dv = feats_v.shape[-1]
d_swin = Da + Dv
print("Using d_swin =", d_swin)

# ---------- 4) Build pretraining system ---------- #

d_a = 512
d_b = 512

model = AVPretrainSystem(
    swin_backbone=swin_backbone,
    d_swin=d_swin,
    d_a=d_a,
    d_b=d_b,
    learning_rate=3e-4,
    weight_decay=1e-4,
)

# ---------- 5) Trainer ---------- #

checkpoint_cb = ModelCheckpoint(
    monitor="val_pretrain_loss",
    mode="min",
    save_top_k=3,
    filename="pretrain-{epoch:02d}-{val_pretrain_loss:.4f}",
)
lr_monitor = LearningRateMonitor(logging_interval="epoch")

# #Train on multi-gpu
# trainer = Trainer(
#     accelerator="gpu",
#     devices=8,
#     strategy="ddp_find_unused_parameters_false",
#     max_epochs=3,           # start small for a smoke test
#     precision="bf16-mixed",
#     callbacks=[checkpoint_cb, lr_monitor],
#     log_every_n_steps=50,
# )

#Train on ARM/MAC
trainer = Trainer(
    accelerator="auto",
    devices=1,
    strategy=None,
    max_epochs=3,           # start small for a smoke test
    precision=None,
    callbacks=[checkpoint_cb, lr_monitor],
    log_every_n_steps=50,
)

# ---------- 6) Run pretraining ---------- #

trainer.fit(model, datamodule=dm)
