import torch
from pytorch_lightning import Trainer

from scripts.dataloaders.dataloader import AVSegmentTokenDataModule
from scripts.feature_extraction.main.main_feature_extraction_wrapper import DualSwinBackbone
from core.training_systems.training_systems.system_pretrain import AVPretrainSystem

def main():
    dm = AVSegmentTokenDataModule(
        index_json_path="preprocessed_index.json",
        batch_size=2,
        num_workers=0,
        root_dir=".",
    )

    audio_cfg_path = "configs/audio_swin.yaml"
    video_cfg_path = "configs/video_swin.yaml"

    swin_backbone = DualSwinBackbone(
        audio_cfg_path=audio_cfg_path,
        video_cfg_path=video_cfg_path,
        audio_opts=None,
        video_opts=None,
    )

    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    with torch.no_grad():
        feats_a, feats_v = swin_backbone(batch["audio_tokens"], batch["video_tokens"])

    Da = feats_a.shape[-1]
    Dv = feats_v.shape[-1]
    d_swin = Da + Dv

    model = AVPretrainSystem(
        swin_backbone=swin_backbone,
        d_swin=d_swin,
        d_a=256,
        d_b=256,
        learning_rate=1e-3,
        weight_decay=1e-4,
    )

    trainer = Trainer(
        accelerator="auto",  # MPS or CPU
        devices=1,
        precision=32,
        max_epochs=1,
        log_every_n_steps=1,
    )

    trainer.fit(model, datamodule=dm)

if __name__ == "__main__":
    main()