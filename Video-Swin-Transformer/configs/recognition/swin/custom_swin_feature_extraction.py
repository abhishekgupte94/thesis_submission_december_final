#
# # configs/custom_swin_feature_extraction.py
# import os
# from pathlib import Path
# # Use relative paths instead of absolute paths
# swin_checkpoint = "checkpoints/swin_base_patch244_window877_kinetics400_22k.pth"
# # project_dir = Path(__file__).resolve().parents[4]  # Adjust as per your script's location
#
# # # Example usage
# # ann_file_train_real = str(project_dir / "Video-Swin-Transformer" / "data" / "train" / "real" / "lip_train_text_real.txt")
# # data_root_real = str(project_dir / "Video-Swin-Transformer" / "data" / "train" / "real" )
# #
# # ann_file_train_fake = str(project_dir / "Video-Swin-Transformer" / "data" / "train" / "fake" / "lip_train_text_fake.txt")
# # data_root_fake = str(project_dir / "Video-Swin-Transformer" / "data" / "train" / "fake" )
#
# # ann_file_train = "data/train/lip_train_text.txt"
# # data_root = "data/train"
# ann_file_train_real = "data/train/real/lip_train_text_real.txt"
# data_root_real = "data/train/real"
# ann_file_train_fake = "data/train/fake/lip_train_text_fake.txt"
# data_root_fake = "data/train/fake"
#
# # Debugging: Print resolved paths
# # print("Checkpoint Path:", swin_checkpoint)
# # print("Training List Path:", ann_file_train)
# # print("Data Root Path:", data_root)
#
# # Load base Swin-B config
# _base_ = "swin_base_patch244_window877_kinetics400_22k.py"
#
# # Update dataset locations
# data = dict(
#     test=dict(
#         ann_file=ann_file_train_real,
#         data_prefix=data_root_real))
#
# # Modify Swin-B model for feature extraction (Remove classification head)
# model = dict(
#     backbone=dict(
#         type='SwinTransformer3D',
#         pretrained=swin_checkpoint,  # Load pre-trained model
#         patch_size=(2, 4, 4),
#         window_size=(8, 7, 7),
#         embed_dim=128,
#         depths=[2, 2, 18, 2],
#         num_heads=[4, 8, 16, 32],
#         mlp_ratio=4.,
#         qkv_bias=True,
#         drop_path_rate=0.2,
#         patch_norm=True),
#
#     # Change cls_head to an existing MMAction2-compatible head
#     cls_head=None
# )
#
# # # Data pipeline (Ensure input is 224x224)
# # test_pipeline = [
# #     dict(type='DecordInit'),  # Video decoding
# #     dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1),  # Sample every 2 frames (~12.5 FPS)
# #     dict(type='DecordDecode'),
# #     dict(type='Resize', scale=(224, 224)),  # Resize to 224x224
# #     dict(type='CenterCrop', crop_size=224),
# #     dict(type='ToTensor'),
# #     dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
# #     dict(type='Collect', keys=['imgs'], meta_keys=[]),
# #     dict(type='FormatShape', input_format='NCTHW')
# # ]
#
#
#
#
# test_pipeline = [
#     dict(type='DecordInit'),  # Initialize video decoding
#     dict(type='SampleFrames', clip_len=32, frame_interval=1, num_clips=4),  # Extract valid frames
#     dict(type='DecordDecode'),  # Decode frames
# #     dict(type='CenterCrop', crop_size=224),
#     dict(type='ToTensor'),  # Convert images to PyTorch tensors
#     dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),  # Normalize after tensor conversion
#     dict(type='Collect', keys=['imgs'], meta_keys=[]),
#     dict(type='FormatShape', input_format='NCTHW')  # Format into correct shape for processing
# ]
# # print("🛠️ DEBUG: Using test pipeline configuration:", test_pipeline)
#
#
#


# configs/custom_swin_feature_extraction.py




# _base_ = [
#     '../../_base_/models/swin/swin_base.py', '../../_base_/default_runtime.py'
# ]
# Dataset config
# data = dict(
#     test=dict(
#         ann_file=ann_file_train_real,
#         data_prefix=data_root_real
#     )
# )
_base_ = "swin_base_patch244_window877_kinetics600_22k.py"

swin_checkpoint = "checkpoints/swin_base_patch244_window877_kinetics600_22k.pth"

ann_file_train_real = "data/train/real/lip_train_text_real.txt"
data_root_real = "data/train/real"
ann_file_train_fake = "data/train/fake/lip_train_text_fake.txt"
data_root_fake = "data/train/fake"

# ✅ Use backbone only — no classification head
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='SwinTransformer3D',
        # pretrained=swin_checkpoint,
        patch_size=(2, 4, 4),
        window_size=(8, 7, 7),
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        mlp_ratio=4.,
        qkv_bias=True,
        drop_path_rate=0.2,
        patch_norm=True
        # init_cfg=dict(
        #     type='Pretrained',
        #     checkpoint='checkpoints/swin_base_patch244_window877_kinetics400_22k.pth',
        #     prefix='backbone'
        # )
    ),
    cls_head=None,  # ✅ Fixed comma placement
    test_cfg=dict(
        # max_testing_views=4,
        feature_extraction=True
    )
)



data = dict(

    test=dict(
        ann_file=ann_file_train_real,
        data_prefix=data_root_real,
    ))

# ✅ Feature extraction test pipeline
test_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=32, frame_interval=1, num_clips=4),
    dict(type='DecordDecode'),
    dict(type='ToTensor'),
    dict(type='Normalize', mean=[123.675, 116.28, 103.53],
         std=[58.395, 57.12, 57.375]),
    dict(type='Collect', keys=['imgs'], meta_keys=[]),
    dict(type='FormatShape', input_format='NCTHW')
]




