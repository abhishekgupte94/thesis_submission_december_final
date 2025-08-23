import numpy as np
z = np.load("/Users/abhishekgupte_macbookpro/PycharmProjects/new_pitch_estimator_project/mvitv2_pipeline_feature_Extraction/mvitv2_torchvision/output_dir_video/saved_video_features_without_overlap.npz", allow_pickle=True)
print("Keys:", z.files)
for k in z.files:
    v = z[k]
    print(k, type(v), getattr(v, "shape", None), getattr(v, "dtype", None))
