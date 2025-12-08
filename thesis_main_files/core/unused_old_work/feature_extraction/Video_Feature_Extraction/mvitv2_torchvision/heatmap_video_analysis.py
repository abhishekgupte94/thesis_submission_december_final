import numpy as np
import matplotlib.pyplot as plt

# Load the npz
data = np.load("/Users/abhishekgupte_macbookpro/PycharmProjects/new_pitch_estimator_project/mvitv2_pipeline_feature_Extraction/mvitv2_torchvision/output_dir_video/saved_video_features_without_overlap.npz")
arrays = [np.asarray(data[k]) for k in data.files]

# Stack into (N, L, D)
X = np.stack(arrays, axis=0)  # (N, 393, 768)

# Variance across video
var_matrix = np.var(X, axis=0)  # (393, 768)

plt.figure(figsize=(10,6))
plt.imshow(var_matrix, aspect="auto", cmap="viridis")
plt.colorbar(label="Variance across video")
plt.xlabel("Feature dimension (768)")
plt.ylabel("Token index (393)")
plt.title("Per-token, per-feature variance across video")
plt.tight_layout()
plt.show()
