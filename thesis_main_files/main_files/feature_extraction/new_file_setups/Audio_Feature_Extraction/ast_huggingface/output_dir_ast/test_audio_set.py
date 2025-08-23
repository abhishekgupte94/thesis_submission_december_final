import numpy as np
z = np.load("/Users/abhishekgupte_macbookpro/PycharmProjects/new_pitch_estimator_project/AST_MODEL_SETUP/new_updated_hope/output_dir_ast/audio_set.npz", allow_pickle=True)
print("Keys:", z.files)
for k in z.files:
    v = z[k]
    print(k, type(v), getattr(v, "shape", None), getattr(v, "dtype", None))
