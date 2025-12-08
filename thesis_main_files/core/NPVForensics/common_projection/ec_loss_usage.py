import torch
from multimodal_projection_heads import MultiModalProjectionHeads
from ec_loss import evolutionary_consistency_loss

# Example Dimensions
da = 128   # Audio dim
df = 256   # Face dim
d_common_fa = 128 # Common space dimension (S_fa)

# Instantiate only the Face-Audio projector
proj_heads = MultiModalProjectionHeads(
    d_a=da, 
    d_f=df,
    d_fa=d_common_fa,
)

# Dummy inputs (Batch=8, Dim=d)
# In practice, these are your backbone outputs pooled over NPV regions
X_f_seg = torch.randn(8, df)
X_a_seg = torch.randn(8, da)

# Forward pass
Z = proj_heads(
    X_f=X_f_seg,
    X_a=X_a_seg,
)

# Extract aligned features in Common Space S_fa
Z_f_fa = Z["Z_f_fa"]  # (N, d_fa)
Z_a_fa = Z["Z_a_fa"]  # (N, d_fa)

# Compute Evolutionary Consistency Loss
# Note: Paper uses temperature scaling tau (Eq. 4)
L_EC = evolutionary_consistency_loss(Z_f_fa, Z_a_fa, temperature=0.1)

print(f"Face-Audio EC Loss: {L_EC.item()}")