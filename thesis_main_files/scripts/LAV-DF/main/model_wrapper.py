import torch
from core.boundary_localisation.main.build_model import LAVDFInferenceWrapper

# dummy input
x = torch.randn(1, 3, 224, 224)

wrapper = LAVDFInferenceWrapper(
    checkpoint_path="lavdf_pretrained.pth",
    device="cuda"
)

logits = wrapper(x)
probs = wrapper.predict_proba(x)
pred  = wrapper.predict(x)

print("logits:", logits)
print("probs :", probs)
print("pred  :", pred)
