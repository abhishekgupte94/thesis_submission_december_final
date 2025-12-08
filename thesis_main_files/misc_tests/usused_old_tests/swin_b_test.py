from torchvision.models import swin_b
import torch
def load_video_swin():
    # Initialize model
    model = swin_b()

    # Load pretrained weights
    checkpoint = torch.load('/Users/abhishekgupte_macbookpro/PycharmProjects/thesis_main_files/misc_files/swin_base_patch244_window877_kinetics400_1k.pth')

    # Remove 'backbone.' prefix from state dict keys if present
    state_dict = {k.replace('backbone.', ''): v for k, v in checkpoint['state_dict'].items()}

    # Load weights with strict=False to ignore missing keys
    model.load_state_dict(state_dict, strict=False)

    # Set to evaluation mode
    model.eval()

    return model
model = load_video_swin()
