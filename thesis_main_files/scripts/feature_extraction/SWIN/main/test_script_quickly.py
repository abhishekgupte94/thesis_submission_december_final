
import torch
def mel_b1_64x96_to_swin_input(
    mel: torch.Tensor,
    *,
    out_hw=(224, 224),
    to_3ch=True,
    normalize="per_sample",   # "none" | "per_sample"
    eps=1e-6,
    pad_value="min",          # "min" | float
    time_align="left",        # "left" | "center"
) -> torch.Tensor:
    """
    Converts (B,1,64,96) log-mel into Swin-friendly input:
      -> (B,3,224,224) if to_3ch else (B,1,224,224)

    Steps:
      1) ensure (B,1,H,W) and fix (96,64) orientation if needed
      2) per-sample normalization (recommended)
      3) pad/crop into (224,224) without resizing
      4) replicate channel to 3 if desired
    """
    # ---- shape to (B,1,H,W) ----
    if mel.ndim == 2:
        mel = mel.unsqueeze(0).unsqueeze(0)
    elif mel.ndim == 3:
        if mel.shape[0] == 1:
            mel = mel.unsqueeze(0)      # (1,H,W)->(1,1,H,W)
        else:
            mel = mel.unsqueeze(1)      # (B,H,W)->(B,1,H,W)
    elif mel.ndim != 4:
        raise ValueError(f"Unexpected mel shape: {tuple(mel.shape)}")

    B, C, H, W = mel.shape
    if C != 1:
        raise ValueError(f"Expected C=1 mel, got {C} for shape {tuple(mel.shape)}")

    # ---- fix common axis swap (96,64) -> (64,96) ----
    if (H, W) == (96, 64):
        mel = mel.transpose(2, 3).contiguous()
        H, W = 64, 96

    # ---- normalize ----
    if normalize == "per_sample":
        mean = mel.mean(dim=(2, 3), keepdim=True)
        std = mel.std(dim=(2, 3), keepdim=True).clamp_min(eps)
        mel = (mel - mean) / std
    elif normalize != "none":
        raise ValueError("normalize must be 'none' or 'per_sample'")

    # ---- pad/crop into (out_hw) without resizing ----
    OH, OW = out_hw
    mel = mel[:, :, :min(H, OH), :min(W, OW)]
    _, _, Hc, Wc = mel.shape

    if pad_value == "min":
        fill = mel.amin(dim=(2, 3), keepdim=True)  # (B,1,1,1)
    elif isinstance(pad_value, (int, float)):
        fill = torch.tensor(float(pad_value), device=mel.device, dtype=mel.dtype).view(1, 1, 1, 1)
    else:
        raise ValueError("pad_value must be 'min' or a float")

    canvas = fill.expand(B, 1, OH, OW).clone()

    if time_align == "left":
        top = (OH - Hc) // 2
        left = 0
    elif time_align == "center":
        top = (OH - Hc) // 2
        left = (OW - Wc) // 2
    else:
        raise ValueError("time_align must be 'left' or 'center'")

    canvas[:, :, top:top + Hc, left:left + Wc] = mel
    mel = canvas

    if to_3ch:
        mel = mel.repeat(1, 3, 1, 1)  # (B,3,OH,OW)

    return mel



if __name__ =="__main__":
    from scripts.preprocessing.audio.AudioPreprocessorNPV import AudioPreprocessorNPV
    from pathlib import Path
    audio_path = Path("/Users/abhishekgupte_macbookpro/PycharmProjects/project_combined_repo_clean/thesis_main_files/data/processed/video_files/AVSpeech/audio/1.wav")
    preprocessor = AudioPreprocessorNPV()
    mel = preprocessor.process_audio_file(audio_path)

    if not isinstance(mel, torch.Tensor) or mel.ndim != 2:
        raise RuntimeError(
            f"Expected mel Tensor of shape (H, W), got {type(mel)} {getattr(mel, 'shape', None)}"
        )

    x = mel.unsqueeze(0).unsqueeze(0)  # (1, 1, 96, 64) (no resize)
    print(f"{x.shape} is the shape of x prior")
    x = mel_b1_64x96_to_swin_input(
        x,
        out_hw=(224, 224),
        to_3ch=True,  # keep vanilla Swin patch embed unchanged (expects 3)
        normalize="per_sample",
        pad_value="min",
        time_align="left",
    )
    print(f"{x.shape} is the shape of x prior")