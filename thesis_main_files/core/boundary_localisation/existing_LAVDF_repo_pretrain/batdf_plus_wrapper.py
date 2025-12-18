# file: batfdplus_prb_wrapper.py

import torch
from torch import nn

# ⬇️ Adapt this import to the actual path in LAV_DF
# e.g. from model.batfd_plus import BatfdPlus
from model.batfd_plus import BatfdPlus  # <-- adjust if needed


class BatfdPlusPRBExtractor(nn.Module):
    """
    Wrapper around the official BA-TFD+ model that:
      - loads a pretrained checkpoint
      - runs a standard forward pass
      - grabs the PRB outputs via forward hooks

    You can treat this as a "teacher" that gives you PRB outputs to
    compare / fuse with your own Zsv/Zsa pipeline.
    """

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        super().__init__()
        self.device = torch.device(device)
        self.checkpoint_path = checkpoint_path

        # Load LightningModule from checkpoint (strict=False just in case)
        self.model: BatfdPlus = BatfdPlus.load_from_checkpoint(
            checkpoint_path,
            strict=False,
        )
        self.model.to(self.device)
        self.model.eval()

        # Storage for hook outputs
        self._features = {}
        self._handles = []

        # --- Register hooks on PRB / boundary module ---
        #
        # In the official repo, the PRB logic is implemented inside the
        # "boundary module" that takes encoder features + frame scores and
        # returns D×T boundary maps for each modality.
        #
        # You might have something like:
        #   self.boundary_module = BoundaryModulePlus(...)
        #
        # Adapt 'boundary_module' to the actual attribute name if different.
        if not hasattr(self.model, "boundary_module"):
            raise AttributeError(
                "Expected self.model.boundary_module to exist. "
                "Please adapt BatfdPlusPRBExtractor to the actual attribute "
                "name used for the PRB / boundary module in the LAV_DF repo."
            )

        def make_hook(name):
            def _hook(module, inputs, output):
                # Detach to avoid autograd graph growth
                self._features[name] = output.detach()
            return _hook

        # Register a single hook on the boundary module output
        # If the module returns a tuple (video_prb, audio_prb, ...),
        # you will see that structure in self._features["prb_out"].
        handle = self.model.boundary_module.register_forward_hook(
            make_hook("prb_out")
        )
        self._handles.append(handle)

        # Optionally print some info about temporal sampling T
        self.print_temporal_setup()

    def print_temporal_setup(self):
        """
        Print to terminal what temporal length T the model expects,
        and how LAV_DF does sampling, according to the paper/config.
        """

        print("\n=== [BA-TFD+] Temporal Setup (LAV_DF) ===")
        # From paper + repo defaults: video padded/cropped to 512 frames,
        # audio mel-spec 64×2048 reshaped with τ = 4 -> 512 steps. :contentReference[oaicite:1]{index=1}
        print("* Default temporal length T          : 512")
        print("* Video: clips are padded/cropped to : 512 frames at 96×96")
        print("* Audio: mel-spectrogram 64×2048; τ  : 2048 / 512 = 4")
        print("          -> reshaped so audio also has T = 512 time steps")
        # If the LightningModule exposes hparams, show anything relevant
        if hasattr(self.model, "hparams"):
            print("\n[Debug] Model hparams (truncated):")
            print(str(self.model.hparams)[:400])
        print("====================================\n")

    def forward_until_prb(self, batch):
        """
        Run a full forward pass of the BA-TFD+ model, but only use the
        features captured at the PRB / boundary module.

        Parameters
        ----------
        batch : whatever the original DataModule yields
            Usually a dict or tuple like:
              {
                'video':  (B, C, T, H, W),
                'audio':  (B, 1, n_mels, T),
                'target': ...
              }

        Returns
        -------
        prb_out : torch.Tensor or tuple of tensors
            The raw output of self.model.boundary_module(...) as captured
            by the forward hook. Typically contains per-modality boundary
            maps with shape (B, D, T, ...) or similar.
        """
        self._features.clear()

        with torch.no_grad():
            _ = self.model(batch)

        if "prb_out" not in self._features:
            raise RuntimeError(
                "PRB hook did not fire. Check that boundary_module "
                "is actually called inside BatfdPlus.forward()."
            )

        return self._features["prb_out"]

    def remove_hooks(self):
        """
        Clean up forward hooks (e.g., if you are done with extraction).
        """
        for h in self._handles:
            h.remove()
        self._handles.clear()


#### Example Use Case

# file: batdf_plus_wrapper.py
#
# import torch
# from torch.utils.data import DataLoader
#
# from batfdplus_prb_wrapper import BatfdPlusPRBExtractor
#
# # ⬇️ Put your real dataset / dataloader here
# def get_lavdf_dataloader():
#     # Example placeholder – adapt to the repo's dataset code:
#     #   from dataset.lavdf_dataset import LAVDFDataset
#     #   ds = LAVDFDataset(root=..., split="test", ...)
#     #   return DataLoader(ds, batch_size=1, ...)
#     raise NotImplementedError("Hook in the actual LAV_DF dataloader here.")
#
#
# def main():
#     ckpt_path = "ckpt/batfd_plus.ckpt"  # or pretrained path you downloaded
#
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     extractor = BatfdPlusPRBExtractor(ckpt_path, device=device)
#
#     loader = get_lavdf_dataloader()
#
#     for batch in loader:
#         # Make sure batch is moved to the same device as the model
#         batch = {
#             k: v.to(device) if isinstance(v, torch.Tensor) else v
#             for k, v in batch.items()
#         }
#
#         prb_out = extractor.forward_until_prb(batch)
#
#         # Print shapes/info to terminal the first time
#         print("PRB output type:", type(prb_out))
#         if isinstance(prb_out, (tuple, list)):
#             for i, x in enumerate(prb_out):
#                 print(f"  component {i}: shape = {tuple(x.shape)}")
#         else:
#             print("  shape:", tuple(prb_out.shape))
#
#         # For your use-case: break after one batch or keep going
#         break
#
#
# if __name__ == "__main__":
#     main()
