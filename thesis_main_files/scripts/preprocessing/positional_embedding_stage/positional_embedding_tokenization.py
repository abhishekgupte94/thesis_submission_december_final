# positional_embedding_tokenization.py

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn, Tensor


# ---------------------------------------------------------------------
# [NEW] Configuration for a single modality's tokenisation layer
# ---------------------------------------------------------------------
@dataclass
class ModalityConfig:
    """
    [EXPLAIN] Configuration for one modality (e.g. "audio" or "frame").

    Fields
    ------
    name:
        [EXPLAIN] Identifier for the modality, used only for logging / errors.

    input_dim:
        [EXPLAIN] Feature dimension BEFORE tokenisation:
            - audio:   n_mels (if we treat each mel time frame as one token)
            - frame:   flattened RGB frame dimension = 3 * H * W
            - other:   whatever your preprocessor outputs per timestep.

    embed_dim:
        [EXPLAIN] Shared model dimension (d_model). This is the size of the
        token embeddings that downstream modules (Swin/LFA-ST/VACL) expect.

    max_seq_len:
        [EXPLAIN] Maximum allowed sequence length S for this modality.
        The learnable positional embedding table is allocated up to this length.

    dropout:
        [EXPLAIN] Dropout applied after adding positional embeddings and LayerNorm.
    """

    name: str
    input_dim: int
    embed_dim: int
    max_seq_len: int
    dropout: float = 0.0


# ---------------------------------------------------------------------
# [NEW] Single-modality tokeniser with learnable positional embeddings
# ---------------------------------------------------------------------
class ModalityTokeniser(nn.Module):
    """
    [EXPLAIN] Modality-specific tokenisation layer.

    Concept
    -------
    - This corresponds to the paper's idea:
        "Modality-specific Tokenization layer, where each modality has its
         position embeddings."
    - You feed in a sequence of *preprocessed features* of shape (B, S, D_in).
    - The module:
        1) Projects D_in → embed_dim (shared d_model).
        2) Adds a *learnable* positional embedding per timestep (S ≤ max_seq_len).
        3) Applies LayerNorm + dropout for stability.

    Expected Input
    --------------
    x: Tensor of shape (B, S, D_in)
        - B: batch size (typically 1 in your offline pre-processing)
        - S: sequence length (e.g., # mel frames, or # video frames in a segment)
        - D_in: must equal cfg.input_dim

    Output
    ------
    tokens: Tensor of shape (B, S, embed_dim)
        - This is the shape you pass downstream (e.g., Swin / LFA-ST / VACL).
    """

    def __init__(self, cfg: ModalityConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # [NEW] Linear projection from raw feature dimension → shared embedding dim
        self.proj = nn.Linear(cfg.input_dim, cfg.embed_dim)

        # [NEW] Learnable absolute positional embeddings, specific to this modality.
        # Shape: (1, max_seq_len, embed_dim)
        self.positional_embed = nn.Parameter(
            torch.zeros(1, cfg.max_seq_len, cfg.embed_dim)
        )

        # [NEW] LayerNorm stabilises token scale across time / modalities
        self.norm = nn.LayerNorm(cfg.embed_dim)

        # [NEW] Optional dropout (Identity if dropout=0)
        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0.0 else nn.Identity()

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """
        [EXPLAIN] Initialise learnable parameters in a standard, stable way.
        """
        nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)
        nn.init.normal_(self.positional_embed, mean=0.0, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        """
        [EXPLAIN] Forward pass for one modality.

        Parameters
        ----------
        x : Tensor
            Shape (B, S, D_in), D_in must equal cfg.input_dim.

        Returns
        -------
        Tensor
            Shape (B, S, embed_dim) after projection + positional encodings.
        """
        # [SAFEGUARD] Check dimensionality
        if x.ndim != 3:
            raise ValueError(
                f"[{self.cfg.name}] Expected input of shape (B, S, D_in), "
                f"got {tuple(x.shape)}"
            )

        batch_size, seq_len, feat_dim = x.shape

        # [SAFEGUARD] Feature dim must match config
        if feat_dim != self.cfg.input_dim:
            raise ValueError(
                f"[{self.cfg.name}] Input feature dim {feat_dim} "
                f"≠ cfg.input_dim {self.cfg.input_dim}"
            )

        # [SAFEGUARD] Sequence length cannot exceed allocated positional table
        if seq_len > self.cfg.max_seq_len:
            raise ValueError(
                f"[{self.cfg.name}] Sequence length {seq_len} exceeds "
                f"cfg.max_seq_len {self.cfg.max_seq_len}. "
                f"Increase max_seq_len or truncate your sequence."
            )

        # [EXPLAIN] 1) Linear projection: raw features → shared d_model
        tokens = self.proj(x)                        # (B, S, embed_dim)

        # [EXPLAIN] 2) Add modality-specific positional embeddings
        # We slice positional_embed along time dimension up to S.
        pos_slice = self.positional_embed[:, :seq_len, :]  # (1, S, embed_dim)
        tokens = tokens + pos_slice

        # [EXPLAIN] 3) Normalisation + dropout
        tokens = self.norm(tokens)
        tokens = self.dropout(tokens)
        return tokens


# ---------------------------------------------------------------------
# [NEW] Multi-modal wrapper around multiple ModalityTokeniser instances
# ---------------------------------------------------------------------
class MultiModalTokeniser(nn.Module):
    """
    [EXPLAIN] Thin wrapper for handling multiple modalities at once.

    Usage
    -----
        cfg_audio = ModalityConfig(
            name="audio",
            input_dim=audio_input_dim,  # e.g. n_mels
            embed_dim=d_model,
            max_seq_len=max_audio_tokens,
            dropout=0.1,
        )

        cfg_frame = ModalityConfig(
            name="frame",
            input_dim=frame_input_dim,  # e.g. 3 * H * W
            embed_dim=d_model,
            max_seq_len=max_frame_tokens,
            dropout=0.1,
        )

        tokeniser = MultiModalTokeniser({"audio": cfg_audio, "frame": cfg_frame})

        outputs = tokeniser(
            {
                "audio": audio_features,  # (B, S_a, D_audio_in)
                "frame": frame_features,  # (B, S_v, D_frame_in)
            }
        )

        audio_tokens = outputs["audio"]  # (B, S_a, d_model)
        frame_tokens = outputs["frame"]  # (B, S_v, d_model)
    """

    def __init__(self, modality_cfgs: Dict[str, ModalityConfig]) -> None:
        super().__init__()

        # [NEW] Build a sub-tokeniser per modality
        self.tokenisers = nn.ModuleDict(
            {
                name: ModalityTokeniser(cfg)
                for name, cfg in modality_cfgs.items()
            }
        )

    def forward(self, inputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        [EXPLAIN] Apply the appropriate ModalityTokeniser to each input.

        Parameters
        ----------
        inputs:
            Dict mapping modality name → feature tensor (B, S, D_in).

        Returns
        -------
        Dict[str, Tensor]:
            Same keys, but each value is (B, S, embed_dim).
        """
        outputs: Dict[str, Tensor] = {}

        for name, x in inputs.items():
            if name not in self.tokenisers:
                # [SAFEGUARD] Catch typos / misconfigured modality names
                raise KeyError(
                    f"[MultiModalTokeniser] No tokeniser for modality '{name}'. "
                    f"Available: {list(self.tokenisers.keys())}"
                )
            outputs[name] = self.tokenisers[name](x)

        return outputs


# ---------------------------------------------------------------------
# [NEW] Convenience builder for the FRAME + AUDIO setting
# ---------------------------------------------------------------------
def build_frame_audio_tokeniser(
    audio_input_dim: int,
    frame_input_dim: int,
    d_model: int,
    max_audio_tokens: int,
    max_frame_tokens: int,
    dropout: float = 0.1,
) -> MultiModalTokeniser:
    """
    [EXPLAIN] Construct a MultiModalTokeniser for the specific case:
        - audio stream
        - frame (facial) stream

    Inputs
    ------
    audio_input_dim:
        D_in for audio (e.g., n_mels).

    frame_input_dim:
        D_in for frames (e.g., 3 * H * W after resize).

    d_model:
        Shared embedding dimension.

    max_audio_tokens:
        Maximum Sa (mel time steps per segment).

    max_frame_tokens:
        Maximum Sv (frames per segment).

    dropout:
        Dropout probability for both modalities.

    Returns
    -------
    MultiModalTokeniser
        With "audio" and "frame" entries.
    """

    cfg_audio = ModalityConfig(
        name="audio",
        input_dim=audio_input_dim,
        embed_dim=d_model,
        max_seq_len=max_audio_tokens,
        dropout=dropout,
    )

    cfg_frame = ModalityConfig(
        name="frame",
        input_dim=frame_input_dim,
        embed_dim=d_model,
        max_seq_len=max_frame_tokens,
        dropout=dropout,
    )

    # [EXPLAIN] This object can still be used to access individual ModalityTokenisers
    #           via tokeniser.tokenisers["audio"] / ["frame"] when needed.
    return MultiModalTokeniser({"audio": cfg_audio, "frame": cfg_frame})
