# # =======================
# # alignment_pipeline_gpu.py (refurbished: Change B + video ConvTranspose1d removed, Linear(8->K) added)
# # =======================
#
# """
# GPU-Optimized Temporal Alignment Pipeline for Multi-Modal SSL
# Designed for 8x NVIDIA A100 80GB GPUs with NVMe storage
# Author: SSL Team
# Date: 2024
# """
#
# import os
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.data import DataLoader, Dataset
# from torch.utils.data.distributed import DistributedSampler
# from torch.distributed.optim import ZeroRedundancyOptimizer
# from typing import Dict, List, Tuple, Optional
#
# # Optional flash attention availability flag (left as-is; now unused)
# try:
#     import flash_attn
#     FLASH_AVAILABLE = True
# except Exception:
#     FLASH_AVAILABLE = False
#
#
# # ============================================
# # DATASET (unchanged)
# # NOTE: This mock still emits video as [392,768]; your real pipeline should
# #       supply [B,8,768] to match the model's new expectation.
# # ============================================
# class NVMeOptimizedDataset(Dataset):
#     def __init__(self, nvme_root: str, split: str = "train"):
#         self.nvme_root = nvme_root
#         self.split = split
#         self.items = self._load_index()
#
#     def _load_index(self) -> List[str]:
#         # Placeholder for NVMe index loading
#         return [f"{self.split}_item_{i}" for i in range(1000)]
#
#     def __len__(self) -> int:
#         return len(self.items)
#
#     def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
#         # Mocked features for example purposes
#         # Audio: [101, 768]; Video: [392, 768] (no CLS)  <-- mock only
#         audio = torch.randn(101, 768)
#         video = torch.randn(392, 768)
#         label = torch.randint(0, 2, (1,)).item()
#         return {"audio": audio, "video": video, "label": label}
#
#
# # ============================================
# # TOKEN PROCESSOR
# # ============================================
# class GPUOptimizedTokenProcessor(nn.Module):
#     def __init__(self, rank: int):
#         super().__init__()
#         self.rank = rank
#
#         # Audio path: drop CLS, normalize, downsample to K (example placeholder)
#         self.audio_norm = nn.GroupNorm(8, 768)
#         self.audio_downsample = nn.Conv1d(768, 768, kernel_size=3, stride=2, padding=1)
#
#         # Video path: expects [B,8,768]; normalize (no temporal upsample here)
#         self.video_norm = nn.GroupNorm(8, 768)
#         # removed: self.video_upsample (we align to K with a Linear in the main model)
#
#     @torch.no_grad()
#     def _drop_cls(self, x: torch.Tensor) -> torch.Tensor:
#         # Drop the first token if it's CLS-like; assume AST provides [B,101,768] with CLS at 0
#         if x.shape[1] == 101:
#             return x[:, 1:, :]
#         return x
#
#     def process_ast_tokens_parallel(self, audio_tokens: torch.Tensor) -> torch.Tensor:
#         """
#         audio_tokens: [B, 101, 768] (with CLS) or [B, 100, 768] (no CLS)
#         returns: [B, T', 768] (placeholder flow kept intact)
#         """
#         x = self._drop_cls(audio_tokens)        # [B,100,768] if CLS existed
#         x = x.transpose(1, 2)                   # [B,768,100]
#         x = self.audio_norm(x)                  # GroupNorm over channels
#         x = self.audio_downsample(x)            # Conv1d downsample (placeholder)
#         x = x.transpose(1, 2).contiguous()      # [B,T',768] -> treated as K later in model
#         return x
#
#     def process_mvit_tokens_parallel(self, video_tokens: torch.Tensor) -> torch.Tensor:
#         """
#         video_tokens: [B, 8, 768]  (already temporal tokens; no CLS/spatial grid handling)
#         returns: [B, 8, 768]       (temporal length kept at 8; align-to-K done in main model)
#         """
#         # Enforce expected shape for clarity/debuggability.
#         assert video_tokens.dim() == 3 and video_tokens.shape[2] == 768, \
#             f"Expected [B, 8, 768], got {tuple(video_tokens.shape)}"
#         assert video_tokens.shape[1] == 8, \
#             f"Expected 8 temporal tokens, got {video_tokens.shape[1]}"
#
#         # Input is already [B,8,768]; do not drop CLS or reshape.
#         x = video_tokens                      # [B,8,768]
#         x = x.transpose(1, 2)                 # [B,768,8]
#         x = self.video_norm(x)                # GroupNorm over channels
#         # removed upsample: keep temporal length = 8
#         x = x.transpose(1, 2).contiguous()    # [B,8,768]
#         return x
#
# class SelfSupervisedAVLoss(nn.Module):
#     """
#     Audio-Video contrastive loss with dynamic temperature.
#     - Expects f_audio and f_video as [B, D] (already pooled/aggregated).
#     - No g_transform. Forward returns only the scalar loss.
#     """
#     def __init__(self, initial_temperature: float = 0.1):
#         super().__init__()
#         self.initial_temperature = initial_temperature
#
#     @torch.no_grad()
#     def adjust_temperature(self, f_audio: torch.Tensor, f_video: torch.Tensor) -> torch.Tensor:
#         """
#         Dynamically adjusts temperature based on feature variance (per-batch scalar).
#         """
#         # variances over feature dim
#         a_var = torch.var(f_audio, dim=1).mean()
#         v_var = torch.var(f_video, dim=1).mean()
#         avg_var = (a_var + v_var) / 2
#         # inverse proportional scaling (add epsilon to avoid div-by-zero)
#         dynamic_temperature = self.initial_temperature * (1.0 / (avg_var + 1e-6))
#         return dynamic_temperature
#
#     def forward(self, f_audio: torch.Tensor, f_video: torch.Tensor) -> torch.Tensor:
#         """
#         f_audio: [B, D]
#         f_video: [B, D]
#         returns: scalar loss tensor
#         """
#         # Normalize for cosine similarity
#         f_audio_norm = F.normalize(f_audio, p=2, dim=1)   # [B, D]
#         f_video_norm = F.normalize(f_video, p=2, dim=1)   # [B, D]
#
#         # Dynamic temperature (uses original or normalized—matching your pasted logic)
#         temperature = self.adjust_temperature(f_audio, f_video_norm)
#
#         # Similarity matrix: [B, B]
#         sim = torch.matmul(f_video_norm, f_audio_norm.T)
#
#         B = f_audio.shape[0]
#         mask = torch.eye(B, device=f_audio.device, dtype=torch.bool)
#         pos = torch.diagonal(sim)                         # [B]
#         neg = sim[~mask].view(B, -1)                      # [B, B-1]
#
#         logits = torch.cat([pos.unsqueeze(1), neg], dim=1) / temperature
#         labels = torch.zeros(B, dtype=torch.long, device=f_audio.device)
#         loss = F.cross_entropy(logits, labels)
#         return loss
#
# # ============================================
# # MAIN GPU-OPTIMIZED ALIGNMENT MODEL
# # ============================================
# class GPUOptimizedAlignment(nn.Module):
#     """
#     Full alignment pipeline optimized for multi-GPU training
#     """
#
#     def __init__(
#             self,
#             rank: int,
#             world_size: int,
#             K: int = 50,
#             common_dim: int = 512,
#             use_flash_attention: bool = True  # kept in signature for API stability (unused)
#     ):
#         super().__init__()
#         self.rank = rank
#         self.world_size = world_size
#         self.K = K
#
#         # Token processor
#         self.token_processor = GPUOptimizedTokenProcessor(rank)
#
#         # Projection layers (optimized for GPU)
#         self.audio_proj = nn.Linear(768, common_dim, bias=False)
#         self.audio_bias = nn.Parameter(torch.zeros(common_dim))
#
#         self.video_proj = nn.Linear(768, common_dim, bias=False)
#         self.video_bias = nn.Parameter(torch.zeros(common_dim))
#
#         # Shared TPE - register as buffer for DDP
#         self.register_buffer('shared_tpe', torch.randn(K, common_dim) * 0.02)
#
#         # CPU-style temporal alignment for video (GPU-friendly)
#         # Maps [B, 8, 768] along time dim to [B, K, 768]
#         self.mvit_to_k = nn.Linear(8, self.K, bias=False)
#
#         # NOTE: Cross-modal attention has been removed (Change B).
#         # No attention modules or flags are created here.
#
#         # Gradient checkpointing flag (left intact; not used for attention anymore)
#         self.gradient_checkpointing = False
#
#     def enable_gradient_checkpointing(self):
#         """Enable gradient checkpointing to save memory"""
#         self.gradient_checkpointing = True
#
#     def forward(
#             self,
#             audio_features: torch.Tensor,
#             video_features: torch.Tensor
#     ) -> Dict[str, torch.Tensor]:
#         """
#         Forward pass with GPU optimizations
#
#         Args:
#             audio_features: [B, 101, 768] from AST
#             video_features: [B, 8, 768]   from MViTv2 (temporal tokens already)
#         """
#         B = audio_features.shape[0]
#         B, Ta, Da = audio_features.shape
#         Bv, Tv, Dv = video_features.shape
#         if B != Bv or Da != Dv or Da != 768:
#             raise ValueError(f"Aligner input mismatch: audio={audio_features.shape}, video={video_features.shape}")
#         # if Tv not in (8, 392):
#         #     raise ValueError(f"Video tokens must be 8 or 392, got {Tv}")
#         # if Ta not in (101, 100, self.K):
#         #     print(f"[aligner] Warning: audio tokens {Ta} (expected 101/100/{self.K})")
#
#         # Create CUDA streams for parallel processing
#         audio_stream = torch.cuda.Stream()
#         video_stream = torch.cuda.Stream()
#
#         # Process audio and video in parallel
#         with torch.cuda.stream(audio_stream):
#             audio_k = self.token_processor.process_ast_tokens_parallel(audio_features)    # [B, 50, 768] when K=50
#
#         with torch.cuda.stream(video_stream):
#             video_k = self.token_processor.process_mvit_tokens_parallel(video_features)   # [B, 8, 768]
#
#         # Synchronize streams
#         torch.cuda.current_stream().wait_stream(audio_stream)
#         torch.cuda.current_stream().wait_stream(video_stream)
#
#         # 🔁 Align video temporal length to K using Linear(8→K), CPU-style (GPU-accelerated)
#         # [B, 8, 768] → [B, 768, 8] → Linear(8→K) → [B, 768, K] → [B, K, 768]
#         video_k = self.mvit_to_k(video_k.transpose(1, 2)).transpose(1, 2).contiguous()   # [B, K, 768]
#
#         # Project to common space (uses Tensor Cores on A100)
#         with torch.cuda.amp.autocast(dtype=torch.bfloat16):
#             # Batched matrix multiplication
#             audio_projected = F.linear(audio_k, self.audio_proj.weight) + self.audio_bias   # [B, 50, 512] when K=50
#             video_projected = F.linear(video_k, self.video_proj.weight) + self.video_bias   # [B, K, 512]
#
#             # Add shared TPE (now both time lengths are K when K=50 for audio)
#             audio_with_tpe = audio_projected + self.shared_tpe   # [B, K, 512]
#             video_with_tpe = video_projected + self.shared_tpe   # [B, K, 512]
#
#             # Cross-modal attention removed: pass-through CPU-style (Change B)
#             audio_final = audio_with_tpe
#             video_final = video_with_tpe
#
#         return {
#             'audio_aligned': audio_final,  # [B, K, common_dim]
#             'video_aligned': video_final,  # [B, K, common_dim]
#             'audio_k': audio_k,            # [B, T'_a, 768] (pre-projection)
#             'video_k': video_k             # [B, K, 768]    (pre-projection, after Linear(8->K))
#         }
#
#
# # ============================================
# # DDP / TRAIN LOOP (unchanged)
# # ============================================
# def setup_distributed(rank: int, world_size: int, master_port: str = "12355"):
#     """Initialize distributed training"""
#     os.environ["MASTER_ADDR"] = "localhost"
#     os.environ["MASTER_PORT"] = master_port
#     dist.init_process_group("nccl", rank=rank, world_size=world_size)
#     torch.cuda.set_device(rank)
#
#     # Enable TF32 for A100 Tensor Cores
#     torch.backends.cuda.matmul.allow_tf32 = True
#     torch.backends.cudnn.allow_tf32 = True
#     torch.backends.cudnn.benchmark = True
#
#     print(f"GPU {rank} initialized successfully")
#
#
# def cleanup_distributed():
#     """Cleanup distributed training"""
#     dist.destroy_process_group()
#
#
# # ============================================
# # OPTIM / TRAINING STEP EXAMPLE (unchanged)
# # ============================================
# def train_one_epoch_distributed(
#         rank: int,
#         world_size: int,
#         model: GPUOptimizedAlignment,
#         loader: DataLoader
# ):
#     model.train()
#     scaler = torch.cuda.amp.GradScaler(enabled=True)
#
#     # ZeroRedundancyOptimizer wraps AdamW (example)
#     optimizer = ZeroRedundancyOptimizer(
#         model.parameters(),
#         optimizer_class=torch.optim.AdamW,
#         lr=3e-4,
#         weight_decay=0.01
#     )
#
#     for batch_idx, batch in enumerate(loader):
#         audio = batch["audio"].to(f"cuda:{rank}", non_blocking=True).unsqueeze(0)
#         video = batch["video"].to(f"cuda:{rank}", non_blocking=True).unsqueeze(0)
#
#         optimizer.zero_grad(set_to_none=True)
#
#         with torch.cuda.amp.autocast(dtype=torch.bfloat16):
#             # Mocked feature extractors (placeholders)
#             audio_features = torch.randn(audio.shape[0], 101, 768, device=f'cuda:{rank}')
#             # NOTE: For real runs, supply video_features as [B,8,768]; this mock uses [B,392,768]
#             video_features = torch.randn(video.shape[0], 392, 768, device=f'cuda:{rank}')
#
#             # Alignment
#             output = model(audio_features, video_features)
#
#             # Contrastive loss (example: global mean pooling then InfoNCE)
#             audio_global = output['audio_aligned'].mean(dim=1)  # [B, 512]
#             video_global = output['video_aligned'].mean(dim=1)  # [B, 512]
#             similarity = torch.matmul(audio_global, video_global.T) / 0.07
#             labels = torch.arange(audio.shape[0], device=f'cuda:{rank}')
#             loss = F.cross_entropy(similarity, labels)
#
#         # Backward pass with AMP
#         scaler.scale(loss).backward()
#
#         # Gradient clipping + optimizer step
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#         scaler.step(optimizer)
#         scaler.update()
#
#         if batch_idx % 50 == 0 and rank == 0:
#             print(f"[Rank {rank}] Batch {batch_idx}: loss = {loss.item():.4f}")
#
#
# # ============================================
# # MAIN (unchanged)
# # ============================================
# def main_worker(rank: int, world_size: int):
#     setup_distributed(rank, world_size)
#
#     # Dataset / DataLoader
#     dataset = NVMeOptimizedDataset("/nvme/dataset", split="train")
#     sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
#     loader = DataLoader(
#         dataset,
#         batch_size=1,
#         sampler=sampler,
#         num_workers=4,
#         pin_memory=True,
#         persistent_workers=True
#     )
#
#     # Model
#     model = GPUOptimizedAlignment(rank=rank, world_size=world_size, K=50, common_dim=512)
#     model = model.to(f"cuda:{rank}")
#     model = DDP(model, device_ids=[rank], output_device=rank, broadcast_buffers=False, find_unused_parameters=False)
#
#     # Train one epoch (example)
#     train_one_epoch_distributed(rank, world_size, model, loader)
#
#     cleanup_distributed()
#
#
#
# =======================
# alignment_pipeline_gpu.py (refurbished: Change B + video ConvTranspose1d removed, Linear(8->K) added)
# =======================

"""
GPU-Optimized Temporal Alignment Pipeline for Multi-Modal SSL
Designed for 8x NVIDIA A100 80GB GPUs with NVMe storage
Author: SSL Team
Date: 2024
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.optim import ZeroRedundancyOptimizer
from typing import Dict, List, Tuple, Optional

# Optional flash attention availability flag (left as-is; now unused)
try:
    import flash_attn
    FLASH_AVAILABLE = True
except Exception:
    FLASH_AVAILABLE = False


# ============================================
# DATASET (unchanged)
# NOTE: This mock still emits video as [392,768]; your real pipeline should
#       supply [B,8,768] to match the model's new expectation.
# ============================================
class NVMeOptimizedDataset(Dataset):
    def __init__(self, nvme_root: str, split: str = "train"):
        self.nvme_root = nvme_root
        self.split = split
        self.items = self._load_index()

    def _load_index(self) -> List[str]:
        # Placeholder for NVMe index loading
        return [f"{self.split}_item_{i}" for i in range(1000)]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Mocked features for example purposes
        # Audio: [101, 768]; Video: [392, 768] (no CLS)  <-- mock only
        audio = torch.randn(101, 768)
        video = torch.randn(392, 768)
        label = torch.randint(0, 2, (1,)).item()
        return {"audio": audio, "video": video, "label": label}


# ============================================
# TOKEN PROCESSOR
# ============================================
class GPUOptimizedTokenProcessor(nn.Module):
    def __init__(self, rank: int):
        super().__init__()
        self.rank = rank

        # Audio path: drop CLS, normalize, downsample to K (example placeholder)
        self.audio_norm = nn.GroupNorm(8, 768)
        self.audio_downsample = nn.Conv1d(768, 768, kernel_size=3, stride=2, padding=1)

        # Video path: expects [B,8,768]; normalize (no temporal upsample here)
        self.video_norm = nn.GroupNorm(8, 768)
        # removed: self.video_upsample (we align to K with a Linear in the main model)

    @torch.no_grad()
    def _drop_cls(self, x: torch.Tensor) -> torch.Tensor:
        # Drop the first token if it's CLS-like; assume AST provides [B,101,768] with CLS at 0
        if x.shape[1] == 101:
            return x[:, 1:, :]
        return x

    def process_ast_tokens_parallel(self, audio_tokens: torch.Tensor) -> torch.Tensor:
        """
        audio_tokens: [B, 101, 768] (with CLS) or [B, 100, 768] (no CLS)
        returns: [B, T', 768] (placeholder flow kept intact)
        """
        x = self._drop_cls(audio_tokens)        # [B,100,768] if CLS existed
        x = x.transpose(1, 2)                   # [B,768,100]
        x = self.audio_norm(x)                  # GroupNorm over channels
        x = self.audio_downsample(x)            # Conv1d downsample (placeholder)
        x = x.transpose(1, 2).contiguous()      # [B,T',768] -> treated as K later in model
        return x

    def process_mvit_tokens_parallel(self, video_tokens: torch.Tensor) -> torch.Tensor:
        """
        video_tokens: [B, 8, 768]  (already temporal tokens; no CLS/spatial grid handling)
        returns: [B, 8, 768]       (temporal length kept at 8; align-to-K done in main model)
        """
        # Enforce expected shape for clarity/debuggability.
        assert video_tokens.dim() == 3 and video_tokens.shape[2] == 768, \
            f"Expected [B, 8, 768], got {tuple(video_tokens.shape)}"
        assert video_tokens.shape[1] == 8, \
            f"Expected 8 temporal tokens, got {video_tokens.shape[1]}"

        # Input is already [B,8,768]; do not drop CLS or reshape.
        x = video_tokens                      # [B,8,768]
        x = x.transpose(1, 2)                 # [B,768,8]
        x = self.video_norm(x)                # GroupNorm over channels
        # removed upsample: keep temporal length = 8
        x = x.transpose(1, 2).contiguous()    # [B,8,768]
        return x

class SelfSupervisedAVLoss(nn.Module):
    """
    Audio-Video contrastive loss with dynamic temperature.
    - Expects f_audio and f_video as [B, D] (already pooled/aggregated).
    - No g_transform. Forward returns only the scalar loss.
    """
    def __init__(self, initial_temperature: float = 0.1):
        super().__init__()
        self.initial_temperature = initial_temperature

    @torch.no_grad()
    def adjust_temperature(self, f_audio: torch.Tensor, f_video: torch.Tensor) -> torch.Tensor:
        """
        Dynamically adjusts temperature based on feature variance (per-batch scalar).
        """
        # variances over feature dim
        a_var = torch.var(f_audio, dim=1).mean()
        v_var = torch.var(f_video, dim=1).mean()
        avg_var = (a_var + v_var) / 2
        # inverse proportional scaling (add epsilon to avoid div-by-zero)
        dynamic_temperature = self.initial_temperature * (1.0 / (avg_var + 1e-6))
        return dynamic_temperature

    def forward(self, f_audio: torch.Tensor, f_video: torch.Tensor) -> torch.Tensor:
        """
        f_audio: [B, D]
        f_video: [B, D]
        returns: scalar loss tensor
        """
        # Normalize for cosine similarity
        f_audio_norm = F.normalize(f_audio, p=2, dim=1)   # [B, D]
        f_video_norm = F.normalize(f_video, p=2, dim=1)   # [B, D]

        # Dynamic temperature (uses original or normalizedâ€”matching your pasted logic)
        temperature = self.adjust_temperature(f_audio, f_video_norm)

        # Similarity matrix: [B, B]
        sim = torch.matmul(f_video_norm, f_audio_norm.T)

        B = f_audio.shape[0]
        mask = torch.eye(B, device=f_audio.device, dtype=torch.bool)
        pos = torch.diagonal(sim)                         # [B]
        neg = sim[~mask].view(B, -1)                      # [B, B-1]

        logits = torch.cat([pos.unsqueeze(1), neg], dim=1) / temperature
        labels = torch.zeros(B, dtype=torch.long, device=f_audio.device)
        loss = F.cross_entropy(logits, labels)
        return loss

# ============================================
# MAIN GPU-OPTIMIZED ALIGNMENT MODEL
# ============================================
class GPUOptimizedAlignment(nn.Module):
    """
    Full alignment pipeline optimized for multi-GPU training
    """

    def __init__(
            self,
            rank: int,
            world_size: int,
            K: int = 50,
            common_dim: int = 512,
            use_flash_attention: bool = True  # kept in signature for API stability (unused)
    ):
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        self.K = K

        # Token processor
        self.token_processor = GPUOptimizedTokenProcessor(rank)

        # Projection layers (optimized for GPU)
        self.audio_proj = nn.Linear(768, common_dim, bias=False)
        self.audio_bias = nn.Parameter(torch.zeros(common_dim))

        self.video_proj = nn.Linear(768, common_dim, bias=False)
        self.video_bias = nn.Parameter(torch.zeros(common_dim))

        # Shared TPE - register as buffer for DDP
        self.register_buffer('shared_tpe', torch.randn(K, common_dim) * 0.02)

        # CPU-style temporal alignment for video (GPU-friendly)
        # Maps [B, 8, 768] along time dim to [B, K, 768]
        self.mvit_to_k = nn.Linear(8, self.K, bias=False)

        # NOTE: Cross-modal attention has been removed (Change B).
        # No attention modules or flags are created here.

        # Gradient checkpointing flag (left intact; not used for attention anymore)
        self.gradient_checkpointing = False

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing to save memory"""
        self.gradient_checkpointing = True

    def forward(
            self,
            audio_features: torch.Tensor,
            video_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with GPU optimizations

        Args:
            audio_features: [B, 101, 768] from AST
            video_features: [B, 8, 768]   from MViTv2 (temporal tokens already)
        """
        B = audio_features.shape[0]
        B, Ta, Da = audio_features.shape
        Bv, Tv, Dv = video_features.shape
        if B != Bv or Da != Dv or Da != 768:
            raise ValueError(f"Aligner input mismatch: audio={audio_features.shape}, video={video_features.shape}")
        # if Tv not in (8, 392):
        #     raise ValueError(f"Video tokens must be 8 or 392, got {Tv}")
        # if Ta not in (101, 100, self.K):
        #     print(f"[aligner] Warning: audio tokens {Ta} (expected 101/100/{self.K})")

        # Create CUDA streams for parallel processing
        audio_stream = torch.cuda.Stream()
        video_stream = torch.cuda.Stream()

        # Process audio and video in parallel
        with torch.cuda.stream(audio_stream):
            audio_k = self.token_processor.process_ast_tokens_parallel(audio_features)    # [B, 50, 768] when K=50

        with torch.cuda.stream(video_stream):
            video_k = self.token_processor.process_mvit_tokens_parallel(video_features)   # [B, 8, 768]

        # Synchronize streams
        torch.cuda.current_stream().wait_stream(audio_stream)
        torch.cuda.current_stream().wait_stream(video_stream)

        # ðŸ” Align video temporal length to K using Linear(8â†’K), CPU-style (GPU-accelerated)
        # [B, 8, 768] â†’ [B, 768, 8] â†’ Linear(8â†’K) â†’ [B, 768, K] â†’ [B, K, 768]
        video_k = self.mvit_to_k(video_k.transpose(1, 2)).transpose(1, 2).contiguous()   # [B, K, 768]

        # Project to common space (uses Tensor Cores on A100)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            # Batched matrix multiplication
            audio_projected = F.linear(audio_k, self.audio_proj.weight) + self.audio_bias   # [B, 50, 512] when K=50
            video_projected = F.linear(video_k, self.video_proj.weight) + self.video_bias   # [B, K, 512]

            # Add shared TPE (now both time lengths are K when K=50 for audio)
            audio_with_tpe = audio_projected + self.shared_tpe   # [B, K, 512]
            video_with_tpe = video_projected + self.shared_tpe   # [B, K, 512]

            # Cross-modal attention removed: pass-through CPU-style (Change B)
            audio_final = audio_with_tpe
            video_final = video_with_tpe

        return {
            'audio_aligned': audio_final,  # [B, K, common_dim]
            'video_aligned': video_final,  # [B, K, common_dim]
            'audio_k': audio_k,            # [B, T'_a, 768] (pre-projection)
            'video_k': video_k             # [B, K, 768]    (pre-projection, after Linear(8->K))
        }


# ============================================
# DDP / TRAIN LOOP (unchanged)
# ============================================
def setup_distributed(rank: int, world_size: int, master_port: str = "12355"):
    """Initialize distributed training"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = master_port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # Enable TF32 for A100 Tensor Cores
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    print(f"GPU {rank} initialized successfully")


def cleanup_distributed():
    """Cleanup distributed training"""
    dist.destroy_process_group()


# ============================================
# OPTIM / TRAINING STEP EXAMPLE (unchanged)
# ============================================
def train_one_epoch_distributed(
        rank: int,
        world_size: int,
        model: GPUOptimizedAlignment,
        loader: DataLoader
):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # ZeroRedundancyOptimizer wraps AdamW (example)
    optimizer = ZeroRedundancyOptimizer(
        model.parameters(),
        optimizer_class=torch.optim.AdamW,
        lr=3e-4,
        weight_decay=0.01
    )

    for batch_idx, batch in enumerate(loader):
        audio = batch["audio"].to(f"cuda:{rank}", non_blocking=True).unsqueeze(0)
        video = batch["video"].to(f"cuda:{rank}", non_blocking=True).unsqueeze(0)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            # Mocked feature extractors (placeholders)
            audio_features = torch.randn(audio.shape[0], 101, 768, device=f'cuda:{rank}')
            # NOTE: For real runs, supply video_features as [B,8,768]; this mock uses [B,392,768]
            video_features = torch.randn(video.shape[0], 392, 768, device=f'cuda:{rank}')

            # Alignment
            output = model(audio_features, video_features)

            # Contrastive loss (example: global mean pooling then InfoNCE)
            audio_global = output['audio_aligned'].mean(dim=1)  # [B, 512]
            video_global = output['video_aligned'].mean(dim=1)  # [B, 512]
            similarity = torch.matmul(audio_global, video_global.T) / 0.07
            labels = torch.arange(audio.shape[0], device=f'cuda:{rank}')
            loss = F.cross_entropy(similarity, labels)

        # Backward pass with AMP
        scaler.scale(loss).backward()

        # Gradient clipping + optimizer step
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        if batch_idx % 50 == 0 and rank == 0:
            print(f"[Rank {rank}] Batch {batch_idx}: loss = {loss.item():.4f}")


# ============================================
# MAIN (unchanged)
# ============================================
def main_worker(rank: int, world_size: int):
    setup_distributed(rank, world_size)

    # Dataset / DataLoader
    dataset = NVMeOptimizedDataset("/nvme/dataset", split="train")
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    # Model
    model = GPUOptimizedAlignment(rank=rank, world_size=world_size, K=50, common_dim=512)
    model = model.to(f"cuda:{rank}")
    model = DDP(model, device_ids=[rank], output_device=rank, broadcast_buffers=False, find_unused_parameters=False)

    # Train one epoch (example)
    train_one_epoch_distributed(rank, world_size, model, loader)

    cleanup_distributed()


