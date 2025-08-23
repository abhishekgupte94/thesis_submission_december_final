"""
CPU-Optimized Temporal Alignment Pipeline for Multi-Modal SSL
Assumes pre-extracted features: AST [B, 101, 768] and MViTv2 [B, 8, 768]
Single-machine CPU implementation with memory efficiency
Author: SSL Team
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import multiprocessing as mp
import warnings
warnings.filterwarnings('ignore')

# Set CPU-specific optimizations
torch.set_num_threads(mp.cpu_count())
torch.set_num_interop_threads(2)

# ============================================
# CONFIGURATION
# ============================================

@dataclass
class AlignmentConfig:
    """Configuration for alignment pipeline"""
    ast_temporal_tokens: int = 101  # AST temporal tokens
    mvit_temporal_tokens: int = 8   # MViTv2 temporal positions
    k_steps: int = 50               # Common temporal resolution
    feature_dim: int = 768          # Input feature dimension
    common_dim: int = 512           # Output dimension after projection
    use_cross_attention: bool = False  # Optional cross-attention

# ============================================
# CPU-OPTIMIZED ALIGNMENT MODEL
# ============================================

class CPUOptimizedAlignment(nn.Module):
    """
    Temporal alignment optimized for CPU execution
    Input: AST [B, 101, 768] and MViTv2 [B, 8, 768] pre-extracted features
    """
    def __init__(self, config: AlignmentConfig):
        super().__init__()
        self.config = config

        # ============================================
        # STEP 1: Alignment to K temporal steps
        # ============================================
        # Linear layers are more CPU-friendly than Conv1d
        self.ast_to_k = nn.Linear(config.ast_temporal_tokens, config.k_steps)
        self.mvit_to_k = nn.Linear(config.mvit_temporal_tokens, config.k_steps)

        # LayerNorm for CPU efficiency
        self.audio_norm = nn.LayerNorm(config.feature_dim)
        self.video_norm = nn.LayerNorm(config.feature_dim)

        # ============================================
        # STEP 2: Projection to common space
        # ============================================
        self.audio_proj = nn.Sequential(
            nn.Linear(config.feature_dim, config.common_dim),
            nn.LayerNorm(config.common_dim),
            nn.ReLU(inplace=True)
        )

        self.video_proj = nn.Sequential(
            nn.Linear(config.feature_dim, config.common_dim),
            nn.LayerNorm(config.common_dim),
            nn.ReLU(inplace=True)
        )

        # ============================================
        # STEP 3: Shared Temporal Positional Encoding
        # ============================================
        self.shared_tpe = nn.Parameter(
            torch.randn(config.k_steps, config.common_dim) * 0.02
        )

        # ============================================
        # OPTIONAL: Simplified cross-attention for CPU
        # ============================================
        if config.use_cross_attention:
            # Single head attention for CPU efficiency
            self.cross_attention = nn.MultiheadAttention(
                config.common_dim,
                num_heads=1,  # Single head for CPU
                dropout=0.0,
                batch_first=True
            )

    def forward(
        self,
        ast_features: torch.Tensor,  # [B, 101, 768]
        mvit_features: torch.Tensor  # [B, 8, 768]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass: K-steps → Projection → TPE

        Args:
            ast_features: [B, 101, 768] pre-extracted AST features
            mvit_features: [B, 8, 768] pre-extracted MViTv2 temporal features
        """
        B = ast_features.shape[0]

        # Process in chunks for memory efficiency on CPU
        chunk_size = 4
        outputs = []

        for i in range(0, B, chunk_size):
            end_idx = min(i + chunk_size, B)

            # Get chunk
            ast_chunk = ast_features[i:end_idx]
            mvit_chunk = mvit_features[i:end_idx]

            # Process chunk
            chunk_output = self._process_chunk(ast_chunk, mvit_chunk)
            outputs.append(chunk_output)

        # Combine outputs
        combined = {}
        for key in outputs[0].keys():
            combined[key] = torch.cat([o[key] for o in outputs], dim=0)

        return combined

    def _process_chunk(
        self,
        ast_chunk: torch.Tensor,
        mvit_chunk: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Process a single chunk (memory efficient)
        """
        # ============================================
        # STEP 1: Normalize and align to K steps
        # ============================================
        # Normalize
        ast_normalized = self.audio_norm(ast_chunk)  # [chunk, 101, 768]
        mvit_normalized = self.video_norm(mvit_chunk)  # [chunk, 8, 768]

        # Map to K temporal steps
        # AST: 101 → K
        ast_weights = F.softmax(self.ast_to_k.weight, dim=1)  # [K, 101]
        audio_k = torch.matmul(ast_weights.unsqueeze(0), ast_normalized)  # [chunk, K, 768]

        # MViTv2: 8 → K
        mvit_weights = F.softmax(self.mvit_to_k.weight, dim=1)  # [K, 8]
        video_k = torch.matmul(mvit_weights.unsqueeze(0), mvit_normalized)  # [chunk, K, 768]

        # ============================================
        # STEP 2: Project to common space
        # ============================================
        audio_projected = self.audio_proj(audio_k)  # [chunk, K, 512]
        video_projected = self.video_proj(video_k)  # [chunk, K, 512]

        # ============================================
        # STEP 3: Add shared TPE
        # ============================================
        audio_with_tpe = audio_projected + self.shared_tpe.unsqueeze(0)
        video_with_tpe = video_projected + self.shared_tpe.unsqueeze(0)

        # ============================================
        # OPTIONAL: Cross-attention (expensive on CPU)
        # ============================================
        if self.config.use_cross_attention:
            # Limit to first 25 tokens for CPU efficiency
            max_k = min(25, self.config.k_steps)

            audio_limited = audio_with_tpe[:, :max_k, :]
            video_limited = video_with_tpe[:, :max_k, :]

            # Simple cross-attention
            audio_attended, _ = self.cross_attention(
                audio_limited, video_limited, video_limited
            )

            # Update only attended portion with small residual
            audio_with_tpe[:, :max_k, :] = audio_limited + 0.1 * audio_attended
            video_with_tpe[:, :max_k, :] = video_limited  # Keep video unchanged

        return {
            'audio_aligned': audio_with_tpe,  # [chunk, K, 512]
            'video_aligned': video_with_tpe,  # [chunk, K, 512]
            'audio_k': audio_k,  # [chunk, K, 768]
            'video_k': video_k,  # [chunk, K, 768]
        }

    def get_global_features(
        self,
        output: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pool K-step features to get global representation
        """
        audio_global = output['audio_aligned'].mean(dim=1)  # [B, 512]
        video_global = output['video_aligned'].mean(dim=1)  # [B, 512]
        return audio_global, video_global

    @torch.no_grad()
    def inference(
        self,
        ast_features: torch.Tensor,
        mvit_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Optimized inference mode for CPU
        """
        self.eval()

        # Process one sample at a time for minimal memory
        B = ast_features.shape[0]
        outputs = []

        for i in range(B):
            single_ast = ast_features[i:i+1]
            single_mvit = mvit_features[i:i+1]

            output = self._process_chunk(single_ast, single_mvit)
            outputs.append(output)

        # Combine
        combined = {}
        for key in outputs[0].keys():
            combined[key] = torch.cat([o[key] for o in outputs], dim=0)

        return combined

# ============================================
# CONTRASTIVE LOSS
# ============================================

class CPUContrastiveLoss(nn.Module):
    """
    Memory-efficient contrastive loss for CPU
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        audio_features: torch.Tensor,
        video_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss with chunked similarity matrix
        """
        B = audio_features.shape[0]

        # Normalize
        audio_norm = F.normalize(audio_features, dim=-1)
        video_norm = F.normalize(video_features, dim=-1)

        # Compute similarity in blocks for memory efficiency
        if B > 32:
            similarity = torch.zeros(B, B)
            block_size = 32

            for i in range(0, B, block_size):
                for j in range(0, B, block_size):
                    i_end = min(i + block_size, B)
                    j_end = min(j + block_size, B)

                    similarity[i:i_end, j:j_end] = torch.matmul(
                        audio_norm[i:i_end],
                        video_norm[j:j_end].T
                    )
        else:
            similarity = torch.matmul(audio_norm, video_norm.T)

        # Scale and compute loss
        similarity = similarity / self.temperature
        labels = torch.arange(B)
        loss = F.cross_entropy(similarity, labels)

        return loss

# ============================================
# DATA LOADING
# ============================================

class PreExtractedFeaturesDataset:
    """
    Simple dataset for pre-extracted features
    """
    def __init__(self, features_path: str, num_samples: int = 1000):
        self.features_path = features_path
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # In practice, load from disk
        # Here we simulate
        ast_features = torch.randn(101, 768)
        mvit_features = torch.randn(8, 768)

        return {
            'ast': ast_features,
            'mvit': mvit_features,
            'idx': idx
        }

    def get_batch(self, indices: List[int]) -> Dict[str, torch.Tensor]:
        """Get a batch of samples"""
        batch_ast = []
        batch_mvit = []

        for idx in indices:
            sample = self.__getitem__(idx)
            batch_ast.append(sample['ast'])
            batch_mvit.append(sample['mvit'])

        return {
            'ast': torch.stack(batch_ast),
            'mvit': torch.stack(batch_mvit)
        }

# ============================================
# TRAINING LOOP
# ============================================

def train_cpu(
    model: CPUOptimizedAlignment,
    dataset: PreExtractedFeaturesDataset,
    num_epochs: int = 10,
    batch_size: int = 4,
    learning_rate: float = 1e-4
):
    """
    CPU-optimized training loop
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = CPUContrastiveLoss(temperature=0.07)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        # Create batches
        indices = list(range(len(dataset)))

        for i in range(0, len(dataset), batch_size):
            batch_indices = indices[i:i + batch_size]
            batch = dataset.get_batch(batch_indices)

            # Forward pass
            output = model(batch['ast'], batch['mvit'])

            # Get global features
            audio_global, video_global = model.get_global_features(output)

            # Compute loss
            loss = loss_fn(audio_global, video_global)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            if i % (10 * batch_size) == 0:
                print(f"Epoch {epoch}, Sample {i}, Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")

# ============================================
# MAIN FUNCTION
# ============================================

def main():
    """
    Main function for CPU execution
    """
    print("Initializing CPU-optimized alignment pipeline...")
    print(f"Using {torch.get_num_threads()} CPU threads")

    # Configuration
    config = AlignmentConfig(
        k_steps=50,
        common_dim=512,
        use_cross_attention=False  # Set to False for faster training
    )

    # Create model
    model = CPUOptimizedAlignment(config)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {num_params:,} parameters")
    print(f"Cross-attention: {'Enabled' if config.use_cross_attention else 'Disabled'}")

    # Create dataset
    dataset = PreExtractedFeaturesDataset(
        features_path="./features",
        num_samples=1000
    )

    # Train
    train_cpu(
        model=model,
        dataset=dataset,
        num_epochs=10,
        batch_size=4,
        learning_rate=1e-4
    )

    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
    }, 'cpu_model_final.pt')

    print("Training completed!")

    # Test inference
    print("\nTesting inference...")
    test_ast = torch.randn(2, 101, 768)
    test_mvit = torch.randn(2, 8, 768)

    with torch.no_grad():
        output = model.inference(test_ast, test_mvit)
        print(f"Inference output shapes:")
        for key, value in output.items():
            print(f"  {key}: {value.shape}")

#
# if __name__ == "__main__":
#     main()