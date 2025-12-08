import torch
import torch.nn.functional as F


def evolutionary_consistency_loss(
        z_f: torch.Tensor,
        z_a: torch.Tensor,
        temperature: float = 0.1,
        reduction: str = "mean",
) -> torch.Tensor:
    """
    Implements the contrastive loss for Face-Audio alignment.

    In the context of the paper (NPVForensics):
    - Original L_EC was for Viseme-Audio alignment (Eq 4-8).
    - L_Info was for Face-Audio alignment (Section 3.4).

    Since the user has excluded visemes, this function now effectively
    implements L_Info (Face-Audio contrastive loss) using the same
    symmetric InfoNCE logic.

    Args:
        z_f: (N, D) Face embeddings in S_fa common space.
        z_a: (N, D) Audio embeddings in S_fa common space.
             N is the batch size (or batch * segments).
        temperature: Scaling factor for logits.

    Returns:
        Scalar loss value.
    """

    assert z_f.shape == z_a.shape, "z_f and z_a must have same shape (N, D)"
    N, D = z_f.shape

    # 1) Normalize for cosine similarity
    z_f_norm = F.normalize(z_f, dim=-1)  # (N, D)
    z_a_norm = F.normalize(z_a, dim=-1)  # (N, D)

    # 2) Similarity matrix S_ij = (f_i · a_j) / tau
    sim = torch.matmul(z_f_norm, z_a_norm.t())  # (N, N)
    sim = sim / temperature

    # 3) Direction 1: Audio as anchor, Face as positive
    #    For each i, positive is sim[i, i], denominator is sum(exp(sim[i, :]))
    pos_a = sim.diag()  # (N,)
    # LogSumExp over j (all faces)
    log_prob_a = pos_a - torch.logsumexp(sim, dim=1)
    loss_a = -log_prob_a  # (N,)

    # 4) Direction 2: Face as anchor, Audio as positive
    #    For each i, positive is sim[i, i], denominator is sum(exp(sim[:, i]))
    pos_f = sim.diag()  # (N,)
    # LogSumExp over i (all audios)
    log_prob_f = pos_f - torch.logsumexp(sim, dim=0)
    loss_f = -log_prob_f  # (N,)

    # 5) Combine symmetric losses
    loss = loss_a + loss_f  # (N,)

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Invalid reduction: {reduction}")