"""Multi-task loss computation."""
import torch
import torch.nn.functional as F

def compute_multi_task_loss(
    outputs: dict,
    batch: dict,
    lambda_next: float = 1.0,
    lambda_class: float = 0.5,
) -> dict:
    """
    Compute multi-task loss.
    
    Args:
        outputs: Model outputs with next_token_logits, class_logits
        batch: Batch dict with next_tokens, state_classes, loss_mask
        lambda_next: Weight for next-token loss
        lambda_class: Weight for class loss
    
    Returns:
        Dict with component losses and total
    """
    B, T = batch["tokens"].shape
    
    # Next-token loss (masked, ignore -1 padding)
    next_logits = outputs["next_token_logits"]  # (B, T, vocab)
    next_targets = batch["next_tokens"]  # (B, T)
    loss_mask = batch["loss_mask"]  # (B, T)
    
    next_loss = F.cross_entropy(
        next_logits.reshape(-1, next_logits.size(-1)),
        next_targets.reshape(-1),
        ignore_index=-1,
        reduction='none'
    ).reshape(B, T)
    next_loss = (next_loss * loss_mask).sum() / loss_mask.sum().clamp(min=1)
    
    # Class loss (masked, exclude last position)
    class_logits = outputs["class_logits"][:, :-1, :]  # (B, T-1, num_classes)
    class_targets = batch["state_classes"][:, 1:T]  # (B, T-1) - skip start state
    class_mask = loss_mask[:, :-1]  # (B, T-1)
    
    class_loss = F.cross_entropy(
        class_logits.reshape(-1, class_logits.size(-1)),
        class_targets.reshape(-1),
        reduction='none'
    ).reshape(B, T-1)
    class_loss = (class_loss * class_mask).sum() / class_mask.sum().clamp(min=1)
    
    # Total loss
    total_loss = lambda_next * next_loss + lambda_class * class_loss
    
    return {
        "loss": total_loss,
        "next_token_loss": next_loss,
        "class_loss": class_loss,
    }
