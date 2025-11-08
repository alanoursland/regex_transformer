"""Evaluation metrics computation."""
import torch
from typing import Dict
from collections import defaultdict

def compute_metrics(model, dataloader, device) -> Dict:
    """
    Compute comprehensive evaluation metrics.
    
    Returns dict with:
        - token_acc: Next-token accuracy
        - class_acc: State-class accuracy
        - nll: Negative log-likelihood
        - length_binned: Metrics by sequence length
    """
    model.eval()
    
    total_tokens = 0
    correct_tokens = 0
    total_classes = 0
    correct_classes = 0
    total_nll = 0.0
    
    # Length-binned metrics
    length_bins = defaultdict(lambda: {"correct": 0, "total": 0})
    
    with torch.no_grad():
        for batch in dataloader:
            tokens = batch["tokens"].to(device)
            attn_mask = batch["attn_mask"].to(device)
            next_targets = batch["next_tokens"].to(device)
            class_targets = batch["state_classes"].to(device)
            loss_mask = batch["loss_mask"].to(device)
            
            outputs = model(tokens, attn_mask)
            
            # Next-token accuracy
            next_preds = outputs["next_token_logits"].argmax(dim=-1)
            correct = ((next_preds == next_targets) & loss_mask).sum().item()
            total = loss_mask.sum().item()
            
            correct_tokens += correct
            total_tokens += total
            
            # Class accuracy
            # class_logits: (B, T, num_classes) - predicts state AFTER each token
            # state_classes: (B, T+1) - trace includes start + states after each token
            # We predict state_classes[:, 1:T+1] from class_logits[:, :T, :]
            # But only use first T-1 for loss (last position is after EOS)
            class_preds = outputs["class_logits"].argmax(dim=-1)  # (B, T)
            class_tgt = class_targets[:, 1:]  # (B, T) - states after each token
            class_mask = loss_mask  # (B, T) - but last is excluded
            
            correct_classes += ((class_preds == class_tgt) & class_mask).sum().item()
            total_classes += class_mask.sum().item()
            
            # NLL (cross-entropy) - only on valid positions
            log_probs = torch.nn.functional.log_softmax(outputs["next_token_logits"], dim=-1)
            
            # Clamp targets to valid range (replace -1 with 0 temporarily, will be masked out)
            safe_targets = next_targets.clamp(min=0)
            nll = -log_probs.gather(2, safe_targets.unsqueeze(-1)).squeeze(-1)
            nll_masked = (nll * loss_mask).sum()
            total_nll += nll_masked.item()
            
            # Length-binned accuracy
            for i, length in enumerate((attn_mask.sum(dim=1)).tolist()):
                length = int(length)
                seq_mask = loss_mask[i]
                seq_correct = ((next_preds[i] == next_targets[i]) & seq_mask).sum().item()
                seq_total = seq_mask.sum().item()
                
                length_bins[length]["correct"] += seq_correct
                length_bins[length]["total"] += seq_total
    
    # Compute final metrics
    metrics = {
        "token_acc": correct_tokens / total_tokens if total_tokens > 0 else 0.0,
        "class_acc": correct_classes / total_classes if total_classes > 0 else 0.0,
        "nll": total_nll / total_tokens if total_tokens > 0 else float('inf'),
        "perplexity": torch.exp(torch.tensor(total_nll / total_tokens)).item() if total_tokens > 0 else float('inf'),
    }
    
    # Add length-binned metrics
    length_binned = {}
    for length, stats in sorted(length_bins.items()):
        if stats["total"] > 0:
            length_binned[length] = stats["correct"] / stats["total"]
    
    metrics["length_binned_acc"] = length_binned
    
    return metrics
