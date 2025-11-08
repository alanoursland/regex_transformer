"""Training loop."""
import torch
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from ..fsm.regex_def import RegexDefinition
from ..fsm.compile import compile_regex
from ..data.generator import generate_corpus, GenConfig
from ..data.tokenizer import Vocab
from ..data.dataset import FsmDataset
from ..data.loader import make_dataloaders
from ..model.config import ModelConfig
from ..model.transformer import RegexTransformer
from .losses import compute_multi_task_loss
from .optim import build_optimizer, clip_gradients
from .checkpoint import save_checkpoint
from ..eval.metrics import compute_metrics as eval_compute_metrics

@dataclass
class TrainConfig:
    """Training configuration."""
    epochs: int = 10
    lr: float = 1e-3
    batch_size: int = 32
    seed: int = 42
    device: str = "cpu"
    log_every: int = 10
    patience: int = 5

def set_seed(seed: int):
    """Set all random seeds."""
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate(model, dataloader, device):
    """Evaluate model on a dataset."""
    return eval_compute_metrics(model, dataloader, device)

def train_one_experiment(
    regex_def: RegexDefinition,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    results_dir: Path,
    n_samples: int = 1000,
):
    """Train a single experiment."""
    set_seed(train_cfg.seed)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate data
    fsm = compile_regex(regex_def)
    p_class = {cls: 1.0/len(fsm.classes) for cls in fsm.classes}
    gen_cfg = GenConfig(L_min=1, L_max=model_cfg.max_seq_len//2, p_class=p_class)
    samples, class_names, report = generate_corpus(fsm, gen_cfg, n_samples, seed=train_cfg.seed)
    
    # Split data (simple 80/10/10)
    n_train = int(0.8 * len(samples))
    n_val = int(0.1 * len(samples))
    
    datasets = {
        "train": FsmDataset(fsm, samples[:n_train], "train"),
        "val": FsmDataset(fsm, samples[n_train:n_train+n_val], "val"),
        "test": FsmDataset(fsm, samples[n_train+n_val:], "test"),
    }
    
    vocab = Vocab.from_alphabet(regex_def.alphabet)
    dataloaders = make_dataloaders(datasets, vocab, train_cfg.batch_size, train_cfg.seed)
    
    # Build model
    model = RegexTransformer(model_cfg).to(train_cfg.device)
    optimizer = build_optimizer(model, train_cfg.lr)
    
    # Training
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(train_cfg.epochs):
        model.train()
        train_loss = 0
        
        for batch_idx, batch in enumerate(dataloaders["train"]):
            tokens = batch["tokens"].to(train_cfg.device)
            attn_mask = batch["attn_mask"].to(train_cfg.device)
            
            outputs = model(tokens, attn_mask)
            losses = compute_multi_task_loss(outputs, {k: v.to(train_cfg.device) for k, v in batch.items()})
            
            optimizer.zero_grad()
            losses["loss"].backward()
            clip_gradients(model, max_norm=1.0)
            optimizer.step()
            
            train_loss += losses["loss"].item()
        
        # Validation
        val_metrics = evaluate(model, dataloaders["val"], train_cfg.device)
        
        print(f"Epoch {epoch+1}/{train_cfg.epochs} - "
              f"Train Loss: {train_loss/len(dataloaders['train']):.4f}, "
              f"Val NLL: {val_metrics.get('nll', float('nan')):.4f}, "
              f"Val Acc: {val_metrics['token_acc']:.4f}")
        
        # Early stopping
        val_loss = val_metrics.get("nll", float("inf"))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(
                results_dir / "best.pt",
                model, optimizer, epoch, 0, val_metrics, train_cfg.seed
            )
        else:
            patience_counter += 1
            if patience_counter >= train_cfg.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Test
    test_metrics = evaluate(model, dataloaders["test"], train_cfg.device)
    
    # Save final metrics
    with open(results_dir / "metrics.json", "w") as f:
        json.dump({
            "val": val_metrics,
            "test": test_metrics,
            "best_val_loss": best_val_loss,
        }, f, indent=2)
    
    return {
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
