"""Checkpointing utilities."""
import torch
from pathlib import Path

def save_checkpoint(path: Path, model, optimizer, epoch, step, metrics, seed):
    """Save checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix('.tmp')
    
    torch.save({
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict() if optimizer else None,
        'epoch': epoch,
        'step': step,
        'metrics': metrics,
        'seed': seed,
    }, tmp_path)
    
    tmp_path.rename(path)

def load_checkpoint(path: Path, model, optimizer=None):
    """Load checkpoint."""
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model_state'])
    
    if optimizer and ckpt['optimizer_state']:
        optimizer.load_state_dict(ckpt['optimizer_state'])
    
    return {
        'epoch': ckpt['epoch'],
        'step': ckpt['step'],
        'metrics': ckpt['metrics'],
        'seed': ckpt['seed'],
    }
