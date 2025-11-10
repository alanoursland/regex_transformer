#!/usr/bin/env python
"""
Train a transformer model on a regex pattern and save results.

This script trains a single-layer transformer to learn FSM behavior from a regex pattern.
Results are saved in a timestamped directory with full reproducibility metadata.

Usage:
    python -m src.experiments.train_model --pattern "a+" --epochs 50
    python -m src.experiments.train_model --pattern "a*b*" --n_samples 5000 --batch_size 64
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fsm.regex_def import RegexDefinition
from fsm.compile import compile_regex
from data.generator import generate_corpus, GenConfig
from data.tokenizer import Vocab
from data.dataset import FsmDataset
from data.loader import make_dataloaders
from model.config import ModelConfig
from model.transformer import RegexTransformer
from model.fsm_construction import construct_qkv_from_fsm
from train.loop import set_seed, evaluate
from train.losses import compute_multi_task_loss
from train.optim import build_optimizer, clip_gradients
from train.checkpoint import save_checkpoint


def create_results_dir(pattern: str, base_dir: Path = None) -> Path:
    """
    Create a timestamped results directory.

    Structure: results/{pattern_clean}/{timestamp}/

    Args:
        pattern: Regex pattern string
        base_dir: Base directory for results (default: src/results)

    Returns:
        Path to created results directory
    """
    if base_dir is None:
        base_dir = Path(__file__).parent.parent / "results"

    # Clean pattern for directory name (replace special chars)
    pattern_clean = pattern.replace("+", "plus").replace("*", "star").replace("|", "or")
    pattern_clean = pattern_clean.replace("(", "").replace(")", "").replace("?", "opt")
    pattern_clean = pattern_clean[:50]  # Limit length

    # Timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create directory
    results_dir = base_dir / pattern_clean / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)

    return results_dir


def save_experiment_metadata(
    results_dir: Path,
    regex_def: RegexDefinition,
    fsm,
    model_cfg: ModelConfig,
    train_args: dict,
):
    """Save full experiment metadata for reproducibility."""

    metadata = {
        "pattern": regex_def.patterns[0][0],
        "alphabet": list(regex_def.alphabet),
        "timestamp": datetime.now().isoformat(),
        "fsm": {
            "num_states": fsm.states,
            "num_classes": len(fsm.classes),
            "classes": list(fsm.classes),
            "alphabet_size": len(fsm.alphabet),
        },
        "model_config": {
            "vocab_size": model_cfg.vocab_size,
            "num_classes": model_cfg.num_classes,
            "num_states": model_cfg.num_states,
            "d_model": model_cfg.d_model,
            "n_heads": model_cfg.n_heads,
            "d_mlp": model_cfg.d_mlp,
            "dropout": model_cfg.dropout,
            "max_seq_len": model_cfg.max_seq_len,
            "tie_weights": model_cfg.tie_weights,
            "positional_type": model_cfg.positional_type,
        },
        "training": train_args,
    }

    with open(results_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Save regex definition separately
    with open(results_dir / "regex_def.json", "w") as f:
        json.dump({
            "alphabet": list(regex_def.alphabet),
            "patterns": regex_def.patterns,
        }, f, indent=2)


def save_fsm_construction(results_dir: Path, fsm):
    """Save the constructed QKV matrices from the FSM."""
    qkv = construct_qkv_from_fsm(fsm)

    # Convert to tensors and save
    torch.save({
        'V': qkv['V'],
        'initial_state': qkv['initial_state'],
        'num_states': fsm.states,
        'alphabet': fsm.alphabet,
    }, results_dir / "fsm_construction.pt")


def train_model(args):
    """Main training function."""

    # Set seed
    set_seed(args.seed)

    # Create results directory
    results_dir = create_results_dir(args.pattern)
    print(f"Results directory: {results_dir}")

    # Build regex and compile FSM
    alphabet = tuple(args.alphabet) if args.alphabet else ("a", "b")
    regex_def = RegexDefinition(
        alphabet=alphabet,
        patterns=((args.pattern, "accept"),)
    )
    fsm = compile_regex(regex_def)

    print(f"\nFSM compiled:")
    print(f"  States: {fsm.states}")
    print(f"  Classes: {fsm.classes}")
    print(f"  Alphabet: {fsm.alphabet}")

    # Save FSM construction baseline
    save_fsm_construction(results_dir, fsm)

    # Generate data
    print(f"\nGenerating {args.n_samples} samples...")
    p_class = {cls: 1.0/len(fsm.classes) for cls in fsm.classes}
    gen_cfg = GenConfig(
        L_min=args.min_len,
        L_max=args.max_len,
        p_class=p_class
    )
    samples, class_names, report = generate_corpus(
        fsm, gen_cfg, args.n_samples, seed=args.seed
    )

    print(f"  Generated: {len(samples)} samples")
    print(f"  Class distribution: {dict(report['class_distribution'])}")

    # Split data (80/10/10)
    n_train = int(0.8 * len(samples))
    n_val = int(0.1 * len(samples))

    datasets = {
        "train": FsmDataset(fsm, samples[:n_train], "train"),
        "val": FsmDataset(fsm, samples[n_train:n_train+n_val], "val"),
        "test": FsmDataset(fsm, samples[n_train+n_val:], "test"),
    }

    vocab = Vocab.from_alphabet(alphabet)
    dataloaders = make_dataloaders(
        datasets, vocab, args.batch_size, args.seed
    )

    # Build model config
    model_cfg = ModelConfig(
        vocab_size=len(vocab),
        num_classes=len(fsm.classes),
        num_states=fsm.states,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_mlp=args.d_mlp,
        dropout=args.dropout,
        max_seq_len=args.max_len * 2,  # Extra buffer
        tie_weights=args.tie_weights,
        positional_type=args.positional_type,
    )

    # Save experiment metadata
    train_args_dict = vars(args)
    save_experiment_metadata(results_dir, regex_def, fsm, model_cfg, train_args_dict)

    # Build model
    device = torch.device(args.device)
    model = RegexTransformer(model_cfg).to(device)
    optimizer = build_optimizer(model, args.lr)

    print(f"\nModel:")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Device: {device}")

    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    best_val_loss = float('inf')
    patience_counter = 0
    history = []

    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0
        train_batches = 0

        for batch in dataloaders["train"]:
            tokens = batch["tokens"].to(device)
            attn_mask = batch["attn_mask"].to(device)

            outputs = model(tokens, attn_mask)
            losses = compute_multi_task_loss(
                outputs,
                {k: v.to(device) for k, v in batch.items()}
            )

            optimizer.zero_grad()
            losses["loss"].backward()
            clip_gradients(model, max_norm=1.0)
            optimizer.step()

            train_loss += losses["loss"].item()
            train_batches += 1

        avg_train_loss = train_loss / train_batches

        # Validation
        val_metrics = evaluate(model, dataloaders["val"], device)

        # Log
        epoch_info = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_nll": val_metrics.get("nll", float("nan")),
            "val_token_acc": val_metrics.get("token_acc", 0.0),
            "val_class_acc": val_metrics.get("class_acc", 0.0),
        }
        history.append(epoch_info)

        if (epoch + 1) % args.log_every == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{args.epochs} | "
                  f"Train: {avg_train_loss:.4f} | "
                  f"Val NLL: {val_metrics.get('nll', float('nan')):.4f} | "
                  f"Val Tok Acc: {val_metrics.get('token_acc', 0.0):.3f} | "
                  f"Val Cls Acc: {val_metrics.get('class_acc', 0.0):.3f}")

        # Early stopping
        val_loss = val_metrics.get("nll", float("inf"))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # Save best checkpoint
            save_checkpoint(
                results_dir / "best.pt",
                model, optimizer, epoch, 0, val_metrics, args.seed
            )
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    # Final evaluation on test set
    print("\nFinal evaluation on test set...")
    test_metrics = evaluate(model, dataloaders["test"], device)

    print(f"Test NLL: {test_metrics.get('nll', float('nan')):.4f}")
    print(f"Test Token Acc: {test_metrics.get('token_acc', 0.0):.3f}")
    print(f"Test Class Acc: {test_metrics.get('class_acc', 0.0):.3f}")

    # Save final checkpoint
    save_checkpoint(
        results_dir / "final.pt",
        model, optimizer, epoch, 0, test_metrics, args.seed
    )

    # Save training history
    with open(results_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Save final metrics
    with open(results_dir / "metrics.json", "w") as f:
        json.dump({
            "val": val_metrics,
            "test": test_metrics,
            "best_val_loss": best_val_loss,
        }, f, indent=2)

    print(f"\nResults saved to: {results_dir}")
    print(f"  - best.pt: Best model checkpoint")
    print(f"  - final.pt: Final model checkpoint")
    print(f"  - fsm_construction.pt: Theoretical QKV matrices")
    print(f"  - metadata.json: Full experiment config")
    print(f"  - history.json: Training history")
    print(f"  - metrics.json: Final metrics")

    return results_dir


def main():
    parser = argparse.ArgumentParser(
        description="Train transformer on regex pattern"
    )

    # Pattern
    parser.add_argument(
        "--pattern",
        type=str,
        required=True,
        help="Regex pattern to learn (e.g., 'a+', 'a*b*')"
    )
    parser.add_argument(
        "--alphabet",
        type=str,
        nargs="+",
        default=None,
        help="Alphabet symbols (default: ['a', 'b'])"
    )

    # Data generation
    parser.add_argument(
        "--n_samples",
        type=int,
        default=2000,
        help="Number of training samples to generate"
    )
    parser.add_argument(
        "--min_len",
        type=int,
        default=1,
        help="Minimum sequence length"
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=20,
        help="Maximum sequence length"
    )

    # Model architecture
    parser.add_argument(
        "--d_model",
        type=int,
        default=64,
        help="Model dimension"
    )
    parser.add_argument(
        "--n_heads",
        type=int,
        default=4,
        help="Number of attention heads"
    )
    parser.add_argument(
        "--d_mlp",
        type=int,
        default=256,
        help="MLP hidden dimension"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate"
    )
    parser.add_argument(
        "--tie_weights",
        action="store_true",
        help="Tie embedding and output weights"
    )
    parser.add_argument(
        "--positional_type",
        type=str,
        choices=["sin", "none"],
        default="sin",
        help="Positional encoding type"
    )

    # Training
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience"
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=5,
        help="Log every N epochs"
    )

    # System
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    # Train
    results_dir = train_model(args)

    print(f"\nDone! Use this path for weight comparison:")
    print(f"  python -m src.experiments.compare_weights {results_dir}")


if __name__ == "__main__":
    main()
