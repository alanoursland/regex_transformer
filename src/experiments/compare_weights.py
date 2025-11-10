#!/usr/bin/env python
"""
Compare learned transformer weights to constructed FSM weights.

This script loads a trained model and compares its learned attention weights
to the theoretically constructed QKV matrices from the FSM.

Usage:
    python -m experiments.compare_weights results/aplus/20241110_153000
    python -m experiments.compare_weights results/aplus/20241110_153000 --checkpoint best.pt
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fsm.regex_def import RegexDefinition
from fsm.compile import compile_regex
from model.config import ModelConfig
from model.transformer import RegexTransformer
from train.checkpoint import load_checkpoint


def load_experiment(results_dir: Path, checkpoint_name: str = "best.pt"):
    """Load trained model and metadata from results directory."""

    results_dir = Path(results_dir)

    # Load metadata
    with open(results_dir / "metadata.json") as f:
        metadata = json.load(f)

    # Rebuild FSM
    with open(results_dir / "regex_def.json") as f:
        regex_def_data = json.load(f)

    regex_def = RegexDefinition(
        alphabet=tuple(regex_def_data["alphabet"]),
        patterns=tuple(tuple(p) for p in regex_def_data["patterns"])
    )
    fsm = compile_regex(regex_def)

    # Rebuild model config
    model_cfg = ModelConfig(**metadata["model_config"])

    # Load model
    model = RegexTransformer(model_cfg)
    checkpoint_path = results_dir / checkpoint_name

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt_info = load_checkpoint(checkpoint_path, model)

    # Load constructed weights
    construction = torch.load(results_dir / "fsm_construction.pt")

    return {
        "model": model,
        "fsm": fsm,
        "metadata": metadata,
        "checkpoint_info": ckpt_info,
        "construction": construction,
        "results_dir": results_dir,
    }


def extract_value_matrices(model: RegexTransformer):
    """
    Extract value projection matrices from the model.

    Returns:
        V: (n_heads, d_model, d_head) tensor
    """
    # Access the attention module
    attn = model.block.attn

    # V projection weight is (d_model, d_model)
    V_weight = attn.v_proj.weight.data  # (d_model, d_model)

    # Reshape to (n_heads, d_head, d_model)
    n_heads = attn.n_heads
    d_head = attn.d_head

    V_reshaped = V_weight.view(n_heads, d_head, -1)  # (n_heads, d_head, d_model)

    return V_reshaped


def compare_value_matrices(learned_V, constructed_V, num_heads, num_states):
    """
    Compare learned value matrices to constructed ones.

    Args:
        learned_V: (n_heads, d_head, d_model) tensor from model
        constructed_V: List of matrices from FSM construction [n_heads][from_state][to_state]
        num_heads: Number of attention heads
        num_states: Number of FSM states

    Returns:
        dict with comparison metrics
    """

    metrics = {
        "head_similarities": [],
        "head_correlations": [],
        "frobenius_distances": [],
        "per_head_analysis": [],
    }

    # Convert constructed V to tensor
    # constructed_V is [n_heads][from_state][to_state]
    constructed_V_tensor = torch.tensor(constructed_V, dtype=torch.float32)  # (n_heads, n_states, n_states)

    for head_idx in range(num_heads):
        learned_head = learned_V[head_idx]  # (d_head, d_model)
        constructed_head = constructed_V_tensor[head_idx]  # (n_states, n_states)

        # Extract the relevant portion of learned weights
        # The learned matrix is (d_head, d_model)
        # The constructed matrix is (n_states, n_states)
        # We need to match dimensions

        # Take first n_states dimensions from both axes
        d_head = learned_head.shape[0]
        d_model = learned_head.shape[1]

        if d_head >= num_states and d_model >= num_states:
            learned_submatrix = learned_head[:num_states, :num_states]

            # Compute similarity metrics
            # 1. Cosine similarity (flatten and compare)
            learned_flat = learned_submatrix.flatten()
            constructed_flat = constructed_head.flatten()

            cosine_sim = F.cosine_similarity(
                learned_flat.unsqueeze(0),
                constructed_flat.unsqueeze(0)
            ).item()

            # 2. Correlation
            corr = np.corrcoef(
                learned_flat.numpy(),
                constructed_flat.numpy()
            )[0, 1]

            # 3. Frobenius norm distance
            frobenius_dist = torch.norm(
                learned_submatrix - constructed_head,
                p='fro'
            ).item()

            # 4. Relative error
            constructed_norm = torch.norm(constructed_head, p='fro').item()
            relative_error = frobenius_dist / (constructed_norm + 1e-8)

            metrics["head_similarities"].append(cosine_sim)
            metrics["head_correlations"].append(corr)
            metrics["frobenius_distances"].append(frobenius_dist)

            metrics["per_head_analysis"].append({
                "head": head_idx,
                "cosine_similarity": cosine_sim,
                "correlation": corr,
                "frobenius_distance": frobenius_dist,
                "relative_error": relative_error,
                "learned_norm": torch.norm(learned_submatrix, p='fro').item(),
                "constructed_norm": constructed_norm,
            })

    # Aggregate metrics
    metrics["mean_similarity"] = np.mean(metrics["head_similarities"])
    metrics["mean_correlation"] = np.mean(metrics["head_correlations"])
    metrics["mean_frobenius"] = np.mean(metrics["frobenius_distances"])

    return metrics


def visualize_weight_comparison(learned_V, constructed_V, head_idx, num_states):
    """Print side-by-side comparison of weight matrices for a specific head."""

    constructed = torch.tensor(constructed_V[head_idx], dtype=torch.float32)[:num_states, :num_states]
    learned_submatrix = learned_V[head_idx][:num_states, :num_states]

    print(f"\n{'='*70}")
    print(f"HEAD {head_idx} - Weight Matrix Comparison")
    print(f"{'='*70}")

    print("\nConstructed (FSM):")
    print(constructed.numpy())

    print("\nLearned (Transformer):")
    print(learned_submatrix.numpy())

    print("\nDifference (Learned - Constructed):")
    diff = learned_submatrix - constructed
    print(diff.numpy())

    print(f"\nStatistics:")
    print(f"  Constructed - mean: {constructed.mean():.4f}, std: {constructed.std():.4f}")
    print(f"  Learned     - mean: {learned_submatrix.mean():.4f}, std: {learned_submatrix.std():.4f}")
    print(f"  Difference  - mean: {diff.mean():.4f}, std: {diff.std():.4f}")


def analyze_attention_patterns(model: RegexTransformer, fsm, test_strings: list):
    """
    Analyze whether the model's attention patterns match FSM transitions.

    Args:
        model: Trained transformer
        fsm: Compiled FSM
        test_strings: List of test strings

    Returns:
        dict with analysis results
    """

    model.eval()
    results = []

    with torch.no_grad():
        for test_str in test_strings:
            if not test_str:
                continue

            # Get FSM trace
            tokens = fsm.tokens_from_string(test_str)
            fsm_states = fsm.trace(tokens)

            # Get model predictions
            token_ids = torch.tensor([fsm.alphabet.index(c) for c in test_str]).unsqueeze(0)

            # Forward pass
            outputs = model(token_ids)

            # Get class predictions
            class_logits = outputs["class_logits"].squeeze(0)  # (T, num_classes)
            predicted_classes = class_logits.argmax(dim=-1).tolist()

            # Compare
            actual_classes = [fsm.classify(fsm_states[i]) for i in range(len(fsm_states))]
            actual_class_ids = [fsm.classes.index(c) for c in actual_classes]

            match = predicted_classes == actual_class_ids[1:]  # Skip initial state

            results.append({
                "string": test_str,
                "fsm_states": fsm_states,
                "fsm_classes": actual_classes,
                "predicted_classes": [fsm.classes[c] for c in predicted_classes],
                "match": match,
            })

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compare learned weights to FSM construction"
    )
    parser.add_argument(
        "results_dir",
        type=str,
        help="Path to results directory"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best.pt",
        help="Checkpoint file to load (default: best.pt)"
    )
    parser.add_argument(
        "--visualize_heads",
        type=int,
        nargs="+",
        default=None,
        help="Visualize specific heads (default: all)"
    )
    parser.add_argument(
        "--test_strings",
        type=str,
        nargs="+",
        default=["a", "aa", "aaa", "b", "ab", "ba"],
        help="Test strings for behavior analysis"
    )

    args = parser.parse_args()

    # Load experiment
    print(f"Loading experiment from {args.results_dir}...")
    exp = load_experiment(args.results_dir, args.checkpoint)

    model = exp["model"]
    fsm = exp["fsm"]
    metadata = exp["metadata"]
    construction = exp["construction"]

    print(f"\nExperiment Info:")
    print(f"  Pattern: {metadata['pattern']}")
    print(f"  FSM States: {metadata['fsm']['num_states']}")
    print(f"  Model Heads: {metadata['model_config']['n_heads']}")
    print(f"  Model d_model: {metadata['model_config']['d_model']}")

    # Extract learned weights
    print(f"\nExtracting learned value matrices...")
    learned_V = extract_value_matrices(model)

    print(f"  Shape: {learned_V.shape}")

    # Compare weights
    print(f"\nComparing weights to FSM construction...")
    comparison = compare_value_matrices(
        learned_V,
        construction["V"],
        metadata['model_config']['n_heads'],
        metadata['fsm']['num_states']
    )

    print(f"\n{'='*70}")
    print(f"WEIGHT COMPARISON RESULTS")
    print(f"{'='*70}")

    print(f"\nAggregate Metrics:")
    print(f"  Mean Cosine Similarity: {comparison['mean_similarity']:.4f}")
    print(f"  Mean Correlation:       {comparison['mean_correlation']:.4f}")
    print(f"  Mean Frobenius Dist:    {comparison['mean_frobenius']:.4f}")

    print(f"\nPer-Head Analysis:")
    for head_analysis in comparison["per_head_analysis"]:
        print(f"  Head {head_analysis['head']}:")
        print(f"    Cosine Similarity: {head_analysis['cosine_similarity']:7.4f}")
        print(f"    Correlation:       {head_analysis['correlation']:7.4f}")
        print(f"    Relative Error:    {head_analysis['relative_error']:7.4f}")

    # Visualize specific heads
    if args.visualize_heads:
        for head_idx in args.visualize_heads:
            if head_idx < metadata['model_config']['n_heads']:
                visualize_weight_comparison(
                    learned_V,
                    construction["V"],
                    head_idx,
                    metadata['fsm']['num_states']
                )

    # Analyze behavior on test strings
    print(f"\n{'='*70}")
    print(f"BEHAVIORAL ANALYSIS")
    print(f"{'='*70}")

    behavior = analyze_attention_patterns(model, fsm, args.test_strings)

    for result in behavior:
        match_str = "✓" if result["match"] else "✗"
        print(f"\n{match_str} '{result['string']}'")
        print(f"  FSM States:  {result['fsm_states']}")
        print(f"  FSM Classes: {result['fsm_classes']}")
        print(f"  Predicted:   {result['predicted_classes']}")

    # Save analysis results
    output_path = exp["results_dir"] / "weight_comparison.json"
    with open(output_path, "w") as f:
        json.dump({
            "comparison_metrics": {
                "mean_similarity": comparison["mean_similarity"],
                "mean_correlation": comparison["mean_correlation"],
                "mean_frobenius": comparison["mean_frobenius"],
                "per_head": comparison["per_head_analysis"],
            },
            "behavior_analysis": behavior,
        }, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
