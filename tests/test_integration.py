"""End-to-end integration test."""
import pytest
import torch
import tempfile
from pathlib import Path

from fsm.regex_def import RegexDefinition
from fsm.compile import compile_regex
from data.generator import generate_corpus, GenConfig
from data.tokenizer import Vocab
from data.dataset import FsmDataset
from data.loader import make_dataloaders
from model.config import ModelConfig
from model.transformer import RegexTransformer
from train.loop import train_one_experiment, TrainConfig
from eval.metrics import compute_metrics


def test_end_to_end_pipeline():
    """Test complete pipeline from regex to trained model."""
    # 1. Define regex and compile to FSM
    regex_def = RegexDefinition(
        alphabet=("a", "b"),
        patterns=(("a+", "accept"),)
    )
    
    # 2. Create model and train config
    model_cfg = ModelConfig(
        vocab_size=4,  # PAD, EOS, a, b
        num_classes=3,  # accept, incomplete, reject
        num_states=20,
        d_model=32,
        n_heads=2,
        d_mlp=64,
        max_seq_len=16,
    )
    
    train_cfg = TrainConfig(
        epochs=3,
        lr=1e-3,
        batch_size=8,
        seed=42,
        device="cpu",
        patience=10,
    )
    
    # 3. Train model
    with tempfile.TemporaryDirectory() as tmpdir:
        results = train_one_experiment(
            regex_def,
            model_cfg,
            train_cfg,
            results_dir=Path(tmpdir),
            n_samples=64,  # Small dataset
        )
        
        # 4. Verify training completed
        assert "val_metrics" in results
        assert "test_metrics" in results
        
        # 5. Verify metrics structure
        val_metrics = results["val_metrics"]
        assert "token_acc" in val_metrics
        assert "class_acc" in val_metrics


def test_overfit_micro_dataset():
    """Test that model can overfit a tiny dataset."""
    # This is a critical sanity check
    regex_def = RegexDefinition(
        alphabet=("a",),
        patterns=(("a+", "accept"),)
    )
    
    fsm = compile_regex(regex_def)
    
    # Generate tiny dataset (8 samples)
    p_class = {cls: 1.0/len(fsm.classes) for cls in fsm.classes}
    gen_cfg = GenConfig(L_min=1, L_max=4, p_class=p_class)
    samples, class_names, _ = generate_corpus(fsm, gen_cfg, n_samples=8, seed=42)
    
    # Use same samples for train and val (overfit test)
    datasets = {
        "train": FsmDataset(fsm, samples, "train"),
        "val": FsmDataset(fsm, samples, "val"),
        "test": FsmDataset(fsm, samples, "test"),
    }
    
    vocab = Vocab.from_alphabet(regex_def.alphabet)
    dataloaders = make_dataloaders(datasets, vocab, batch_size=4, seed=42)
    
    # Build small model
    cfg = ModelConfig(
        vocab_size=len(vocab),
        num_classes=len(fsm.classes),
        num_states=fsm.states,
        d_model=32,
        n_heads=2,
        d_mlp=64,
        max_seq_len=16,
    )
    
    model = RegexTransformer(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    
    # Train for several epochs
    model.train()
    for epoch in range(20):
        for batch in dataloaders["train"]:
            tokens = batch["tokens"]
            attn_mask = batch["attn_mask"]
            
            outputs = model(tokens, attn_mask)
            
            from train.losses import compute_multi_task_loss
            losses = compute_multi_task_loss(outputs, batch)
            
            optimizer.zero_grad()
            losses["loss"].backward()
            optimizer.step()
    
    # Evaluate - should have high accuracy on this tiny dataset
    metrics = compute_metrics(model, dataloaders["val"], device="cpu")
    
    # Model should achieve reasonable accuracy (>50%) on tiny overfit dataset
    assert metrics["token_acc"] > 0.5, f"Failed to overfit: acc={metrics['token_acc']}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
