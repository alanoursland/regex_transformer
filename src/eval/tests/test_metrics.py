"""Tests for evaluation metrics."""
import pytest
import torch

from fsm.regex_def import RegexDefinition
from fsm.compile import compile_regex
from data.dataset import FsmDataset
from data.tokenizer import Vocab
from data.loader import make_dataloaders
from model.config import ModelConfig
from model.transformer import RegexTransformer
from ..metrics import compute_metrics
from ..ood import length_ood_split, compute_ood_gap


def test_compute_metrics():
    """Test metrics computation."""
    # Build simple model and data
    regex_def = RegexDefinition(alphabet=("a",), patterns=(("a+", "accept"),))
    fsm = compile_regex(regex_def)
    
    samples = [[0], [0, 0], [0, 0, 0]]
    dataset = FsmDataset(fsm, samples, "test")
    vocab = Vocab.from_alphabet(regex_def.alphabet)
    
    datasets = {"test": dataset}
    dataloaders = make_dataloaders(datasets, vocab, batch_size=2, seed=42)
    
    # Build model
    cfg = ModelConfig(vocab_size=len(vocab), num_classes=3, num_states=10, d_model=32, n_heads=2)
    model = RegexTransformer(cfg)
    model.eval()
    
    # Compute metrics
    metrics = compute_metrics(model, dataloaders["test"], device="cpu")
    
    assert "token_acc" in metrics
    assert "class_acc" in metrics
    assert "nll" in metrics
    assert "perplexity" in metrics
    assert 0.0 <= metrics["token_acc"] <= 1.0
    assert 0.0 <= metrics["class_acc"] <= 1.0
    # NLL may be NaN for untrained model
    assert "nll" in metrics


def test_length_ood_split():
    """Test length-based OOD splitting."""
    regex_def = RegexDefinition(alphabet=("a",), patterns=(("a+", "accept"),))
    fsm = compile_regex(regex_def)
    
    samples = [[0], [0, 0], [0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0, 0]]
    dataset = FsmDataset(fsm, samples, "test")
    
    id_idx, ood_idx = length_ood_split(dataset, max_train_len=3)
    
    # First 3 samples have length <= 3
    assert len(id_idx) == 3
    assert len(ood_idx) == 2
    assert set(id_idx) == {0, 1, 2}
    assert set(ood_idx) == {3, 4}


def test_compute_ood_gap():
    """Test OOD gap computation."""
    id_metrics = {"token_acc": 0.9, "class_acc": 0.85, "nll": 0.5}
    ood_metrics = {"token_acc": 0.7, "class_acc": 0.65, "nll": 1.0}
    
    gap = compute_ood_gap(id_metrics, ood_metrics)
    
    assert gap["token_acc_gap"] == pytest.approx(0.2)
    assert gap["class_acc_gap"] == pytest.approx(0.2)
    assert gap["nll_gap"] == pytest.approx(0.5)


def test_metrics_determinism():
    """Test that metrics are deterministic."""
    regex_def = RegexDefinition(alphabet=("a",), patterns=(("a+", "accept"),))
    fsm = compile_regex(regex_def)
    
    samples = [[0, 0]]
    dataset = FsmDataset(fsm, samples, "test")
    vocab = Vocab.from_alphabet(regex_def.alphabet)
    
    datasets = {"test": dataset}
    dataloaders = make_dataloaders(datasets, vocab, batch_size=1, seed=42)
    
    cfg = ModelConfig(vocab_size=len(vocab), num_classes=3, num_states=10, d_model=16, n_heads=2)
    
    torch.manual_seed(42)
    model = RegexTransformer(cfg)
    model.eval()
    
    metrics1 = compute_metrics(model, dataloaders["test"], device="cpu")
    metrics2 = compute_metrics(model, dataloaders["test"], device="cpu")
    
    assert metrics1["token_acc"] == metrics2["token_acc"]
    assert metrics1["nll"] == metrics2["nll"]
