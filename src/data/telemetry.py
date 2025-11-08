"""Telemetry and reporting for data generation."""

from dataclasses import dataclass, asdict
from typing import Dict, List
from collections import Counter
import json


@dataclass
class DataReport:
    """
    Report capturing dataset statistics and coverage.

    Attributes:
        n_samples: Total number of samples generated
        length_histogram: Dict mapping length -> count
        class_histogram: Dict mapping class_name -> count
        edge_coverage: Number of unique edges covered
        state_coverage: Number of unique states visited
        reject_subtypes: Dict mapping subtype -> count (if applicable)
        failed_attempts: Number of failed generation attempts
        retry_rate: Fraction of samples that required retries
    """

    n_samples: int
    length_histogram: Dict[int, int]
    class_histogram: Dict[str, int]
    edge_coverage: int
    state_coverage: int
    reject_subtypes: Dict[str, int]
    failed_attempts: int
    retry_rate: float

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "DataReport":
        """Deserialize from JSON string."""
        data = json.loads(json_str)
        # Convert string keys back to ints for length_histogram
        if 'length_histogram' in data:
            data['length_histogram'] = {int(k): v for k, v in data['length_histogram'].items()}
        return cls(**data)

    def print_report(self) -> None:
        """Print human-readable summary."""
        print("=== Data Generation Report ===")
        print(f"Total samples: {self.n_samples}")
        print(f"Failed attempts: {self.failed_attempts} ({self.retry_rate:.2%} retry rate)")
        print()

        print("Class distribution:")
        for class_name, count in sorted(self.class_histogram.items()):
            pct = 100.0 * count / self.n_samples if self.n_samples > 0 else 0
            print(f"  {class_name:12} {count:6} ({pct:5.1f}%)")
        print()

        print("Length distribution:")
        for length in sorted(self.length_histogram.keys())[:10]:  # First 10
            count = self.length_histogram[length]
            pct = 100.0 * count / self.n_samples if self.n_samples > 0 else 0
            print(f"  L={length:2} {count:6} ({pct:5.1f}%)")
        if len(self.length_histogram) > 10:
            print(f"  ... ({len(self.length_histogram) - 10} more lengths)")
        print()

        print(f"Coverage:")
        print(f"  States visited: {self.state_coverage}")
        print(f"  Edges covered: {self.edge_coverage}")
        print()

        if self.reject_subtypes:
            print("Reject subtypes:")
            for subtype, count in sorted(self.reject_subtypes.items()):
                pct = 100.0 * count / self.n_samples if self.n_samples > 0 else 0
                print(f"  {subtype:12} {count:6} ({pct:5.1f}%)")
            print()


def build_report(
    samples: List[List[int]],
    class_names: List[str],
    quota_summary: Dict,
    reject_subtypes: Counter,
    failed_attempts: int,
) -> DataReport:
    """
    Build a DataReport from generation results.

    Args:
        samples: List of token sequences
        class_names: List of class names per sample
        quota_summary: Dict from QuotaManager.summary()
        reject_subtypes: Counter of reject subtypes
        failed_attempts: Number of failed generation attempts

    Returns:
        DataReport instance
    """
    n_samples = len(samples)

    # Length histogram
    length_histogram = Counter(len(s) for s in samples)

    # Class histogram
    class_histogram = Counter(class_names)

    # Coverage from quota manager
    edge_coverage = quota_summary.get("edges_covered", 0)
    state_coverage = quota_summary.get("states_covered", 0)

    # Retry rate
    total_attempts = n_samples + failed_attempts
    retry_rate = failed_attempts / total_attempts if total_attempts > 0 else 0.0

    return DataReport(
        n_samples=n_samples,
        length_histogram=dict(length_histogram),
        class_histogram=dict(class_histogram),
        edge_coverage=edge_coverage,
        state_coverage=state_coverage,
        reject_subtypes=dict(reject_subtypes),
        failed_attempts=failed_attempts,
        retry_rate=retry_rate,
    )
