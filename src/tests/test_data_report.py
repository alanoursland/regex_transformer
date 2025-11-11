"""Test DataReport usage to prevent subscript access bugs."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False

# Import directly from telemetry module, bypassing data package __init__
import importlib.util
spec = importlib.util.spec_from_file_location("telemetry", Path(__file__).parent.parent / "data" / "telemetry.py")
telemetry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(telemetry)
DataReport = telemetry.DataReport


def test_data_report_is_not_subscriptable():
    """DataReport is a dataclass and should be accessed via attributes, not subscripts."""

    # Create a sample DataReport
    report = DataReport(
        n_samples=100,
        length_histogram={1: 20, 2: 30, 3: 50},
        class_histogram={'accept': 40, 'reject': 35, 'incomplete': 25},
        edge_coverage=50,
        state_coverage=20,
        reject_subtypes={'overrun': 10, 'premature': 5},
        failed_attempts=10,
        retry_rate=0.1
    )

    # Verify it's NOT subscriptable (should raise TypeError)
    if HAS_PYTEST:
        with pytest.raises(TypeError, match="not subscriptable"):
            _ = report['class_histogram']

        with pytest.raises(TypeError, match="not subscriptable"):
            _ = report['n_samples']
    else:
        # Manual version without pytest
        try:
            _ = report['class_histogram']
            raise AssertionError("Should have raised TypeError for subscript access")
        except TypeError as e:
            assert "not subscriptable" in str(e)

        try:
            _ = report['n_samples']
            raise AssertionError("Should have raised TypeError for subscript access")
        except TypeError as e:
            assert "not subscriptable" in str(e)


def test_data_report_attribute_access():
    """DataReport should be accessed using attribute notation."""

    report = DataReport(
        n_samples=100,
        length_histogram={1: 20, 2: 30, 3: 50},
        class_histogram={'accept': 40, 'reject': 35, 'incomplete': 25},
        edge_coverage=50,
        state_coverage=20,
        reject_subtypes={'overrun': 10, 'premature': 5},
        failed_attempts=10,
        retry_rate=0.1
    )

    # Correct access via attributes
    assert report.n_samples == 100
    assert report.class_histogram == {'accept': 40, 'reject': 35, 'incomplete': 25}
    assert report.length_histogram == {1: 20, 2: 30, 3: 50}
    assert report.edge_coverage == 50
    assert report.state_coverage == 20
    assert report.reject_subtypes == {'overrun': 10, 'premature': 5}
    assert report.failed_attempts == 10
    assert report.retry_rate == 0.1


def test_data_report_has_class_histogram_not_class_distribution():
    """DataReport has 'class_histogram' attribute, not 'class_distribution'."""

    report = DataReport(
        n_samples=100,
        length_histogram={1: 20, 2: 30, 3: 50},
        class_histogram={'accept': 40, 'reject': 35, 'incomplete': 25},
        edge_coverage=50,
        state_coverage=20,
        reject_subtypes={},
        failed_attempts=10,
        retry_rate=0.1
    )

    # Correct attribute name
    assert hasattr(report, 'class_histogram')
    assert report.class_histogram == {'accept': 40, 'reject': 35, 'incomplete': 25}

    # Wrong attribute name should not exist
    assert not hasattr(report, 'class_distribution')

    # Attempting to access non-existent attribute should raise AttributeError
    if HAS_PYTEST:
        with pytest.raises(AttributeError):
            _ = report.class_distribution
    else:
        try:
            _ = report.class_distribution
            raise AssertionError("Should have raised AttributeError")
        except AttributeError:
            pass  # Expected


def test_data_report_dict_conversion():
    """Converting class_histogram to dict should work (it's already a dict)."""

    report = DataReport(
        n_samples=100,
        length_histogram={1: 20, 2: 30, 3: 50},
        class_histogram={'accept': 40, 'reject': 35, 'incomplete': 25},
        edge_coverage=50,
        state_coverage=20,
        reject_subtypes={},
        failed_attempts=10,
        retry_rate=0.1
    )

    # class_histogram is already a dict, so dict() is redundant but harmless
    class_dist = dict(report.class_histogram)
    assert class_dist == {'accept': 40, 'reject': 35, 'incomplete': 25}
    assert isinstance(class_dist, dict)


def test_train_model_usage_pattern():
    """Test the exact usage pattern from train_model.py."""

    # Simulate what generate_corpus returns
    report = DataReport(
        n_samples=2000,
        length_histogram={1: 200, 2: 300, 3: 500},
        class_histogram={'accept': 700, 'reject': 650, 'incomplete': 650},
        edge_coverage=100,
        state_coverage=21,
        reject_subtypes={'overrun': 200, 'premature': 150},
        failed_attempts=100,
        retry_rate=0.05
    )

    # This is what the buggy code tried to do:
    # print(f"  Class distribution: {dict(report.class_histogram)}")
    # Should raise TypeError

    if HAS_PYTEST:
        with pytest.raises(TypeError):
            _ = report['class_histogram']  # Wrong: subscript access
    else:
        try:
            _ = report['class_histogram']  # Wrong: subscript access
            raise AssertionError("Should have raised TypeError")
        except TypeError:
            pass  # Expected

    # Correct way:
    class_dist = dict(report.class_histogram)
    assert 'accept' in class_dist
    assert 'reject' in class_dist
    assert 'incomplete' in class_dist

    # This is what should be printed
    output = f"  Class distribution: {dict(report.class_histogram)}"
    assert 'accept' in output
    assert '700' in output


if __name__ == "__main__":
    # Run tests
    test_data_report_is_not_subscriptable()
    print("✓ test_data_report_is_not_subscriptable passed")

    test_data_report_attribute_access()
    print("✓ test_data_report_attribute_access passed")

    test_data_report_has_class_histogram_not_class_distribution()
    print("✓ test_data_report_has_class_histogram_not_class_distribution passed")

    test_data_report_dict_conversion()
    print("✓ test_data_report_dict_conversion passed")

    test_train_model_usage_pattern()
    print("✓ test_train_model_usage_pattern passed")

    print("\n✅ All tests passed!")
