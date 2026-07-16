"""Training metrics tracking for the VGB framework.

Provides MetricTracker for moving averages of loss, grad_norm, etc.,
and utility functions for common metrics like perplexity.

Validates: Requirements 8.3
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict


class MetricTracker:
    """Tracks moving averages of training metrics.

    Maintains running sums and counts for each metric, enabling
    periodic reporting of averages over training intervals.

    Example:
        tracker = MetricTracker()
        tracker.update({"loss": 2.5, "grad_norm": 1.2})
        tracker.update({"loss": 2.3, "grad_norm": 0.9})
        avgs = tracker.get_averages()
        # avgs == {"loss": 2.4, "grad_norm": 1.05}
        tracker.reset()
    """

    def __init__(self) -> None:
        self._sums: Dict[str, float] = defaultdict(float)
        self._counts: Dict[str, int] = defaultdict(int)

    def update(self, metrics_dict: Dict[str, float]) -> None:
        """Update tracker with a new set of metric values.

        Args:
            metrics_dict: Dictionary mapping metric names to their
                current values (e.g., {"loss": 2.5, "grad_norm": 1.0}).
        """
        for key, value in metrics_dict.items():
            self._sums[key] += value
            self._counts[key] += 1

    def get_averages(self) -> Dict[str, float]:
        """Compute averages for all tracked metrics.

        Returns:
            Dictionary mapping metric names to their average values
            since the last reset. Returns empty dict if no metrics
            have been recorded.
        """
        averages: Dict[str, float] = {}
        for key in self._sums:
            count = self._counts[key]
            if count > 0:
                averages[key] = self._sums[key] / count
            else:
                averages[key] = 0.0
        return averages

    def reset(self) -> None:
        """Reset all tracked metrics to zero."""
        self._sums.clear()
        self._counts.clear()

    @property
    def count(self) -> int:
        """Number of updates recorded (based on first metric seen)."""
        if not self._counts:
            return 0
        return max(self._counts.values())


def compute_perplexity(loss: float) -> float:
    """Compute perplexity from cross-entropy loss.

    Perplexity = exp(loss). Capped at 1e8 to avoid overflow
    with very high losses during early training.

    Args:
        loss: Cross-entropy loss value (natural log base).

    Returns:
        Perplexity value. Returns float('inf') if loss is too large.
    """
    if loss > 18.0:  # exp(18) ≈ 6.5e7, approaching cap
        return float("inf")
    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")
