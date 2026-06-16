"""
Benford's Law anomaly engine.

Computes chi-square statistic, per-digit Z-scores, and Mean Absolute
Deviation (MAD) for a sequence of transaction amounts. All three metrics
are combined into a composite Benford anomaly score (0–100).

Reference: Benford, F. (1938). The law of anomalous numbers.
"""
from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

# Benford expected probabilities for leading digits 1–9
BENFORD_EXPECTED: dict[int, float] = {
    d: math.log10(1 + 1 / d) for d in range(1, 10)
}

# MAD conformity thresholds (Nigrini, 2012)
_MAD_CONFORMITY = 0.006
_MAD_ACCEPTABLE = 0.012
_MAD_MARGINAL = 0.015
# Above _MAD_MARGINAL → non-conforming


@dataclass
class BenfordResult:
    window_label: str
    n_samples: int

    # Raw metrics
    chi_square: float
    chi_square_p_value: float
    mad: float
    z_scores: dict[int, float] = field(default_factory=dict)

    # Derived flags
    chi_square_flag: bool = False   # p < 0.05
    mad_flag: bool = False          # MAD > 0.015
    z_score_flag: bool = False      # any |z| > 1.96

    # Composite 0–100 anomaly score for this window
    anomaly_score: float = 0.0

    @property
    def any_flag(self) -> bool:
        return self.chi_square_flag or self.mad_flag or self.z_score_flag


class BenfordEngine:
    """Compute Benford's Law metrics over a list of transaction amounts."""

    def __init__(
        self,
        min_samples: int = 30,
        chi_square_alpha: float = 0.05,
        mad_threshold: float = _MAD_MARGINAL,
        z_threshold: float = 1.96,
    ) -> None:
        self.min_samples = min_samples
        self.chi_square_alpha = chi_square_alpha
        self.mad_threshold = mad_threshold
        self.z_threshold = z_threshold

    def analyse(self, amounts: list[float], window_label: str = "custom") -> Optional[BenfordResult]:
        digits = self._extract_digits(amounts)
        if len(digits) < self.min_samples:
            logger.debug("Insufficient samples (%d) for Benford analysis in %s", len(digits), window_label)
            return None

        observed_freq = self._digit_frequencies(digits)
        chi_sq, p_value = self._chi_square(observed_freq, len(digits))
        mad = self._mad(observed_freq)
        z_scores = self._z_scores(observed_freq, len(digits))

        chi_flag = p_value < self.chi_square_alpha
        mad_flag = mad > self.mad_threshold
        z_flag = any(abs(z) > self.z_threshold for z in z_scores.values())

        anomaly_score = self._composite_score(chi_sq, mad, z_scores, p_value)

        return BenfordResult(
            window_label=window_label,
            n_samples=len(digits),
            chi_square=chi_sq,
            chi_square_p_value=p_value,
            mad=mad,
            z_scores=z_scores,
            chi_square_flag=chi_flag,
            mad_flag=mad_flag,
            z_score_flag=z_flag,
            anomaly_score=anomaly_score,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _extract_digits(amounts: list[float]) -> list[int]:
        digits: list[int] = []
        for amount in amounts:
            if amount <= 0:
                continue
            # Strip to leading significant digit
            s = f"{amount:.10e}".split("e")[0].replace(".", "").lstrip("0")
            if s:
                digits.append(int(s[0]))
        return digits

    @staticmethod
    def _digit_frequencies(digits: list[int]) -> dict[int, float]:
        n = len(digits)
        counts = {d: digits.count(d) for d in range(1, 10)}
        return {d: c / n for d, c in counts.items()}

    @staticmethod
    def _chi_square(observed_freq: dict[int, float], n: int) -> tuple[float, float]:
        observed = np.array([observed_freq[d] * n for d in range(1, 10)])
        expected = np.array([BENFORD_EXPECTED[d] * n for d in range(1, 10)])
        chi_sq, p_value = stats.chisquare(observed, expected)
        return float(chi_sq), float(p_value)

    @staticmethod
    def _mad(observed_freq: dict[int, float]) -> float:
        diffs = [abs(observed_freq[d] - BENFORD_EXPECTED[d]) for d in range(1, 10)]
        return float(np.mean(diffs))

    @staticmethod
    def _z_scores(observed_freq: dict[int, float], n: int) -> dict[int, float]:
        z: dict[int, float] = {}
        for d in range(1, 10):
            p_exp = BENFORD_EXPECTED[d]
            std_err = math.sqrt(p_exp * (1 - p_exp) / n)
            if std_err == 0:
                z[d] = 0.0
            else:
                z[d] = (observed_freq[d] - p_exp) / std_err
        return z

    def _composite_score(
        self,
        chi_sq: float,
        mad: float,
        z_scores: dict[int, float],
        p_value: float,
    ) -> float:
        # Chi-square contribution (0–40): maps p_value 0→40, 1→0
        chi_component = max(0.0, 40.0 * (1.0 - p_value))

        # MAD contribution (0–40): linear between 0 and 2× threshold
        mad_component = min(40.0, 40.0 * mad / (2 * self.mad_threshold))

        # Max absolute Z-score contribution (0–20)
        max_z = max(abs(z) for z in z_scores.values()) if z_scores else 0.0
        z_component = min(20.0, 20.0 * max_z / (3 * self.z_threshold))

        return round(chi_component + mad_component + z_component, 2)
