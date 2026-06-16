"""Tests for the Benford's Law anomaly engine."""
from __future__ import annotations

import math
import random
from typing import List

import pytest

from detection.benford_engine import BENFORD_EXPECTED, BenfordEngine, BenfordResult


def _benford_sample(n: int, seed: int = 42) -> List[float]:
    """Generate amounts that follow Benford's Law distribution."""
    rng = random.Random(seed)
    amounts = []
    for _ in range(n):
        d = rng.choices(range(1, 10), weights=[BENFORD_EXPECTED[i] for i in range(1, 10)])[0]
        mantissa = rng.uniform(0, 1)
        amounts.append(d * (10 ** rng.randint(0, 4)) + mantissa)
    return amounts


def _uniform_sample(n: int, seed: int = 99) -> List[float]:
    """Generate amounts with a uniform leading-digit distribution (non-Benford)."""
    rng = random.Random(seed)
    return [rng.choice(range(1, 10)) * 1000.0 + rng.uniform(0, 1) for _ in range(n)]


# ── Engine behaviour ──────────────────────────────────────────────────────────

class TestBenfordEngine:
    def test_returns_none_below_min_samples(self):
        engine = BenfordEngine(min_samples=30)
        result = engine.analyse([1.0, 2.0, 3.0])
        assert result is None

    def test_returns_result_at_threshold(self):
        engine = BenfordEngine(min_samples=30)
        amounts = _benford_sample(30)
        result = engine.analyse(amounts)
        assert result is not None
        assert isinstance(result, BenfordResult)

    def test_benford_conforming_data_low_score(self):
        engine = BenfordEngine()
        amounts = _benford_sample(500)
        result = engine.analyse(amounts)
        assert result is not None
        # Genuinely Benford data should score low
        assert result.anomaly_score < 60

    def test_uniform_data_high_score(self):
        engine = BenfordEngine()
        amounts = _uniform_sample(500)
        result = engine.analyse(amounts)
        assert result is not None
        # Uniform distribution is strongly non-Benford
        assert result.anomaly_score > 30

    def test_anomaly_score_bounded(self):
        engine = BenfordEngine()
        for amounts in [_benford_sample(100), _uniform_sample(100)]:
            result = engine.analyse(amounts)
            assert result is not None
            assert 0.0 <= result.anomaly_score <= 100.0

    def test_z_scores_cover_all_digits(self):
        engine = BenfordEngine()
        result = engine.analyse(_benford_sample(200))
        assert result is not None
        assert set(result.z_scores.keys()) == set(range(1, 10))

    def test_chi_square_p_value_range(self):
        engine = BenfordEngine()
        result = engine.analyse(_benford_sample(200))
        assert result is not None
        assert 0.0 <= result.chi_square_p_value <= 1.0

    def test_mad_is_positive(self):
        engine = BenfordEngine()
        result = engine.analyse(_benford_sample(200))
        assert result is not None
        assert result.mad >= 0.0

    def test_sample_count_recorded(self):
        engine = BenfordEngine()
        amounts = _benford_sample(150)
        result = engine.analyse(amounts)
        assert result is not None
        assert result.n_samples == 150

    def test_window_label_preserved(self):
        engine = BenfordEngine()
        result = engine.analyse(_benford_sample(50), window_label="24h")
        assert result is not None
        assert result.window_label == "24h"


# ── Digit extraction ─────────────────────────────────────────────────────────

class TestDigitExtraction:
    def test_leading_digit_of_1234(self):
        digits = BenfordEngine._extract_digits([1234.56])
        assert digits == [1]

    def test_leading_digit_of_small_decimal(self):
        digits = BenfordEngine._extract_digits([0.00789])
        assert digits == [7]

    def test_zero_amounts_excluded(self):
        digits = BenfordEngine._extract_digits([0.0, 1.5, 2.0])
        assert 0 not in digits
        assert len(digits) == 2

    def test_negative_amounts_excluded(self):
        digits = BenfordEngine._extract_digits([-100.0, 5.0])
        assert len(digits) == 1
        assert digits[0] == 5


# ── Flag logic ────────────────────────────────────────────────────────────────

class TestFlags:
    def test_any_flag_false_when_all_clear(self):
        result = BenfordResult(
            window_label="1h", n_samples=100,
            chi_square=1.0, chi_square_p_value=0.5,
            mad=0.005, z_scores={d: 0.5 for d in range(1, 10)},
            chi_square_flag=False, mad_flag=False, z_score_flag=False,
            anomaly_score=10.0,
        )
        assert not result.any_flag

    def test_any_flag_true_when_one_flag_set(self):
        result = BenfordResult(
            window_label="1h", n_samples=100,
            chi_square=20.0, chi_square_p_value=0.01,
            mad=0.005, z_scores={d: 0.5 for d in range(1, 10)},
            chi_square_flag=True, mad_flag=False, z_score_flag=False,
            anomaly_score=45.0,
        )
        assert result.any_flag
