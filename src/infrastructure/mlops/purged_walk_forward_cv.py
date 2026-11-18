from __future__ import annotations

import contextlib
import itertools
import math
import threading
from dataclasses import dataclass
from typing import Any, Generator, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from scipy.stats import kurtosis as scipy_kurtosis
from scipy.stats import norm
from scipy.stats import skew as scipy_skew
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import BaseCrossValidator
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset


ArrayLike = Union[np.ndarray, pd.DataFrame, pd.Series, Sequence[float]]
SamplesInfoSets = Union[pd.Series, Sequence[Any], np.ndarray]


@dataclass(frozen=True)
class PurgedSplit:
    """Metadata describing one purged cross-validation split."""

    train_index: np.ndarray
    test_index: np.ndarray
    test_blocks: Tuple[int, ...]
    combination_id: int


@dataclass(frozen=True)
class DeflatedSharpeResult:
    """Computed probabilistic and deflated Sharpe ratio statistics."""

    sharpe_ratio: float
    annualized_sharpe_ratio: Optional[float]
    probabilistic_sharpe_ratio: float
    deflated_sharpe_ratio: float
    benchmark_sharpe: float
    expected_max_sharpe: float
    skewness: float
    kurtosis: float
    n_observations: int
    n_trials: int
    sharpe_trial_mean: float
    sharpe_trial_variance: float


class _PurgedCrossValidationBase(BaseCrossValidator):
    """Shared implementation for purged time-series cross-validation splitters."""

    def __init__(
        self,
        n_splits: int = 5,
        samples_info_sets: Optional[SamplesInfoSets] = None,
        pct_embargo: float = 0.0,
    ) -> None:
        if not isinstance(n_splits, int) or n_splits < 2:
            raise ValueError("n_splits must be an integer greater than or equal to 2.")
        if not isinstance(pct_embargo, (int, float)) or not 0.0 <= pct_embargo < 1.0:
            raise ValueError("pct_embargo must be a float in the interval [0.0, 1.0).")

        self.n_splits = n_splits
        self.pct_embargo = float(pct_embargo)

        if isinstance(samples_info_sets, pd.Series):
            self.samples_info_sets: Optional[SamplesInfoSets] = samples_info_sets.copy(
                deep=True
            )
        elif samples_info_sets is None:
            self.samples_info_sets = None
        else:
            self.samples_info_sets = tuple(samples_info_sets)

    @staticmethod
    def _n_samples(X: Any) -> int:
        try:
            n_samples = len(X)
        except TypeError as exc:
            raise ValueError("X must be a sized array-like object.") from exc

        if n_samples < 2:
            raise ValueError("At least two observations are required.")
        return int(n_samples)

    def _resolve_label_spans(
        self,
        X: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resolve each sample's label interval as inclusive [start_time, end_time].

        When samples_info_sets is omitted, each sample is treated as a point-in-time
        label. This remains usable for embargo-only validation, but callers should
        provide samples_info_sets whenever labels span multiple timestamps.
        """
        n_samples = self._n_samples(X)

        if self.samples_info_sets is None:
            positions = np.arange(n_samples, dtype=np.int64)
            return positions, positions

        if isinstance(self.samples_info_sets, pd.Series):
            info_sets = self.samples_info_sets
            if len(info_sets) != n_samples:
                raise ValueError(
                    "samples_info_sets must contain exactly one label end time per row in X."
                )

            x_index = getattr(X, "index", None)
            if isinstance(x_index, pd.Index) and info_sets.index.is_unique:
                aligned = info_sets.reindex(x_index)
                if len(aligned) == n_samples and not aligned.isna().any():
                    starts = np.asarray(x_index)
                    ends = np.asarray(aligned.to_numpy())
                else:
                    starts = np.asarray(info_sets.index)
                    ends = np.asarray(info_sets.to_numpy())
            else:
                starts = np.asarray(info_sets.index)
                ends = np.asarray(info_sets.to_numpy())
        else:
            ends = np.asarray(self.samples_info_sets)
            if len(ends) != n_samples:
                raise ValueError(
                    "samples_info_sets must contain exactly one label end time per row in X."
                )

            x_index = getattr(X, "index", None)
            starts = (
                np.asarray(x_index)
                if isinstance(x_index, pd.Index) and len(x_index) == n_samples
                else np.arange(n_samples, dtype=np.int64)
            )

        if pd.isna(starts).any() or pd.isna(ends).any():
            raise ValueError("Label start and end times cannot contain missing values.")

        try:
            monotonic_starts = pd.Index(starts).is_monotonic_increasing
            valid_intervals = np.all(ends >= starts)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Label start and end times must be comparable and chronologically ordered."
            ) from exc

        if not monotonic_starts:
            raise ValueError(
                "X must be ordered chronologically. Label start times are not monotonic."
            )
        if not valid_intervals:
            raise ValueError("Each label end time must be greater than or equal to its start.")

        return starts, ends

    def _make_blocks(
        self,
        X: Any,
        groups: Optional[ArrayLike],
    ) -> Tuple[np.ndarray, ...]:
        """Create contiguous sequential sample blocks."""
        n_samples = self._n_samples(X)

        if groups is None:
            if self.n_splits > n_samples:
                raise ValueError("n_splits cannot exceed the number of observations.")
            blocks = tuple(
                block.astype(np.int64, copy=False)
                for block in np.array_split(np.arange(n_samples), self.n_splits)
            )
            return blocks

        group_array = np.asarray(groups)
        if group_array.ndim != 1 or len(group_array) != n_samples:
            raise ValueError("groups must be a one-dimensional array with len(groups) == len(X).")
        if pd.isna(group_array).any():
            raise ValueError("groups cannot contain missing values.")

        codes, _ = pd.factorize(group_array, sort=False)
        if np.any(codes[1:] < codes[:-1]):
            raise ValueError(
                "groups must represent contiguous sequential time blocks. "
                "A group cannot reappear after another group has started."
            )

        n_groups = int(codes.max()) + 1
        if n_groups < self.n_splits:
            raise ValueError(
                "The number of contiguous groups must be greater than or equal to n_splits."
            )

        group_blocks = tuple(
            np.flatnonzero(codes == group_code).astype(np.int64, copy=False)
            for group_code in range(n_groups)
        )

        if n_groups == self.n_splits:
            return group_blocks

        grouped_block_ids = np.array_split(np.arange(n_groups), self.n_splits)
        return tuple(
            np.concatenate([group_blocks[group_id] for group_id in block_ids]).astype(
                np.int64, copy=False
            )
            for block_ids in grouped_block_ids
        )

    def _embargo_count(self, n_samples: int) -> int:
        """Return the number of observations removed after each test segment."""
        return int(math.ceil(self.pct_embargo * n_samples))

    def _purged_indices(
        self,
        starts: np.ndarray,
        ends: np.ndarray,
        test_index: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute train and test indices after exact interval-overlap purging and embargo.
        """
        n_samples = len(starts)
        test_mask = np.zeros(n_samples, dtype=bool)
        test_mask[test_index] = True

        candidate_train = ~test_mask
        candidate_index = np.flatnonzero(candidate_train)
        overlap_mask = np.zeros(n_samples, dtype=bool)

        for test_position in test_index:
            overlaps = (
                (starts[candidate_index] <= ends[test_position])
                & (ends[candidate_index] >= starts[test_position])
            )
            overlap_mask[candidate_index[overlaps]] = True

        embargo_mask = np.zeros(n_samples, dtype=bool)
        embargo_size = self._embargo_count(n_samples)

        if embargo_size > 0 and len(test_index) > 0:
            ordered_test = np.sort(test_index)
            segment_starts = np.r_[
                0, np.flatnonzero(np.diff(ordered_test) > 1) + 1
            ]
            segment_ends = np.r_[
                segment_starts[1:] - 1, len(ordered_test) - 1
            ]

            for segment_end in segment_ends:
                embargo_start = int(ordered_test[segment_end]) + 1
                embargo_stop = min(n_samples, embargo_start + embargo_size)
                embargo_mask[embargo_start:embargo_stop] = True

        train_mask = ~(test_mask | overlap_mask | embargo_mask)
        train_index = np.flatnonzero(train_mask).astype(np.int64, copy=False)
        clean_test_index = np.flatnonzero(test_mask).astype(np.int64, copy=False)

        if len(train_index) == 0:
            raise ValueError(
                "Purging and embargoing removed all training observations. "
                "Reduce n_splits, pct_embargo, or label horizon."
            )

        return train_index, clean_test_index


class PurgedGroupTimeSeriesSplit(_PurgedCrossValidationBase):
    """
    Time-series cross-validator with label-overlap purging and post-test embargoing.

    Labels are represented by inclusive intervals [label_start, label_end]. Training
    labels intersecting any test-label interval are removed. Embargoing removes a
    configurable fraction of observations immediately after each test segment.

    Parameters
    ----------
    n_splits:
        Number of sequential test folds.
    samples_info_sets:
        Series indexed by label start time with values equal to label end time, or a
        sequence of label end times ordered identically to X.
    pct_embargo:
        Fraction of the complete sample removed after every test segment.
    """

    def split(
        self,
        X: Any,
        y: Optional[ArrayLike] = None,
        groups: Optional[ArrayLike] = None,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Yield purged and embargoed train/test index pairs."""
        del y
        starts, ends = self._resolve_label_spans(X)
        blocks = self._make_blocks(X, groups)

        for test_block in blocks:
            train_index, test_index = self._purged_indices(starts, ends, test_block)
            yield train_index, test_index

    def iter_splits(
        self,
        X: Any,
        y: Optional[ArrayLike] = None,
        groups: Optional[ArrayLike] = None,
    ) -> Generator[PurgedSplit, None, None]:
        """Yield split metadata suitable for logging and backtest orchestration."""
        del y
        starts, ends = self._resolve_label_spans(X)
        blocks = self._make_blocks(X, groups)

        for split_id, test_block in enumerate(blocks):
            train_index, test_index = self._purged_indices(starts, ends, test_block)
            yield PurgedSplit(
                train_index=train_index,
                test_index=test_index,
                test_blocks=(split_id,),
                combination_id=split_id,
            )

    def get_n_splits(
        self,
        X: Optional[Any] = None,
        y: Optional[ArrayLike] = None,
        groups: Optional[ArrayLike] = None,
    ) -> int:
        """Return the configured number of sequential folds."""
        del X, y, groups
        return self.n_splits


class CombinatorialPurgedCV(_PurgedCrossValidationBase):
    """
    Combinatorial Purged Cross-Validation (CPCV) from López de Prado.

    The dataset is partitioned into N sequential blocks and every combination of k
    test blocks is evaluated. This produces C(N, k) purged and embargoed splits and
    supports C(N - 1, k - 1) distinct backtest paths.

    Parameters
    ----------
    n_splits:
        Number N of sequential time blocks.
    n_test_splits:
        Number k of blocks included in the test set for each combination.
    samples_info_sets:
        Series indexed by label start time with values equal to label end time, or a
        sequence of label end times ordered identically to X.
    pct_embargo:
        Fraction of the complete sample removed immediately after each test segment.
    """

    def __init__(
        self,
        n_splits: int = 5,
        n_test_splits: int = 2,
        samples_info_sets: Optional[SamplesInfoSets] = None,
        pct_embargo: float = 0.0,
    ) -> None:
        super().__init__(
            n_splits=n_splits,
            samples_info_sets=samples_info_sets,
            pct_embargo=pct_embargo,
        )

        if not isinstance(n_test_splits, int) or not 1 <= n_test_splits < n_splits:
            raise ValueError(
                "n_test_splits must be an integer in the interval [1, n_splits - 1]."
            )

        self.n_test_splits = n_test_splits

    @property
    def n_backtest_paths(self) -> int:
        """Return the number of distinct CPCV backtest paths."""
        return math.comb(self.n_splits - 1, self.n_test_splits - 1)

    @property
    def test_block_combinations(self) -> Tuple[Tuple[int, ...], ...]:
        """Return all C(N, k) test-block combinations."""
        return tuple(itertools.combinations(range(self.n_splits), self.n_test_splits))

    def split(
        self,
        X: Any,
        y: Optional[ArrayLike] = None,
        groups: Optional[ArrayLike] = None,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Yield all combinatorial purged and embargoed train/test index pairs."""
        for split_metadata in self.iter_splits(X=X, y=y, groups=groups):
            yield split_metadata.train_index, split_metadata.test_index

    def iter_splits(
        self,
        X: Any,
        y: Optional[ArrayLike] = None,
        groups: Optional[ArrayLike] = None,
    ) -> Generator[PurgedSplit, None, None]:
        """Yield all CPCV split metadata in deterministic combinatorial order."""
        del y
        starts, ends = self._resolve_label_spans(X)
        blocks = self._make_blocks(X, groups)

        if len(blocks) != self.n_splits:
            raise RuntimeError("Internal block construction did not produce n_splits blocks.")

        for combination_id, test_blocks in enumerate(
            itertools.combinations(range(self.n_splits), self.n_test_splits)
        ):
            test_index = np.concatenate([blocks[block_id] for block_id in test_blocks])
            test_index.sort()
            train_index, clean_test_index = self._purged_indices(
                starts=starts,
                ends=ends,
                test_index=test_index,
            )

            yield PurgedSplit(
                train_index=train_index,
                test_index=clean_test_index,
                test_blocks=tuple(test_blocks),
                combination_id=combination_id,
            )

    def get_n_splits(
        self,
        X: Optional[Any] = None,
        y: Optional[ArrayLike] = None,
        groups: Optional[ArrayLike] = None,
    ) -> int:
        """Return the total number C(N, k) of combinatorial splits."""
        del X, y, groups
        return math.comb(self.n_splits, self.n_test_splits)


class DeflatedSharpeCalculator:
    """
    Calculate Probabilistic Sharpe Ratio (PSR) and Deflated Sharpe Ratio (DSR).

    The PSR incorporates non-normality through sample skewness and Pearson kurtosis.
    The DSR uses the expected maximum Sharpe ratio across multiple trials, estimated
    from the mean and variance of candidate Sharpe ratios.

    Sharpe ratios passed through trial_sharpes must use the same return frequency as
    the evaluated returns. annualization_factor affects reporting only.
    """

    def __init__(self, annualization_factor: Optional[float] = None) -> None:
        if annualization_factor is not None and annualization_factor <= 0:
            raise ValueError("annualization_factor must be positive when supplied.")
        self.annualization_factor = annualization_factor

    @staticmethod
    def sharpe_ratio(returns: ArrayLike) -> float:
        """Compute the arithmetic periodic Sharpe ratio from excess returns."""
        values = np.asarray(returns, dtype=np.float64).reshape(-1)
        if len(values) < 2:
            raise ValueError("At least two return observations are required.")
        if not np.isfinite(values).all():
            raise ValueError("returns must contain only finite values.")

        volatility = float(np.std(values, ddof=1))
        if volatility <= np.finfo(np.float64).eps:
            raise ValueError("Sharpe ratio is undefined for zero-variance returns.")

        return float(np.mean(values) / volatility)

    @staticmethod
    def return_moments(returns: ArrayLike) -> Tuple[float, float]:
        """Return bias-corrected sample skewness and Pearson kurtosis."""
        values = np.asarray(returns, dtype=np.float64).reshape(-1)
        if len(values) < 2:
            raise ValueError("At least two return observations are required.")
        if not np.isfinite(values).all():
            raise ValueError("returns must contain only finite values.")

        if np.std(values, ddof=1) <= np.finfo(np.float64).eps:
            raise ValueError("Moments are undefined for zero-variance returns.")

        skewness = float(scipy_skew(values, bias=False)) if len(values) >= 3 else 0.0
        kurtosis = (
            float(scipy_kurtosis(values, fisher=False, bias=False))
            if len(values) >= 4
            else 3.0
        )

        if not np.isfinite(skewness):
            skewness = 0.0
        if not np.isfinite(kurtosis):
            kurtosis = 3.0

        return skewness, kurtosis

    @staticmethod
    def probabilistic_sharpe_ratio(
        observed_sharpe: float,
        benchmark_sharpe: float,
        n_observations: int,
        skewness: float,
        kurtosis: float,
    ) -> float:
        """
        Calculate PSR: P(true Sharpe ratio > benchmark Sharpe ratio).

        Kurtosis must be Pearson kurtosis, where a normal distribution has kurtosis 3.
        """
        if n_observations < 2:
            raise ValueError("n_observations must be at least 2.")

        denominator_squared = (
            1.0
            - skewness * observed_sharpe
            + ((kurtosis - 1.0) / 4.0) * observed_sharpe * observed_sharpe
        )

        if denominator_squared <= 0.0 or not np.isfinite(denominator_squared):
            raise ValueError(
                "The PSR denominator is non-positive. Check Sharpe ratio and moments."
            )

        z_score = (
            (observed_sharpe - benchmark_sharpe)
            * math.sqrt(n_observations - 1)
            / math.sqrt(denominator_squared)
        )
        return float(norm.cdf(z_score))

    @staticmethod
    def expected_maximum_sharpe(
        mean_sharpe: float,
        sharpe_variance: float,
        n_trials: int,
    ) -> float:
        """
        Estimate E[max(SR)] using the López de Prado extreme-value approximation.
        """
        if n_trials < 1:
            raise ValueError("n_trials must be at least 1.")
        if sharpe_variance < 0.0 or not np.isfinite(sharpe_variance):
            raise ValueError("sharpe_variance must be finite and non-negative.")
        if not np.isfinite(mean_sharpe):
            raise ValueError("mean_sharpe must be finite.")

        if n_trials == 1 or sharpe_variance == 0.0:
            return float(mean_sharpe)

        standard_deviation = math.sqrt(sharpe_variance)
        first_quantile = norm.ppf(1.0 - 1.0 / n_trials)
        second_quantile = norm.ppf(1.0 - 1.0 / (n_trials * math.e))
        extreme_value_adjustment = (
            (1.0 - np.euler_gamma) * first_quantile
            + np.euler_gamma * second_quantile
        )

        return float(mean_sharpe + standard_deviation * extreme_value_adjustment)

    def calculate(
        self,
        returns: ArrayLike,
        trial_sharpes: Optional[ArrayLike] = None,
        benchmark_sharpe: float = 0.0,
        annualization_factor: Optional[float] = None,
    ) -> DeflatedSharpeResult:
        """
        Calculate PSR and DSR from a realized excess-return series.

        Parameters
        ----------
        returns:
            Periodic excess returns generated by the selected strategy or model.
        trial_sharpes:
            Sharpe ratios from all tested candidate strategies, model configurations,
            feature sets, or hyperparameter trials. Passing this collection enables
            multiple-testing deflation.
        benchmark_sharpe:
            Minimum Sharpe ratio used for the ordinary PSR calculation.
        annualization_factor:
            Optional reporting annualization factor. If omitted, the constructor value
            is used. It does not alter PSR or DSR mathematics.
        """
        values = np.asarray(returns, dtype=np.float64).reshape(-1)
        observed_sharpe = self.sharpe_ratio(values)
        skewness, kurtosis = self.return_moments(values)

        if not np.isfinite(benchmark_sharpe):
            raise ValueError("benchmark_sharpe must be finite.")

        psr = self.probabilistic_sharpe_ratio(
            observed_sharpe=observed_sharpe,
            benchmark_sharpe=float(benchmark_sharpe),
            n_observations=len(values),
            skewness=skewness,
            kurtosis=kurtosis,
        )

        if trial_sharpes is None:
            trial_values = np.asarray([benchmark_sharpe], dtype=np.float64)
            expected_max_sharpe = float(benchmark_sharpe)
            trial_variance = 0.0
        else:
            trial_values = np.asarray(trial_sharpes, dtype=np.float64).reshape(-1)
            if len(trial_values) == 0:
                raise ValueError("trial_sharpes cannot be empty.")
            if not np.isfinite(trial_values).all():
                raise ValueError("trial_sharpes must contain only finite values.")

            trial_variance = (
                float(np.var(trial_values, ddof=1)) if len(trial_values) > 1 else 0.0
            )
            expected_max_sharpe = self.expected_maximum_sharpe(
                mean_sharpe=float(np.mean(trial_values)),
                sharpe_variance=trial_variance,
                n_trials=len(trial_values),
            )

        dsr = self.probabilistic_sharpe_ratio(
            observed_sharpe=observed_sharpe,
            benchmark_sharpe=expected_max_sharpe,
            n_observations=len(values),
            skewness=skewness,
            kurtosis=kurtosis,
        )

        resolved_annualization = (
            annualization_factor
            if annualization_factor is not None
            else self.annualization_factor
        )
        if resolved_annualization is not None and resolved_annualization <= 0:
            raise ValueError("annualization_factor must be positive when supplied.")

        annualized_sharpe = (
            float(observed_sharpe * math.sqrt(resolved_annualization))
            if resolved_annualization is not None
            else None
        )

        return DeflatedSharpeResult(
            sharpe_ratio=observed_sharpe,
            annualized_sharpe_ratio=annualized_sharpe,
            probabilistic_sharpe_ratio=psr,
            deflated_sharpe_ratio=dsr,
            benchmark_sharpe=float(benchmark_sharpe),
            expected_max_sharpe=expected_max_sharpe,
            skewness=skewness,
            kurtosis=kurtosis,
            n_observations=len(values),
            n_trials=len(trial_values),
            sharpe_trial_mean=float(np.mean(trial_values)),
            sharpe_trial_variance=trial_variance,
        )


class AttentionLSTMNetwork(nn.Module):
    """LSTM encoder followed by multi-head self-attention for sequence regression."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads.")

        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=recurrent_dropout,
            batch_first=True,
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.normalization = nn.LayerNorm(hidden_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """Produce one regression prediction for every input sequence."""
        encoded, _ = self.lstm(inputs)
        attended, _ = self.attention(
            encoded,
            encoded,
            encoded,
            need_weights=False,
        )
        pooled = self.normalization(encoded + attended).mean(dim=1)
        return self.head(pooled).squeeze(-1)


class AttentionLSTMRegressor(RegressorMixin, BaseEstimator):
    """
    Scikit-learn compatible PyTorch Attention-LSTM regressor.

    Input X must have shape (n_samples, sequence_length, n_features). The estimator
    can be used directly in purged sklearn-style cross-validation loops.

    Parameters
    ----------
    input_size:
        Number of features per time step. If None, inferred during fit.
    hidden_size:
        LSTM hidden dimension.
    num_layers:
        Number of stacked LSTM layers.
    num_heads:
        Attention heads. hidden_size must be divisible by this value.
    dropout:
        Dropout probability in the attention head.
    learning_rate:
        AdamW learning rate.
    epochs:
        Number of optimization epochs.
    batch_size:
        Mini-batch size.
    weight_decay:
        AdamW weight decay.
    random_state:
        Optional deterministic training seed.
    device:
        "auto", "cpu", "cuda", or a valid torch device string.
    verbose:
        If True, emits epoch loss information.
    """

    def __init__(
        self,
        input_size: Optional[int] = None,
        hidden_size: int = 64,
        num_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        epochs: int = 20,
        batch_size: int = 64,
        weight_decay: float = 1e-4,
        random_state: Optional[int] = None,
        device: Optional[str] = "auto",
        verbose: bool = False,
    ) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.device = device
        self.verbose = verbose
        self._lock = threading.RLock()

    @staticmethod
    def _validate_features(X: Any) -> np.ndarray:
        """Validate and convert sequence features to float32."""
        values = np.asarray(X, dtype=np.float32)
        if values.ndim != 3:
            raise ValueError(
                "AttentionLSTMRegressor expects X with shape "
                "(n_samples, sequence_length, n_features)."
            )
        if values.shape[0] < 2 or values.shape[1] < 1 or values.shape[2] < 1:
            raise ValueError("X must contain at least two non-empty sequences.")
        if not np.isfinite(values).all():
            raise ValueError("X must contain only finite values.")
        return values

    @staticmethod
    def _validate_target(y: Any, n_samples: int) -> np.ndarray:
        """Validate and convert regression targets to float32."""
        values = np.asarray(y, dtype=np.float32).reshape(-1)
        if len(values) != n_samples:
            raise ValueError("y must contain exactly one target for every sequence in X.")
        if not np.isfinite(values).all():
            raise ValueError("y must contain only finite values.")
        return values

    def _resolve_device(self) -> torch.device:
        """Resolve the configured PyTorch execution device."""
        if self.device is None or self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

        resolved = torch.device(self.device)
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("A CUDA device was requested but CUDA is unavailable.")
        return resolved

    def _validate_hyperparameters(self) -> None:
        """Validate model and optimizer parameters before fitting."""
        if self.input_size is not None and self.input_size < 1:
            raise ValueError("input_size must be positive when supplied.")
        if self.hidden_size < 1 or self.num_layers < 1 or self.num_heads < 1:
            raise ValueError("hidden_size, num_layers, and num_heads must be positive.")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in the interval [0.0, 1.0).")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative.")
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive integers.")

    def fit(
        self,
        X: Any,
        y: Any,
        sample_weight: Optional[ArrayLike] = None,
    ) -> "AttentionLSTMRegressor":
        """Fit the Attention-LSTM regression model."""
        with self._lock:
            self._validate_hyperparameters()
            features = self._validate_features(X)
            targets = self._validate_target(y, features.shape[0])

            inferred_input_size = int(features.shape[2])
            if self.input_size is not None and self.input_size != inferred_input_size:
                raise ValueError(
                    f"input_size={self.input_size} does not match X.shape[2]={inferred_input_size}."
                )

            if sample_weight is None:
                weights = np.ones(features.shape[0], dtype=np.float32)
            else:
                weights = np.asarray(sample_weight, dtype=np.float32).reshape(-1)
                if len(weights) != features.shape[0]:
                    raise ValueError("sample_weight must have the same length as X.")
                if not np.isfinite(weights).all() or np.any(weights < 0.0):
                    raise ValueError(
                        "sample_weight must contain finite non-negative values."
                    )
                if float(weights.sum()) <= 0.0:
                    raise ValueError("sample_weight must have a strictly positive sum.")

            resolved_device = self._resolve_device()
            cuda_devices: Iterable[int] = ()
            if resolved_device.type == "cuda":
                cuda_devices = (resolved_device.index or 0,)

            with torch.random.fork_rng(devices=list(cuda_devices), enabled=True):
                if self.random_state is not None:
                    torch.manual_seed(self.random_state)
                    if resolved_device.type == "cuda":
                        torch.cuda.manual_seed_all(self.random_state)

                self.model_ = AttentionLSTMNetwork(
                    input_size=inferred_input_size,
                    hidden_size=self.hidden_size,
                    num_layers=self.num_layers,
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                ).to(resolved_device)

                optimizer = torch.optim.AdamW(
                    self.model_.parameters(),
                    lr=self.learning_rate,
                    weight_decay=self.weight_decay,
                )

                dataset = TensorDataset(
                    torch.from_numpy(features),
                    torch.from_numpy(targets),
                    torch.from_numpy(weights),
                )

                generator = None
                if self.random_state is not None:
                    generator = torch.Generator()
                    generator.manual_seed(self.random_state)

                loader = DataLoader(
                    dataset,
                    batch_size=min(self.batch_size, len(dataset)),
                    shuffle=True,
                    num_workers=0,
                    pin_memory=resolved_device.type == "cuda",
                    generator=generator,
                )

                self.training_loss_ = []
                self.model_.train()

                for epoch in range(self.epochs):
                    epoch_weighted_loss = 0.0
                    epoch_weight = 0.0

                    for batch_features, batch_targets, batch_weights in loader:
                        batch_features = batch_features.to(
                            resolved_device,
                            non_blocking=True,
                        )
                        batch_targets = batch_targets.to(
                            resolved_device,
                            non_blocking=True,
                        )
                        batch_weights = batch_weights.to(
                            resolved_device,
                            non_blocking=True,
                        )

                        optimizer.zero_grad(set_to_none=True)
                        predictions = self.model_(batch_features)
                        squared_error = (predictions - batch_targets).pow(2)
                        loss = (squared_error * batch_weights).sum() / batch_weights.sum()

                        if not torch.isfinite(loss):
                            raise FloatingPointError(
                                "Non-finite training loss encountered in AttentionLSTMRegressor."
                            )

                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model_.parameters(), max_norm=5.0)
                        optimizer.step()

                        batch_weight_sum = float(batch_weights.detach().sum().cpu())
                        epoch_weighted_loss += float(loss.detach().cpu()) * batch_weight_sum
                        epoch_weight += batch_weight_sum

                    mean_loss = epoch_weighted_loss / max(epoch_weight, np.finfo(float).eps)
                    self.training_loss_.append(mean_loss)

                    if self.verbose:
                        print(
                            f"AttentionLSTMRegressor epoch={epoch + 1}/{self.epochs} "
                            f"loss={mean_loss:.8f}"
                        )

            self.n_features_in_ = inferred_input_size
            self.sequence_length_ = int(features.shape[1])
            self.device_ = str(resolved_device)
            self.is_fitted_ = True
            return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict regression targets for a batch of input sequences."""
        with self._lock:
            if not getattr(self, "is_fitted_", False):
                raise RuntimeError("AttentionLSTMRegressor must be fitted before prediction.")

            features = self._validate_features(X)
            if features.shape[2] != self.n_features_in_:
                raise ValueError(
                    "X feature count differs from the feature count observed during fit."
                )
            if features.shape[1] != self.sequence_length_:
                raise ValueError(
                    "X sequence length differs from the sequence length observed during fit."
                )

            model_device = torch.device(self.device_)
            self.model_.to(model_device)
            self.model_.eval()

            with torch.no_grad():
                tensor_features = torch.from_numpy(features).to(model_device)
                predictions = self.model_(tensor_features).detach().cpu().numpy()

            return predictions.astype(np.float64, copy=False)

    def __getstate__(self) -> dict[str, Any]:
        """Exclude the non-serializable re-entrant lock from pickled estimator state."""
        state = self.__dict__.copy()
        state.pop("_lock", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore estimator state and recreate its synchronization primitive."""
        self.__dict__.update(state)
        self._lock = threading.RLock()


def _build_demo_sequences(features: np.ndarray, lookback: int) -> np.ndarray:
    """Construct fixed-length rolling sequences for the executable demonstration."""
    sequences = np.empty((len(features), lookback, features.shape[1]), dtype=np.float32)
    for index in range(len(features)):
        start = max(0, index - lookback + 1)
        window = features[start : index + 1]
        if len(window) < lookback:
            padding = np.repeat(window[:1], lookback - len(window), axis=0)
            window = np.vstack((padding, window))
        sequences[index] = window
    return sequences


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    n_samples = 240
    dates = pd.date_range("2020-01-01", periods=n_samples, freq="B")

    raw_features = rng.normal(size=(n_samples, 4))
    target = (
        0.04 * raw_features[:, 0]
        - 0.03 * raw_features[:, 1]
        + rng.normal(scale=0.05, size=n_samples)
    )

    features = pd.DataFrame(
        raw_features,
        index=dates,
        columns=["momentum", "value", "quality", "volatility"],
    )
    label_end_times = pd.Series(
        dates[np.minimum(np.arange(n_samples) + 5, n_samples - 1)],
        index=dates,
    )

    cv = CombinatorialPurgedCV(
        n_splits=6,
        n_test_splits=2,
        samples_info_sets=label_end_times,
        pct_embargo=0.01,
    )

    trial_sharpes = []
    first_train_index: Optional[np.ndarray] = None
    first_test_index: Optional[np.ndarray] = None

    for train_index, test_index in cv.split(features):
        model = HistGradientBoostingRegressor(
            max_iter=100,
            learning_rate=0.05,
            max_leaf_nodes=15,
            random_state=7,
        )
        model.fit(features.iloc[train_index], target[train_index])
        predictions = model.predict(features.iloc[test_index])
        strategy_returns = np.sign(predictions) * target[test_index]
        trial_sharpes.append(DeflatedSharpeCalculator.sharpe_ratio(strategy_returns))

        if first_train_index is None:
            first_train_index = train_index
            first_test_index = test_index

    dsr_result = DeflatedSharpeCalculator(annualization_factor=252).calculate(
        returns=np.sign(
            HistGradientBoostingRegressor(random_state=7)
            .fit(features.iloc[first_train_index], target[first_train_index])
            .predict(features.iloc[first_test_index])
        )
        * target[first_test_index],
        trial_sharpes=trial_sharpes,
    )

    sequence_features = _build_demo_sequences(raw_features, lookback=10)
    attention_model = AttentionLSTMRegressor(
        hidden_size=16,
        num_heads=4,
        epochs=2,
        batch_size=32,
        random_state=7,
        device="cpu",
    )
    attention_model.fit(
        sequence_features[first_train_index],
        target[first_train_index],
    )
    attention_predictions = attention_model.predict(sequence_features[first_test_index])

    print(f"CPCV splits: {cv.get_n_splits()}")
    print(f"CPCV backtest paths: {cv.n_backtest_paths}")
    print(f"PSR: {dsr_result.probabilistic_sharpe_ratio:.6f}")
    print(f"DSR: {dsr_result.deflated_sharpe_ratio:.6f}")
    print(f"Attention-LSTM predictions: {attention_predictions[:5]}")
