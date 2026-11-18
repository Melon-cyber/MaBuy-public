"""Microstructure feature computation and ML-ready feature orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

_EPSILON: float = 1e-12
_BETA_EPSILON: float = 1e-8
_OFI_SUM_WINDOW: int = 50
_OFI_LIQUIDITY_WINDOW: int = 100
_CVD_DIVERGENCE_WINDOW: int = 50


def _require_columns(frame: pd.DataFrame, required_columns: Sequence[str]) -> None:
    """Validate that a DataFrame contains all required columns."""
    missing_columns = sorted(set(required_columns).difference(frame.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def _coerce_finite_numeric(
    series: pd.Series,
    column_name: str,
    *,
    allow_negative: bool = True,
    allow_zero: bool = True,
) -> pd.Series:
    """Convert a Series to finite float values and validate its domain."""
    numeric = pd.to_numeric(series, errors="coerce").astype(float)

    if numeric.isna().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(f"Column '{column_name}' must contain finite numeric values.")

    if not allow_negative and (numeric < 0.0).any():
        raise ValueError(f"Column '{column_name}' must not contain negative values.")

    if not allow_zero and (numeric <= 0.0).any():
        raise ValueError(f"Column '{column_name}' must contain strictly positive values.")

    return numeric


def _safe_divide(
    numerator: pd.Series | np.ndarray | float,
    denominator: pd.Series | np.ndarray | float,
    *,
    default: float = 0.0,
    epsilon: float = _EPSILON,
) -> pd.Series | np.ndarray | float:
    """Safely divide numeric values while replacing invalid results with a default."""
    if isinstance(numerator, pd.Series):
        denominator_series = (
            denominator
            if isinstance(denominator, pd.Series)
            else pd.Series(denominator, index=numerator.index)
        )
        valid = denominator_series.abs() > epsilon
        result = pd.Series(default, index=numerator.index, dtype=float)
        result.loc[valid] = numerator.loc[valid] / denominator_series.loc[valid]
        return result.replace([np.inf, -np.inf], default).fillna(default)

    numerator_array = np.asarray(numerator, dtype=float)
    denominator_array = np.asarray(denominator, dtype=float)
    output = np.full(
        np.broadcast(numerator_array, denominator_array).shape,
        default,
        dtype=float,
    )
    np.divide(
        numerator_array,
        denominator_array,
        out=output,
        where=np.abs(denominator_array) > epsilon,
    )
    output[~np.isfinite(output)] = default

    if output.ndim == 0:
        return float(output)
    return output


def _rolling_zscore(
    series: pd.Series,
    window: int,
    *,
    min_periods: int | None = None,
) -> pd.Series:
    """Compute a rolling z-score with finite fallback values."""
    if window < 1:
        raise ValueError("window must be greater than or equal to one.")

    effective_min_periods = min_periods if min_periods is not None else window
    rolling_mean = series.rolling(window=window, min_periods=effective_min_periods).mean()
    rolling_std = series.rolling(window=window, min_periods=effective_min_periods).std()

    zscore = (series - rolling_mean) / (rolling_std + _EPSILON)
    return zscore.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _latest_zscore(series: pd.Series) -> float:
    """Return the latest full-history z-score, or zero for insufficient variation."""
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan).dropna()

    if numeric.empty:
        return 0.0

    values = numeric.to_numpy(dtype=float)
    standard_deviation = float(np.std(values, ddof=0))
    if not np.isfinite(standard_deviation) or standard_deviation <= _EPSILON:
        return 0.0

    zscore = (float(values[-1]) - float(np.mean(values))) / (standard_deviation + _EPSILON)
    return float(zscore) if np.isfinite(zscore) else 0.0


def _safe_log_return(current: float, previous: float) -> float:
    """Compute a finite logarithmic return with invalid-domain protection."""
    if (
        not np.isfinite(current)
        or not np.isfinite(previous)
        or current <= 0.0
        or previous <= 0.0
    ):
        return 0.0

    result = float(np.log(current / previous))
    return result if np.isfinite(result) else 0.0


def _normalise_mapping_key(key: Any) -> str:
    """Normalize mapping keys for case-insensitive alias matching."""
    return "".join(character for character in str(key).lower() if character.isalnum())


def _mapping_value(
    mapping: Mapping[str, Any],
    aliases: Sequence[str],
    default: Any = None,
) -> Any:
    """Retrieve a mapping value using normalized key aliases."""
    normalized_mapping = {
        _normalise_mapping_key(key): value
        for key, value in mapping.items()
    }

    for alias in aliases:
        normalized_alias = _normalise_mapping_key(alias)
        if normalized_alias in normalized_mapping:
            return normalized_mapping[normalized_alias]

    return default


def _history_numeric_series(
    history: Sequence[Mapping[str, Any]],
    aliases: Sequence[str],
) -> pd.Series:
    """Extract a finite numeric series from a history of mapping records."""
    values: list[float] = []

    for record in history:
        value = _mapping_value(record, aliases)
        numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]

        if pd.notna(numeric_value) and np.isfinite(float(numeric_value)):
            values.append(float(numeric_value))

    return pd.Series(values, dtype=float)


def _extract_price_return_series(
    funding_history: Sequence[Mapping[str, Any]],
    oi_history: Sequence[Mapping[str, Any]],
) -> pd.Series:
    """Extract supplied returns or infer log returns from available price histories."""
    return_aliases = (
        "return",
        "price_return",
        "log_return",
        "asset_return",
        "market_return",
    )
    price_aliases = (
        "price",
        "mark_price",
        "markPrice",
        "index_price",
        "indexPrice",
        "last_price",
        "lastPrice",
    )

    direct_returns = _history_numeric_series(funding_history, return_aliases)
    if direct_returns.empty:
        direct_returns = _history_numeric_series(oi_history, return_aliases)

    if not direct_returns.empty:
        return direct_returns

    prices = _history_numeric_series(funding_history, price_aliases)
    if prices.empty:
        prices = _history_numeric_series(oi_history, price_aliases)

    if prices.empty:
        return pd.Series(dtype=float)

    returns = [
        _safe_log_return(float(current), float(previous))
        for previous, current in zip(prices.iloc[:-1], prices.iloc[1:])
    ]
    return pd.Series(returns, dtype=float)


def _parse_buyer_is_maker(series: pd.Series) -> pd.Series:
    """Convert common boolean representations to Binance buyer-is-maker semantics."""
    parsed_values: list[bool] = []

    for value in series:
        if isinstance(value, (bool, np.bool_)):
            parsed_values.append(bool(value))
            continue

        if isinstance(value, (int, np.integer)) and value in (0, 1):
            parsed_values.append(bool(value))
            continue

        if isinstance(value, float) and value in (0.0, 1.0):
            parsed_values.append(bool(value))
            continue

        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes", "y"}:
                parsed_values.append(True)
                continue
            if normalized in {"false", "0", "no", "n"}:
                parsed_values.append(False)
                continue

        raise ValueError(
            "Column 'buyer_is_maker' must contain boolean-compatible values."
        )

    return pd.Series(parsed_values, index=series.index, dtype=bool)


def compute_top_of_book_features(book_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute top-of-book microstructure features.

    Required columns are ``ts``, ``bid_px``, ``bid_qty``, ``ask_px``, and
    ``ask_qty``. Rows are ordered by timestamp using a stable sort before
    temporal features are calculated.

    Returns:
        A copy of the normalized input frame with ``mid``, ``spread``,
        ``microprice``, ``microprice_delta``, ``ofi_increment``, and
        ``ofi_norm`` columns.
    """
    required_columns = ("ts", "bid_px", "bid_qty", "ask_px", "ask_qty")
    _require_columns(book_df, required_columns)

    result = book_df.copy(deep=True)

    if result.empty:
        for column in (
            "mid",
            "spread",
            "microprice",
            "microprice_delta",
            "ofi_increment",
            "ofi_norm",
        ):
            result[column] = pd.Series(dtype=float)
        return result

    if result["ts"].isna().any():
        raise ValueError("Column 'ts' must not contain null values.")

    result = result.sort_values("ts", kind="mergesort").reset_index(drop=True)
    result["bid_px"] = _coerce_finite_numeric(result["bid_px"], "bid_px")
    result["bid_qty"] = _coerce_finite_numeric(
        result["bid_qty"],
        "bid_qty",
        allow_negative=False,
    )
    result["ask_px"] = _coerce_finite_numeric(result["ask_px"], "ask_px")
    result["ask_qty"] = _coerce_finite_numeric(
        result["ask_qty"],
        "ask_qty",
        allow_negative=False,
    )

    bid_px = result["bid_px"]
    bid_qty = result["bid_qty"]
    ask_px = result["ask_px"]
    ask_qty = result["ask_qty"]

    result["mid"] = (bid_px + ask_px) / 2.0
    result["spread"] = ask_px - bid_px

    displayed_quantity = bid_qty + ask_qty
    microprice_numerator = (ask_px * bid_qty) + (bid_px * ask_qty)
    result["microprice"] = _safe_divide(
        microprice_numerator,
        displayed_quantity,
        default=0.0,
    )

    result["microprice_delta"] = (
        (result["microprice"] - result["mid"]) / (result["spread"] + _EPSILON)
    )
    result["microprice_delta"] = (
        result["microprice_delta"]
        .replace([np.inf, -np.inf], 0.0)
        .fillna(0.0)
    )

    previous_bid_px = bid_px.shift(1)
    previous_bid_qty = bid_qty.shift(1)
    previous_ask_px = ask_px.shift(1)
    previous_ask_qty = ask_qty.shift(1)

    bid_additions = pd.Series(
        np.where(bid_px >= previous_bid_px, bid_qty, 0.0),
        index=result.index,
        dtype=float,
    )
    bid_removals = pd.Series(
        np.where(bid_px <= previous_bid_px, previous_bid_qty, 0.0),
        index=result.index,
        dtype=float,
    )
    ask_additions = pd.Series(
        np.where(ask_px <= previous_ask_px, ask_qty, 0.0),
        index=result.index,
        dtype=float,
    )
    ask_removals = pd.Series(
        np.where(ask_px >= previous_ask_px, previous_ask_qty, 0.0),
        index=result.index,
        dtype=float,
    )

    result["ofi_increment"] = (
        bid_additions - bid_removals - ask_additions + ask_removals
    ).fillna(0.0)

    rolling_ofi = result["ofi_increment"].rolling(
        window=_OFI_SUM_WINDOW,
        min_periods=1,
    ).sum()
    rolling_liquidity = displayed_quantity.rolling(
        window=_OFI_LIQUIDITY_WINDOW,
        min_periods=1,
    ).median()

    result["ofi_norm"] = _safe_divide(
        rolling_ofi,
        rolling_liquidity,
        default=0.0,
    )

    return result


def compute_cvd_features(agg_trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cumulative volume delta and CVD/price divergence features.

    Binance aggregate-trade semantics are used: ``buyer_is_maker=True`` means
    an aggressive seller and therefore negative signed volume; ``False`` means
    an aggressive buyer and therefore positive signed volume.

    Returns:
        A timestamp-ordered copy of the input with ``signed_volume``, ``cvd``,
        and ``cvd_divergence`` columns.
    """
    required_columns = ("ts", "price", "quantity", "buyer_is_maker")
    _require_columns(agg_trades_df, required_columns)

    result = agg_trades_df.copy(deep=True)

    if result.empty:
        for column in ("signed_volume", "cvd", "cvd_divergence"):
            result[column] = pd.Series(dtype=float)
        return result

    if result["ts"].isna().any():
        raise ValueError("Column 'ts' must not contain null values.")

    result = result.sort_values("ts", kind="mergesort").reset_index(drop=True)
    result["price"] = _coerce_finite_numeric(
        result["price"],
        "price",
        allow_negative=False,
        allow_zero=False,
    )
    result["quantity"] = _coerce_finite_numeric(
        result["quantity"],
        "quantity",
        allow_negative=False,
    )
    result["buyer_is_maker"] = _parse_buyer_is_maker(result["buyer_is_maker"])

    aggressor_sign = np.where(result["buyer_is_maker"], -1.0, 1.0)
    result["signed_volume"] = aggressor_sign * result["quantity"]
    result["cvd"] = result["signed_volume"].cumsum()

    cvd_change = result["cvd"].diff(_CVD_DIVERGENCE_WINDOW)
    log_price_change = (
        np.log(result["price"])
        .diff(_CVD_DIVERGENCE_WINDOW)
        .replace([np.inf, -np.inf], np.nan)
    )

    cvd_change_zscore = _rolling_zscore(
        cvd_change,
        _CVD_DIVERGENCE_WINDOW,
    )
    log_price_change_zscore = _rolling_zscore(
        log_price_change,
        _CVD_DIVERGENCE_WINDOW,
    )

    result["cvd_divergence"] = (
        cvd_change_zscore - log_price_change_zscore
    ).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    return result


def compute_funding_velocity_and_oi(
    funding_history: list[dict[str, Any]],
    oi_history: list[dict[str, Any]],
) -> dict[str, float]:
    """
    Compute funding velocity, open-interest change, and squeeze scores.

    Accepted funding aliases include ``funding_rate`` and Binance's
    ``fundingRate``. Accepted open-interest aliases include ``oi``,
    ``open_interest``, and Binance's ``openInterest``.

    The resulting z-scores are calculated over the supplied histories and the
    returned values correspond to the latest observation.
    """
    if not isinstance(funding_history, list):
        raise TypeError("funding_history must be a list of dictionaries.")
    if not isinstance(oi_history, list):
        raise TypeError("oi_history must be a list of dictionaries.")

    if not all(isinstance(record, Mapping) for record in funding_history):
        raise TypeError("Each funding_history record must be a dictionary.")
    if not all(isinstance(record, Mapping) for record in oi_history):
        raise TypeError("Each oi_history record must be a dictionary.")

    funding_rates = _history_numeric_series(
        funding_history,
        ("funding_rate", "fundingRate", "funding", "rate"),
    )
    open_interest = _history_numeric_series(
        oi_history,
        (
            "open_interest",
            "openInterest",
            "oi",
            "sum_open_interest",
            "sumOpenInterest",
        ),
    )

    funding_velocity_series = funding_rates.diff().fillna(0.0)

    if open_interest.empty:
        oi_change_series = pd.Series(dtype=float)
    else:
        oi_changes = [
            _safe_log_return(float(current), float(previous))
            for previous, current in zip(
                open_interest.iloc[:-1],
                open_interest.iloc[1:],
            )
        ]
        oi_change_series = pd.Series([0.0, *oi_changes], dtype=float)

    return_series = _extract_price_return_series(funding_history, oi_history)

    funding_rate = float(funding_rates.iloc[-1]) if not funding_rates.empty else 0.0
    funding_velocity = (
        float(funding_velocity_series.iloc[-1])
        if not funding_velocity_series.empty
        else 0.0
    )
    oi_change = (
        float(oi_change_series.iloc[-1])
        if not oi_change_series.empty
        else 0.0
    )
    latest_return = float(return_series.iloc[-1]) if not return_series.empty else 0.0

    funding_zscore = _latest_zscore(funding_rates)
    funding_velocity_zscore = _latest_zscore(funding_velocity_series)
    oi_change_zscore = _latest_zscore(oi_change_series)
    return_zscore = _latest_zscore(return_series)

    long_squeeze_score = (
        funding_zscore
        + funding_velocity_zscore
        + oi_change_zscore
        - return_zscore
    )
    short_squeeze_score = (
        -funding_zscore
        - funding_velocity_zscore
        + oi_change_zscore
        + return_zscore
    )

    return {
        "funding_rate": funding_rate,
        "funding_velocity": funding_velocity,
        "open_interest": (
            float(open_interest.iloc[-1]) if not open_interest.empty else 0.0
        ),
        "oi_change": oi_change,
        "return": latest_return,
        "funding_zscore": funding_zscore,
        "funding_velocity_zscore": funding_velocity_zscore,
        "oi_change_zscore": oi_change_zscore,
        "return_zscore": return_zscore,
        "long_squeeze_score": float(long_squeeze_score),
        "short_squeeze_score": float(short_squeeze_score),
    }


def compute_cross_asset_beta_residuals(
    asset_returns: pd.Series,
    btc_returns: pd.Series,
    window: int = 60,
) -> pd.Series:
    """
    Compute rolling beta-adjusted residual z-scores against BTC returns.

    Rolling beta is defined as::

        Cov(r_asset, r_btc) / (Var(r_btc) + 1e-8)

    The residual is standardized using its rolling mean and standard deviation.
    Warm-up observations without sufficient data remain ``NaN``.
    """
    if not isinstance(asset_returns, pd.Series):
        raise TypeError("asset_returns must be a pandas Series.")
    if not isinstance(btc_returns, pd.Series):
        raise TypeError("btc_returns must be a pandas Series.")
    if window < 2:
        raise ValueError("window must be at least 2.")

    aligned = pd.concat(
        [
            pd.to_numeric(asset_returns, errors="coerce").rename("asset"),
            pd.to_numeric(btc_returns, errors="coerce").rename("btc"),
        ],
        axis=1,
        join="inner",
    ).sort_index(kind="mergesort")

    if aligned.empty:
        return pd.Series(
            np.nan,
            index=asset_returns.index,
            dtype=float,
            name="cross_asset_residual_z",
        )

    rolling_covariance = aligned["asset"].rolling(window=window).cov(aligned["btc"])
    rolling_variance = aligned["btc"].rolling(window=window).var()
    rolling_beta = rolling_covariance / (rolling_variance + _BETA_EPSILON)

    residual = aligned["asset"] - (rolling_beta * aligned["btc"])
    residual_mean = residual.rolling(window=window).mean()
    residual_std = residual.rolling(window=window).std()

    standardized_residual = (residual - residual_mean) / (
        residual_std + _BETA_EPSILON
    )
    standardized_residual = standardized_residual.replace([np.inf, -np.inf], np.nan)
    standardized_residual.name = "cross_asset_residual_z"

    return standardized_residual.reindex(asset_returns.index)


@dataclass
class _SymbolFeatureState:
    """Mutable per-symbol state retained by the feature engine."""

    previous_mid: float | None = None
    funding_history: list[dict[str, float]] = field(default_factory=list)
    oi_history: list[dict[str, float]] = field(default_factory=list)
    asset_returns: list[float] = field(default_factory=list)
    btc_returns: list[float] = field(default_factory=list)


@dataclass
class MicrostructureFeatureEngine:
    """
    Stateful high-level orchestrator for ML-ready microstructure features.

    The engine retains a bounded per-symbol history for funding, open interest,
    and BTC-relative return calculations. ``extract_features`` accepts either
    canonical top-of-book fields or Binance-style ``bids``/``asks`` snapshots.
    """

    history_limit: int = 500
    _states: dict[str, _SymbolFeatureState] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate engine configuration."""
        if self.history_limit < 60:
            raise ValueError("history_limit must be at least 60.")

    @staticmethod
    def _first_book_level(levels: Any, side: str) -> tuple[Any, Any]:
        """Extract price and quantity from a top depth level."""
        if not isinstance(levels, Sequence) or isinstance(levels, (str, bytes)):
            raise ValueError(f"Orderbook '{side}' levels must be a non-empty sequence.")
        if not levels:
            raise ValueError(f"Orderbook '{side}' levels must not be empty.")

        level = levels[0]
        if isinstance(level, Mapping):
            price = _mapping_value(level, ("price", "px", "p"))
            quantity = _mapping_value(level, ("quantity", "qty", "size", "q"))
            return price, quantity

        if isinstance(level, Sequence) and not isinstance(level, (str, bytes)):
            if len(level) < 2:
                raise ValueError(
                    f"Orderbook '{side}' top level must contain price and quantity."
                )
            return level[0], level[1]

        raise ValueError(f"Unsupported orderbook '{side}' level format.")

    @staticmethod
    def _scalar_or_first(value: Any) -> Any:
        """Return a scalar value or the first item from a non-string sequence."""
        if isinstance(value, pd.Series):
            return value.iloc[0] if not value.empty else None
        if isinstance(value, np.ndarray):
            return value.flat[0] if value.size else None
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return value[0] if value else None
        return value

    def _orderbook_to_frame(self, orderbook: Mapping[str, Any]) -> tuple[pd.DataFrame, float | None]:
        """Normalize canonical and Binance depth snapshots into a DataFrame."""
        if not isinstance(orderbook, Mapping):
            raise TypeError("orderbook must be a dictionary-like mapping.")

        bid_px = _mapping_value(orderbook, ("bid_px", "bidPrice", "best_bid"))
        bid_qty = _mapping_value(orderbook, ("bid_qty", "bidQty", "best_bid_qty"))
        ask_px = _mapping_value(orderbook, ("ask_px", "askPrice", "best_ask"))
        ask_qty = _mapping_value(orderbook, ("ask_qty", "askQty", "best_ask_qty"))
        timestamp = _mapping_value(
            orderbook,
            ("ts", "timestamp", "time", "event_time", "eventTime", "T"),
        )

        if any(value is None for value in (bid_px, bid_qty, ask_px, ask_qty)):
            bids = _mapping_value(orderbook, ("bids", "bid"))
            asks = _mapping_value(orderbook, ("asks", "ask"))
            bid_px, bid_qty = self._first_book_level(bids, "bids")
            ask_px, ask_qty = self._first_book_level(asks, "asks")

        timestamp = self._scalar_or_first(timestamp)
        if timestamp is None:
            timestamp = pd.Timestamp.now(tz="UTC")

        frame = pd.DataFrame(
            [
                {
                    "ts": timestamp,
                    "bid_px": self._scalar_or_first(bid_px),
                    "bid_qty": self._scalar_or_first(bid_qty),
                    "ask_px": self._scalar_or_first(ask_px),
                    "ask_qty": self._scalar_or_first(ask_qty),
                }
            ]
        )

        oi_value = _mapping_value(
            orderbook,
            (
                "open_interest",
                "openInterest",
                "oi",
                "sum_open_interest",
                "sumOpenInterest",
            ),
        )
        oi_value = self._scalar_or_first(oi_value)

        if oi_value is None:
            return frame, None

        numeric_oi = pd.to_numeric(pd.Series([oi_value]), errors="coerce").iloc[0]
        if pd.isna(numeric_oi) or not np.isfinite(float(numeric_oi)):
            return frame, None

        return frame, float(numeric_oi)

    @staticmethod
    def _trades_to_frame(recent_trades: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
        """Normalize Binance-style aggregate trades into canonical trade columns."""
        if not isinstance(recent_trades, Sequence) or isinstance(
            recent_trades,
            (str, bytes),
        ):
            raise TypeError("recent_trades must be a sequence of dictionaries.")

        normalized_records: list[dict[str, Any]] = []

        for trade in recent_trades:
            if not isinstance(trade, Mapping):
                raise TypeError("Each recent trade must be a dictionary-like mapping.")

            timestamp = _mapping_value(
                trade,
                ("ts", "timestamp", "time", "trade_time", "tradeTime", "T"),
            )
            price = _mapping_value(trade, ("price", "p"))
            quantity = _mapping_value(trade, ("quantity", "qty", "q"))
            buyer_is_maker = _mapping_value(
                trade,
                ("buyer_is_maker", "buyerIsMaker", "is_buyer_maker", "m"),
            )

            if timestamp is None:
                timestamp = pd.Timestamp.now(tz="UTC")

            normalized_records.append(
                {
                    "ts": timestamp,
                    "price": price,
                    "quantity": quantity,
                    "buyer_is_maker": buyer_is_maker,
                }
            )

        return pd.DataFrame(
            normalized_records,
            columns=("ts", "price", "quantity", "buyer_is_maker"),
        )

    @staticmethod
    def _finite_feature(value: Any) -> float:
        """Convert a feature value to a finite float suitable for model ingestion."""
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return 0.0

        return numeric_value if np.isfinite(numeric_value) else 0.0

    def _trim_state(self, state: _SymbolFeatureState) -> None:
        """Bound retained state to the configured history limit."""
        state.funding_history = state.funding_history[-self.history_limit :]
        state.oi_history = state.oi_history[-self.history_limit :]
        state.asset_returns = state.asset_returns[-self.history_limit :]
        state.btc_returns = state.btc_returns[-self.history_limit :]

    def extract_features(
        self,
        symbol: str,
        orderbook: dict[str, Any],
        recent_trades: list[dict[str, Any]],
        funding_rate: float,
        btc_return: float,
    ) -> dict[str, float]:
        """
        Extract finite, normalized microstructure features for ML ingestion.

        Args:
            symbol: Instrument identifier used for independent state tracking.
            orderbook: Canonical top-of-book fields or Binance ``bids``/``asks``.
            recent_trades: Canonical or Binance aggregate-trade records.
            funding_rate: Latest perpetual funding rate.
            btc_return: Latest BTC return over the corresponding model horizon.

        Returns:
            A dictionary of finite scale-normalized numerical model features.
        """
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("symbol must be a non-empty string.")

        numeric_funding_rate = self._finite_feature(funding_rate)
        numeric_btc_return = self._finite_feature(btc_return)

        orderbook_frame, open_interest = self._orderbook_to_frame(orderbook)
        book_features = compute_top_of_book_features(orderbook_frame)

        trades_frame = self._trades_to_frame(recent_trades)
        trade_features = compute_cvd_features(trades_frame)

        latest_book = book_features.iloc[-1]
        latest_trade = trade_features.iloc[-1] if not trade_features.empty else None

        current_mid = self._finite_feature(latest_book["mid"])
        spread_bps = _safe_divide(
            float(latest_book["spread"]) * 10_000.0,
            current_mid,
            default=0.0,
        )
        spread_bps = self._finite_feature(spread_bps)

        with self._lock:
            state = self._states.setdefault(symbol, _SymbolFeatureState())

            asset_return = (
                _safe_log_return(current_mid, state.previous_mid)
                if state.previous_mid is not None
                else 0.0
            )
            state.previous_mid = current_mid if current_mid > 0.0 else state.previous_mid

            state.funding_history.append({"funding_rate": numeric_funding_rate})
            if open_interest is not None:
                state.oi_history.append({"open_interest": open_interest})

            state.asset_returns.append(asset_return)
            state.btc_returns.append(numeric_btc_return)
            self._trim_state(state)

            funding_features = compute_funding_velocity_and_oi(
                state.funding_history,
                state.oi_history,
            )

            residual_zscores = compute_cross_asset_beta_residuals(
                pd.Series(state.asset_returns, dtype=float),
                pd.Series(state.btc_returns, dtype=float),
                window=60,
            )
            cross_asset_residual_z = (
                self._finite_feature(residual_zscores.iloc[-1])
                if not residual_zscores.empty
                else 0.0
            )

        latest_cvd_divergence = (
            self._finite_feature(latest_trade["cvd_divergence"])
            if latest_trade is not None
            else 0.0
        )

        return {
            "spread_bps": spread_bps,
            "microprice_delta": self._finite_feature(
                latest_book["microprice_delta"]
            ),
            "ofi_norm": self._finite_feature(latest_book["ofi_norm"]),
            "cvd_divergence": latest_cvd_divergence,
            "funding_zscore": self._finite_feature(
                funding_features["funding_zscore"]
            ),
            "funding_velocity_zscore": self._finite_feature(
                funding_features["funding_velocity_zscore"]
            ),
            "oi_change_zscore": self._finite_feature(
                funding_features["oi_change_zscore"]
            ),
            "return_zscore": self._finite_feature(
                funding_features["return_zscore"]
            ),
            "long_squeeze_score": self._finite_feature(
                funding_features["long_squeeze_score"]
            ),
            "short_squeeze_score": self._finite_feature(
                funding_features["short_squeeze_score"]
            ),
            "cross_asset_residual_z": cross_asset_residual_z,
            "asset_return": self._finite_feature(asset_return),
            "btc_return": numeric_btc_return,
        }