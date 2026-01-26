"""
Liquidity Sweep Reversal Strategy.

Trade reversals after institutional stop hunts (liquidity sweeps).
This is the "liquidity candle" pattern - price sweeps beyond a level,
then reverses sharply.
"""


import logging

import pandas as pd

from frakt.core.liquidity import detect_liquidity_sweep, find_equal_levels
from frakt.core.market_structure import determine_trend, find_swing_points
from frakt.strategies.base import BaseStrategy, Signal

logger = logging.getLogger(__name__)


class LiquiditySweepStrategy(BaseStrategy):
    """
    Trade reversals after liquidity sweeps.

    Entry Logic:
    1. Identify liquidity levels (swing highs/lows, equal levels)
    2. Wait for price to sweep the level (break beyond it)
    3. Enter on reversal candle (close back inside)
    4. Stop loss: Beyond the sweep wick
    5. Take profit: Previous structure level or 2:1 RR

    Multi-Timeframe Mode (ADR 0.04.0014):
    - HTF (H4): Determine trend direction (only trade WITH trend)
    - MTF (H1): Confirm structure (BOS/CHoCH)
    - LTF (M15): Find precise entry (liquidity sweeps)

    Example:
        >>> strategy = LiquiditySweepStrategy({'swing_period': 5})
        >>> signals = strategy.generate_signals(btc_data)
        >>> for signal in signals:
        ...     print(f"{signal.timestamp}: {signal.direction} @ {signal.entry_price}")
    """

    DEFAULT_PARAMS = {
        "swing_period": 5,  # Bars for swing detection
        "max_reversal_bars": 3,  # Must reverse within N bars
        "min_rr_ratio": 1.5,  # Minimum risk:reward
        "tolerance": 0.001,  # Tolerance for equal level detection
        "atr_period": 14,  # ATR period for volatility
        "atr_stop_buffer": 0.5,  # ATR multiplier for stop loss buffer
        # Multi-TF confidence bonuses (ADR 0.04.0014)
        "htf_trend_bonus": 20,  # Bonus for HTF trend alignment
        "mtf_structure_bonus": 10,  # Bonus for MTF structure confirmation
    }

    def __init__(self, params: dict | None = None):
        """Initialize the Liquidity Sweep Strategy."""
        merged_params = {**self.DEFAULT_PARAMS, **(params or {})}
        super().__init__("liquidity_sweep", merged_params)

    def generate_signals(self, data: pd.DataFrame) -> list[Signal]:
        """
        Generate trading signals from OHLCV data.

        Args:
            data: DataFrame with columns [open, high, low, close, volume]

        Returns:
            List of Signal objects for valid setups
        """
        self.validate_data(data)

        # Ensure data is chronologically sorted (required for recency filtering)
        assert data.index.is_monotonic_increasing, "Data must be chronologically sorted"

        logger.debug(f"Analyzing {len(data)} candles for liquidity sweeps")
        signals: list[Signal] = []

        # 1. Find swing points
        swing_highs, swing_lows = find_swing_points(
            data["high"], data["low"], n=self.params["swing_period"]
        )

        # 2. Find equal levels (additional liquidity)
        equal_highs, equal_lows = find_equal_levels(
            swing_highs, swing_lows, tolerance=self.params["tolerance"]
        )

        # 3. Combine swing points and equal levels as liquidity
        # For bullish sweeps: use lows as liquidity levels
        bullish_liquidity = self._combine_liquidity_levels(swing_lows, equal_lows)
        # For bearish sweeps: use highs as liquidity levels
        bearish_liquidity = self._combine_liquidity_levels(swing_highs, equal_highs)

        # 4. Detect sweeps
        bullish_sweeps = detect_liquidity_sweep(
            data["high"],
            data["low"],
            data["close"],
            liquidity_levels=bullish_liquidity,
            reversal_bars=self.params["max_reversal_bars"],
            direction="bullish",
        )

        bearish_sweeps = detect_liquidity_sweep(
            data["high"],
            data["low"],
            data["close"],
            liquidity_levels=bearish_liquidity,
            reversal_bars=self.params["max_reversal_bars"],
            direction="bearish",
        )

        # 5. Generate signals
        for idx in data.index[bullish_sweeps]:
            signal = self._create_long_signal(data, idx, swing_highs, swing_lows)
            if signal is not None:
                signals.append(signal)

        for idx in data.index[bearish_sweeps]:
            signal = self._create_short_signal(data, idx, swing_highs, swing_lows)
            if signal is not None:
                signals.append(signal)

        # 6. Filter by RR ratio
        signals = self.filter_signals_by_rr(signals, self.params["min_rr_ratio"])

        # 7. Return ALL signals (no recency filter)
        # NOTE: Recency filtering moved to BaseStrategy.filter_signals_by_recency()
        # Caller (live bot) can apply it externally if needed
        # Backtests get all historical signals (correct behavior)

        if signals:
            logger.info(
                f"Found {len(signals)} FRESH liquidity sweep signal(s): "
                f"{sum(1 for s in signals if s.direction == 1)} LONG, "
                f"{sum(1 for s in signals if s.direction == -1)} SHORT"
            )
            for signal in signals:
                logger.debug(
                    f"  {signal.timestamp}: {'LONG' if signal.direction == 1 else 'SHORT'} "
                    f"@ {signal.entry_price:.2f}, SL: {signal.stop_loss:.2f}, "
                    f"TP: {signal.take_profit:.2f}"
                )
        else:
            logger.debug("No fresh liquidity sweep signals found")

        return signals

    def generate_signals_multi_tf(
        self,
        htf_data: pd.DataFrame,
        mtf_data: pd.DataFrame,
        ltf_data: pd.DataFrame
    ) -> list[Signal]:
        """
        Generate signals using multi-timeframe analysis (ADR 0.04.0014).

        Flow per Manifesto Chapter 5:
        1. HTF (H4): Determine trend direction → only trade WITH trend
        2. MTF (H1): Confirm structure (recent BOS in trend direction)
        3. LTF (M15): Find liquidity sweeps for precise entry

        Args:
            htf_data: H4 data for trend analysis
            mtf_data: H1 data for structure confirmation
            ltf_data: M15 data for entry signals

        Returns:
            List of signals filtered by HTF trend with confidence adjustments
        """
        # 1. Determine HTF trend
        htf_trend = self._determine_htf_trend(htf_data)
        logger.info(f"HTF Trend: {'BULLISH' if htf_trend == 1 else 'BEARISH' if htf_trend == -1 else 'RANGING'}")

        # 2. Check MTF structure
        mtf_has_bos = self._check_mtf_structure(mtf_data, htf_trend)
        logger.debug(f"MTF Structure confirmed: {mtf_has_bos}")

        # 3. Generate LTF signals (using existing single-TF method)
        ltf_signals = self.generate_signals(ltf_data)

        if not ltf_signals:
            return []

        # 4. Filter and adjust signals based on multi-TF context
        filtered_signals = []
        for signal in ltf_signals:
            # Skip signals against HTF trend (unless ranging)
            if htf_trend != 0:  # Not ranging
                if htf_trend == 1 and signal.direction == -1:
                    logger.debug(f"Skipping SHORT signal (HTF is BULLISH)")
                    continue
                if htf_trend == -1 and signal.direction == 1:
                    logger.debug(f"Skipping LONG signal (HTF is BEARISH)")
                    continue

            # Adjust confidence based on multi-TF alignment
            confidence_bonus = 0

            # HTF trend alignment bonus
            if htf_trend != 0 and htf_trend == signal.direction:
                confidence_bonus += self.params["htf_trend_bonus"]
                signal.metadata["htf_aligned"] = True

            # MTF structure bonus
            if mtf_has_bos:
                confidence_bonus += self.params["mtf_structure_bonus"]
                signal.metadata["mtf_confirmed"] = True

            # Apply bonus (cap at 100)
            new_confidence = min(100, signal.confidence + confidence_bonus)

            # Create new signal with updated confidence
            adjusted_signal = Signal(
                timestamp=signal.timestamp,
                direction=signal.direction,
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                confidence=new_confidence,
                strategy_name=signal.strategy_name,
                metadata={
                    **signal.metadata,
                    "htf_trend": htf_trend,
                    "mtf_bos": mtf_has_bos,
                    "original_confidence": signal.confidence,
                    "multi_tf": True,
                },
            )
            filtered_signals.append(adjusted_signal)

        if filtered_signals:
            logger.info(
                f"Multi-TF signals: {len(filtered_signals)} "
                f"(filtered from {len(ltf_signals)} LTF signals, "
                f"HTF={'BULL' if htf_trend == 1 else 'BEAR' if htf_trend == -1 else 'RANGE'})"
            )

        return filtered_signals

    def _determine_htf_trend(self, htf_data: pd.DataFrame) -> int:
        """
        Determine overall trend from HTF (H4) data.

        Returns:
            1 = bullish, -1 = bearish, 0 = ranging
        """
        if len(htf_data) < 20:
            return 0  # Not enough data

        # Find swing points on HTF
        swing_highs, swing_lows = find_swing_points(
            htf_data["high"], htf_data["low"], n=self.params["swing_period"]
        )

        # Use market structure to determine trend
        trend = determine_trend(swing_highs, swing_lows)

        if len(trend) == 0:
            return 0

        # Get recent trend (last 5 candles average)
        recent_trend = trend.iloc[-5:].mean() if len(trend) >= 5 else trend.mean()

        if recent_trend > 0.3:
            return 1  # Bullish
        elif recent_trend < -0.3:
            return -1  # Bearish
        else:
            return 0  # Ranging

    def _check_mtf_structure(self, mtf_data: pd.DataFrame, expected_direction: int) -> bool:
        """
        Check if MTF (H1) has recent BOS in expected direction.

        Args:
            mtf_data: H1 OHLCV data
            expected_direction: 1 for bullish BOS, -1 for bearish BOS

        Returns:
            True if structure confirms the expected direction
        """
        if len(mtf_data) < 20 or expected_direction == 0:
            return False

        # Find swing points on MTF
        swing_highs, swing_lows = find_swing_points(
            mtf_data["high"], mtf_data["low"], n=self.params["swing_period"]
        )

        # Check for recent BOS in expected direction
        # Bullish BOS: New swing high above previous swing high
        # Bearish BOS: New swing low below previous swing low
        recent_highs = swing_highs.dropna().tail(3)
        recent_lows = swing_lows.dropna().tail(3)

        if expected_direction == 1 and len(recent_highs) >= 2:
            # Check if latest high is above previous high (bullish BOS)
            if recent_highs.iloc[-1] > recent_highs.iloc[-2]:
                return True

        if expected_direction == -1 and len(recent_lows) >= 2:
            # Check if latest low is below previous low (bearish BOS)
            if recent_lows.iloc[-1] < recent_lows.iloc[-2]:
                return True

        return False

    def _combine_liquidity_levels(
        self, swing_levels: pd.Series, equal_levels: pd.Series
    ) -> pd.Series:
        """
        Combine swing levels and equal levels into single liquidity series.

        Prioritizes equal levels (stronger liquidity) over swing levels.
        """
        combined = swing_levels.copy()

        # Override with equal levels where they exist
        equal_mask = equal_levels.notna()
        combined.loc[equal_mask] = equal_levels.loc[equal_mask]

        return combined

    def _create_long_signal(
        self, data: pd.DataFrame, idx: pd.Timestamp, swing_highs: pd.Series, swing_lows: pd.Series
    ) -> Signal | None:
        """
        Create a long signal after bullish liquidity sweep.

        Args:
            data: OHLCV DataFrame
            idx: Timestamp of the signal bar
            swing_highs: Swing high prices
            swing_lows: Swing low prices

        Returns:
            Signal object or None if invalid
        """
        try:
            entry = data.loc[idx, "close"]

            # Calculate ATR for dynamic stop loss buffer
            iloc_idx = data.index.get_loc(idx)
            lookback = min(50, iloc_idx)
            recent_data = data.iloc[max(0, iloc_idx - lookback) : iloc_idx + 1]

            atr = self._calculate_atr(recent_data, self.params["atr_period"])
            atr_values = atr.dropna()

            if len(atr_values) == 0:
                logger.warning(f"ATR calculation failed at {idx} - using default 0.1% buffer")
                atr_buffer = entry * 0.001  # Fallback to 0.1%
            else:
                current_atr = atr_values.iloc[-1]
                atr_buffer = current_atr * self.params["atr_stop_buffer"]

            # Stop loss: Below the sweep low (the wick) with ATR buffer
            sweep_low = data.loc[idx, "low"]
            stop_loss = sweep_low - atr_buffer

            # Validate stop makes sense
            if stop_loss >= entry:
                return None

            # Take profit: Previous swing high or min RR (validate RR before creating signal)
            risk = entry - stop_loss
            prior_highs = swing_highs[swing_highs.index < idx].dropna()

            if len(prior_highs) > 0:
                candidate_tp = prior_highs.iloc[-1]

                # Validate RR before using swing high as TP
                if candidate_tp > entry:
                    potential_rr = (candidate_tp - entry) / risk

                    if potential_rr >= self.params["min_rr_ratio"]:
                        take_profit = candidate_tp
                    else:
                        # Swing TP doesn't meet min RR - extend to minimum
                        take_profit = entry + (risk * self.params["min_rr_ratio"])
                else:
                    # Swing high below entry - use min RR instead
                    take_profit = entry + (risk * self.params["min_rr_ratio"])
            else:
                # No prior swing highs - use min RR
                take_profit = entry + (risk * self.params["min_rr_ratio"])

            # Calculate confidence
            confidence = self.calculate_confidence(data, iloc_idx)

            return Signal(
                timestamp=idx,
                direction=1,
                entry_price=entry,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=confidence,
                strategy_name=self.name,
                metadata={"sweep_low": sweep_low, "signal_type": "bullish_sweep"},
            )
        except (KeyError, IndexError, ValueError) as e:
            logger.warning(f"Long signal creation failed at {idx}: {e}")
            return None

    def _create_short_signal(
        self, data: pd.DataFrame, idx: pd.Timestamp, swing_highs: pd.Series, swing_lows: pd.Series
    ) -> Signal | None:
        """
        Create a short signal after bearish liquidity sweep.

        Args:
            data: OHLCV DataFrame
            idx: Timestamp of the signal bar
            swing_highs: Swing high prices
            swing_lows: Swing low prices

        Returns:
            Signal object or None if invalid
        """
        try:
            entry = data.loc[idx, "close"]

            # Calculate ATR for dynamic stop loss buffer
            iloc_idx = data.index.get_loc(idx)
            lookback = min(50, iloc_idx)
            recent_data = data.iloc[max(0, iloc_idx - lookback) : iloc_idx + 1]

            atr = self._calculate_atr(recent_data, self.params["atr_period"])
            atr_values = atr.dropna()

            if len(atr_values) == 0:
                logger.warning(f"ATR calculation failed at {idx} - using default 0.1% buffer")
                atr_buffer = entry * 0.001  # Fallback to 0.1%
            else:
                current_atr = atr_values.iloc[-1]
                atr_buffer = current_atr * self.params["atr_stop_buffer"]

            # Stop loss: Above the sweep high (the wick) with ATR buffer
            sweep_high = data.loc[idx, "high"]
            stop_loss = sweep_high + atr_buffer

            # Validate stop makes sense
            if stop_loss <= entry:
                return None

            # Take profit: Previous swing low or min RR (validate RR before creating signal)
            risk = stop_loss - entry
            prior_lows = swing_lows[swing_lows.index < idx].dropna()

            if len(prior_lows) > 0:
                candidate_tp = prior_lows.iloc[-1]

                # Validate RR before using swing low as TP
                if candidate_tp < entry:
                    potential_rr = (entry - candidate_tp) / risk

                    if potential_rr >= self.params["min_rr_ratio"]:
                        take_profit = candidate_tp
                    else:
                        # Swing TP doesn't meet min RR - extend to minimum
                        take_profit = entry - (risk * self.params["min_rr_ratio"])
                else:
                    # Swing low above entry - use min RR instead
                    take_profit = entry - (risk * self.params["min_rr_ratio"])
            else:
                # No prior swing lows - use min RR
                take_profit = entry - (risk * self.params["min_rr_ratio"])

            # Calculate confidence
            confidence = self.calculate_confidence(data, iloc_idx)

            return Signal(
                timestamp=idx,
                direction=-1,
                entry_price=entry,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=confidence,
                strategy_name=self.name,
                metadata={"sweep_high": sweep_high, "signal_type": "bearish_sweep"},
            )
        except (KeyError, IndexError, ValueError) as e:
            logger.warning(f"Short signal creation failed at {idx}: {e}")
            return None

    def calculate_confidence(self, data: pd.DataFrame, signal_idx: int) -> int:
        """
        Calculate confidence score for a signal.

        Factors:
        - Trend alignment (0-30 points)
        - Volume confirmation (0-20 points)
        - Volatility regime (0-20 points)
        - Pattern clarity (0-30 points)

        Args:
            data: Full OHLCV DataFrame
            signal_idx: Integer position of the signal bar

        Returns:
            Confidence score 0-100
        """
        score = 0

        try:
            # Get recent data for analysis
            lookback = min(50, signal_idx)
            recent_data = data.iloc[max(0, signal_idx - lookback) : signal_idx + 1]

            if len(recent_data) < 10:
                return 50  # Default confidence for insufficient data

            # Determine signal direction from OHLC
            signal_bar = data.iloc[signal_idx]
            signal_direction = 1 if signal_bar["close"] > signal_bar["open"] else -1

            # 1. Trend alignment (0-30 points)
            swing_h, swing_l = find_swing_points(
                recent_data["high"], recent_data["low"], n=min(5, len(recent_data) // 4)
            )
            trend = determine_trend(swing_h, swing_l)

            if len(trend) > 0:
                current_trend = trend.iloc[-1]
                # Only reward if trend aligns with signal direction
                if current_trend != 0 and current_trend == signal_direction:
                    score += 15
                    # Strong trend (consistent)
                    trend_consistency = (trend == current_trend).mean()
                    if trend_consistency > 0.7:
                        score += 15

            # 2. Volume confirmation (0-20 points) - with defensive checks
            if "volume" in recent_data.columns and not recent_data["volume"].isna().all():
                avg_volume = recent_data["volume"].mean()
                current_volume = data.iloc[signal_idx]["volume"]

                if pd.notna(avg_volume) and pd.notna(current_volume) and avg_volume > 0:
                    if current_volume > avg_volume * 1.5:
                        score += 20  # Volume spike confirms
                    elif current_volume > avg_volume:
                        score += 10  # Above average
            else:
                logger.debug("Volume data missing or invalid - skipping volume scoring")

            # 3. Volatility regime (0-20 points) - stability-based scoring
            atr = self._calculate_atr(recent_data, self.params["atr_period"])
            atr_values = atr.dropna()

            if len(atr_values) > 0:
                # Score based on ATR stability, not absolute value
                if len(atr_values) >= 20:
                    atr_std = atr_values.tail(20).std()
                    atr_mean = atr_values.tail(20).mean()

                    if atr_mean > 0:
                        atr_stability = 1 - min(atr_std / atr_mean, 1)  # 0-1 score
                        score += int(atr_stability * 20)  # Stable ATR = predictable
                    else:
                        score += 10  # Default moderate score
                else:
                    score += 10  # Insufficient data for stability calculation

            # 4. Pattern clarity (0-30 points)
            # Clean sweep = immediate reversal
            score += 20  # Base points for detected pattern

            # ========== FIX #27: Reward SMALL body/wick ratio (not large) ==========
            # BEFORE: body/wick > 0.5 = good (WRONG for sweeps!)
            # AFTER: body/wick < 0.3 = good (sweep = long wick, small body = rejection)
            #
            # Good liquidity sweep candle:
            #    |  <- long wick (swept liquidity)
            #   [ ] <- small body (strong rejection)
            #    |
            # Bad candle for sweep:
            #   [====] <- large body (momentum, no rejection)
            current_bar = data.iloc[signal_idx]
            body_size = abs(current_bar["close"] - current_bar["open"])
            wick_size = current_bar["high"] - current_bar["low"]

            if wick_size > 0:
                body_wick_ratio = body_size / wick_size
                if body_wick_ratio < 0.3:
                    score += 10  # Strong wick rejection (ideal sweep)
                elif body_wick_ratio < 0.5:
                    score += 5   # Moderate rejection

        except (KeyError, IndexError, ValueError) as e:
            logger.warning(f"Confidence calculation failed at index {signal_idx}: {e}")
            return 50  # Default on error

        return min(score, 100)


def create_strategy(params: dict | None = None) -> LiquiditySweepStrategy:
    """
    Factory function to create a LiquiditySweepStrategy.

    Args:
        params: Optional strategy parameters

    Returns:
        Configured LiquiditySweepStrategy instance
    """
    return LiquiditySweepStrategy(params)
