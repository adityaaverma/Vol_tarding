"""
Elite Volatility Trading Signal Generation
============================================

Multi-factor signal engine with:
  - IV vs RV mean reversion signals
  - Skew & term structure analysis
  - Regime detection (trending vs mean-reverting)
  - Multi-timeframe confirmation
  - Signal confidence scoring
  - Anomaly detection
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
from enum import Enum
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# ENUMS & DATA CLASSES
# ============================================================================

class SignalType(Enum):
    """Trading signal types"""
    LONG_STRADDLE = "long_straddle"      # IV too low vs RV
    SHORT_STRADDLE = "short_straddle"    # IV too high vs RV
    NEUTRAL = "neutral"                   # No clear signal
    FADE_VOLATILITY = "fade_volatility"  # Counter IV spike
    CHASE_VOLATILITY = "chase_volatility" # Catch IV crash


class Regime(Enum):
    """Market regimes for context"""
    MEAN_REVERT = "mean_revert"  # IV reverts to RV
    TRENDING = "trending"         # Vol trends persist
    CRISIS = "crisis"             # High volatility spike
    QUIET = "quiet"               # Low vol regime


@dataclass
class Signal:
    """Signal output structure"""
    type: SignalType
    strength: float              # 0-100, confidence
    iv_rv_spread: float          # IV - RV (annualized %)
    regime: Regime
    components: Dict[str, float] # breakdown of signal sources
    timestamp: pd.Timestamp
    reasons: List[str]           # human-readable explanation


@dataclass
class VolatilityProfile:
    """Current volatility snapshot"""
    current_iv: float
    current_rv: float
    iv_percentile: float
    rv_percentile: float
    skew: float
    term_structure_slope: float
    vol_of_vol: float


# ============================================================================
# CORE SIGNAL ENGINE
# ============================================================================

class VolatilitySignalEngine:
    """
    Elite signal generation with multi-factor analysis
    """

    def __init__(
        self,
        iv_window: int = 20,           # days for IV percentile
        rv_window: int = 20,           # days for RV percentile
        regime_lookback: int = 60,     # days for regime detection
        percentile_thresholds: Tuple[float, float] = (25, 75),  # bullish/bearish
        min_data_points: int = 50,
        confidence_decay: float = 0.95  # decay older signals
    ):
        self.iv_window = iv_window
        self.rv_window = rv_window
        self.regime_lookback = regime_lookback
        self.percentile_thresholds = percentile_thresholds
        self.min_data_points = min_data_points
        self.confidence_decay = confidence_decay

        # Internal state
        self.iv_history = []
        self.rv_history = []
        self.prices = []
        self.timestamps = []

    # ========================================================================
    # PRIMARY SIGNAL GENERATION
    # ========================================================================

    def generate_signal(
        self,
        iv_surface: pd.DataFrame,      # ATM IV time series
        realized_vol: pd.DataFrame,     # RV time series
        prices: pd.Series,              # underlying prices
        use_skew: bool = True,
        use_term_structure: bool = True,
    ) -> Signal:
        """
        Generate elite volatility trading signal with multi-factor analysis.

        Args:
            iv_surface: DataFrame with columns [timestamp, atm_iv, ...]
            realized_vol: DataFrame with columns [timestamp, rv, ...]
            prices: Series of underlying prices (daily closes)
            use_skew: Include skew analysis in signal
            use_term_structure: Include term structure slope

        Returns:
            Signal object with type, strength, and breakdown
        """

        # Validate inputs
        if len(iv_surface) < self.min_data_points or len(realized_vol) < self.min_data_points:
            raise ValueError(f"Insufficient data: need {self.min_data_points} points")

        # Update internal state
        self.iv_history = iv_surface['atm_iv'].values
        self.rv_history = realized_vol['rv'].values
        self.prices = prices.values
        self.timestamps = prices.index

        # Get current volatility snapshot
        vol_profile = self._get_volatility_profile()

        # Build signal components
        components = {}
        reasons = []

        # 1. Core IV vs RV spread signal
        spread_signal, spread_score = self._iv_rv_spread_signal(vol_profile)
        components['iv_rv_spread'] = spread_score
        reasons.extend(spread_signal['reasons'])

        # 2. Mean reversion score
        mean_revert_score = self._mean_reversion_score(vol_profile)
        components['mean_reversion'] = mean_revert_score
        if mean_revert_score > 60:
            reasons.append(f"Strong mean reversion signal ({mean_revert_score:.0f})")

        # 3. Regime analysis
        regime = self._detect_regime()
        components['regime_score'] = self._regime_score(regime)

        # 4. Optional: Skew analysis
        if use_skew:
            skew_score = self._skew_signal(vol_profile)
            components['skew'] = skew_score
            if abs(skew_score) > 30:
                reasons.append(f"Skew bias: {skew_score:+.0f}")

        # 5. Optional: Term structure
        if use_term_structure:
            term_score = self._term_structure_signal(vol_profile)
            components['term_structure'] = term_score

        # 6. Multi-timeframe confirmation
        confirmation_score = self._multi_timeframe_confirmation()
        components['multi_tf_confirmation'] = confirmation_score

        # 7. Anomaly detection (recent regime changes)
        anomaly_penalty = self._anomaly_penalty()
        components['anomaly_penalty'] = anomaly_penalty

        # ====================================================================
        # AGGREGATE SIGNAL
        # ====================================================================

        # Calculate final signal type and strength
        net_score = (
            components['iv_rv_spread'] * 0.40 +          # Core signal
            components['mean_reversion'] * 0.25 +         # Mean reversion
            components['regime_score'] * 0.15 +           # Regime context
            confirmation_score * 0.10 +                   # Confirmation
            anomaly_penalty * 0.10                        # Anomaly check
        )

        # If skew available, reweight
        if use_skew:
            net_score = net_score * 0.95 + components['skew'] * 0.05

        # Determine signal type
        signal_type = self._classify_signal(net_score, regime, vol_profile)

        # Confidence/strength (0-100)
        strength = self._calculate_strength(
            net_score, vol_profile, regime, confirmation_score
        )

        return Signal(
            type=signal_type,
            strength=strength,
            iv_rv_spread=vol_profile.current_iv - vol_profile.current_rv,
            regime=regime,
            components=components,
            timestamp=self.timestamps[-1],
            reasons=reasons
        )

    # ========================================================================
    # SIGNAL COMPONENTS
    # ========================================================================

    def _iv_rv_spread_signal(self, vol_profile: VolatilityProfile) -> Tuple[Dict, float]:
        """
        Core signal: IV vs RV mean reversion.

        Returns:
            (signal_dict, score 0-100)
        """
        signal = {'reasons': []}

        iv_percentile = vol_profile.iv_percentile
        rv_percentile = vol_profile.rv_percentile
        spread = vol_profile.current_iv - vol_profile.current_rv

        # Extreme IV levels vs RV
        if iv_percentile > 90:
            signal['reasons'].append(f"IV at {iv_percentile:.0f}th percentile (extreme high)")
            score = -80  # Short bias
        elif iv_percentile < 10:
            signal['reasons'].append(f"IV at {iv_percentile:.0f}th percentile (extreme low)")
            score = +80  # Long bias
        elif iv_percentile > 75:
            signal['reasons'].append(f"IV elevated ({iv_percentile:.0f}th percentile)")
            score = -40  # Moderate short
        elif iv_percentile < 25:
            signal['reasons'].append(f"IV depressed ({iv_percentile:.0f}th percentile)")
            score = +40  # Moderate long
        else:
            signal['reasons'].append("IV in normal range")
            score = 0

        # Normalize spread to -100 to +100
        spread_std = np.std(self.iv_history[-self.iv_window:] - self.rv_history[-self.rv_window:])
        if spread_std > 0:
            spread_score = np.clip((spread / spread_std) * 50, -100, 100)
        else:
            spread_score = 0

        final_score = (score + spread_score) / 2
        signal['reasons'].append(f"IV-RV spread: {spread:+.2f}%")

        return signal, final_score

    def _mean_reversion_score(self, vol_profile: VolatilityProfile) -> float:
        """
        Score for mean reversion probability.
        High score = vol likely to revert.
        """
        if len(self.iv_history) < 10:
            return 0

        # 1. IV extremes mean-revert faster
        iv_percentile = vol_profile.iv_percentile
        extremeness = max(
            abs(iv_percentile - 50) - 25,  # threshold at 25/75
            0
        ) / 25
        extremeness_score = min(extremeness * 100, 100)

        # 2. Auto-correlation of IV changes (mean-reverting = negative corr)
        iv_changes = np.diff(self.iv_history[-20:])
        if len(iv_changes) > 1:
            autocorr = np.corrcoef(iv_changes[:-1], iv_changes[1:])[0, 1]
            autocorr_score = max(-autocorr * 50, 0)  # Negative corr = mean revert
        else:
            autocorr_score = 0

        # 3. IV-RV gap size (larger gap = more likely to revert)
        spread = vol_profile.current_iv - vol_profile.current_rv
        spread_std = np.std(self.iv_history[-self.iv_window:] - self.rv_history[-self.rv_window:])
        if spread_std > 0:
            gap_score = min(abs(spread) / spread_std * 30, 100)
        else:
            gap_score = 0

        # Weighted mean reversion score
        mr_score = (
            extremeness_score * 0.40 +
            autocorr_score * 0.30 +
            gap_score * 0.30
        )

        return mr_score

    def _detect_regime(self) -> Regime:
        """
        Detect current market regime for context.
        """
        if len(self.iv_history) < self.regime_lookback:
            return Regime.MEAN_REVERT

        recent_iv = self.iv_history[-self.regime_lookback:]
        current_iv = self.iv_history[-1]

        # Crisis regime: IV spike
        if current_iv > np.percentile(recent_iv, 90):
            return Regime.CRISIS

        # Quiet regime: Low volatility
        if current_iv < np.percentile(recent_iv, 25):
            return Regime.QUIET

        # Check for trending behavior
        iv_trend = np.polyfit(range(len(recent_iv)), recent_iv, 1)[0]
        rv_trend = np.polyfit(range(len(self.rv_history[-self.regime_lookback:])),
                              self.rv_history[-self.regime_lookback:], 1)[0]

        if abs(iv_trend) > np.std(recent_iv) * 0.1:  # Vol trending
            return Regime.TRENDING
        else:
            return Regime.MEAN_REVERT

    def _regime_score(self, regime: Regime) -> float:
        """Convert regime to signal score contribution"""
        regime_scores = {
            Regime.MEAN_REVERT: 50,    # Good for options
            Regime.TRENDING: -30,       # Bad for short vol
            Regime.CRISIS: -80,         # Avoid selling vol
            Regime.QUIET: 70,           # Best for selling vol
        }
        return regime_scores.get(regime, 0)

    def _skew_signal(self, vol_profile: VolatilityProfile) -> float:
        """
        Skew analysis: positive skew = downside protection = lower IV.
        Negative skew = crash risk = higher IV.
        """
        skew = vol_profile.skew

        if skew < -0.5:
            return -60  # High crash risk, avoid selling
        elif skew < -0.2:
            return -30
        elif skew > 0.5:
            return 40   # High skew, IV too cheap
        elif skew > 0.2:
            return 20
        else:
            return 0

    def _term_structure_signal(self, vol_profile: VolatilityProfile) -> float:
        """
        Term structure slope: contango vs backwardation.
        Steep contango (upward slope) = short vol attractive.
        Backwardation (downward slope) = avoid short vol.
        """
        slope = vol_profile.term_structure_slope

        if slope > 0.02:  # Strong contango
            return -40  # Short vol is attractive
        elif slope > 0.01:
            return -20
        elif slope < -0.01:  # Backwardation
            return 40   # Avoid short vol
        else:
            return 0

    def _multi_timeframe_confirmation(self) -> float:
        """
        Check if signal is confirmed across multiple timeframes.
        Returns 0-100 confidence score.
        """
        if len(self.iv_history) < 60:
            return 50

        # 20-day signal
        signal_20d = np.percentile(self.iv_history[-20:], 50)
        # 40-day signal
        signal_40d = np.percentile(self.iv_history[-40:], 50)
        # 60-day signal
        signal_60d = np.percentile(self.iv_history[-60:], 50)

        # If all trending same direction, high confirmation
        curr = self.iv_history[-1]
        if (curr < signal_20d < signal_40d < signal_60d) or \
           (curr > signal_20d > signal_40d > signal_60d):
            return 100
        elif (curr < signal_20d and signal_40d < signal_60d) or \
             (curr > signal_20d and signal_40d > signal_60d):
            return 70
        else:
            return 40

    def _anomaly_penalty(self) -> float:
        """
        Detect recent anomalies/regime changes that reduce confidence.
        Returns penalty (0 = no penalty, -50 = reduce confidence).
        """
        if len(self.iv_history) < 20:
            return 0

        recent = self.iv_history[-5:]
        older = self.iv_history[-20:-5]

        # Sudden spike/drop
        recent_volatility = np.std(recent)
        older_volatility = np.std(older)

        if recent_volatility > older_volatility * 2:
            return -30  # High uncertainty after spike
        elif recent_volatility > older_volatility * 1.5:
            return -15
        else:
            return 0

    def _classify_signal(
        self,
        net_score: float,
        regime: Regime,
        vol_profile: VolatilityProfile
    ) -> SignalType:
        """
        Classify final signal type based on score and context.
        """
        # Strong thresholds
        if net_score > 50 and regime != Regime.TRENDING:
            return SignalType.LONG_STRADDLE
        elif net_score < -50 and regime == Regime.QUIET:
            return SignalType.SHORT_STRADDLE

        # Moderate signals
        elif net_score > 30:
            return SignalType.LONG_STRADDLE
        elif net_score < -30:
            return SignalType.SHORT_STRADDLE

        # Fade/chase logic
        elif vol_profile.iv_percentile > 95 and regime != Regime.CRISIS:
            return SignalType.FADE_VOLATILITY
        elif vol_profile.iv_percentile < 5 and regime != Regime.QUIET:
            return SignalType.CHASE_VOLATILITY

        else:
            return SignalType.NEUTRAL

    def _calculate_strength(
        self,
        net_score: float,
        vol_profile: VolatilityProfile,
        regime: Regime,
        confirmation: float
    ) -> float:
        """
        Convert net score to 0-100 strength/confidence.
        """
        # Base strength from net score
        strength = np.clip(abs(net_score), 0, 100)

        # Boost if extreme percentiles
        if vol_profile.iv_percentile < 10 or vol_profile.iv_percentile > 90:
            strength = strength * 1.1
        elif vol_profile.iv_percentile < 25 or vol_profile.iv_percentile > 75:
            strength = strength * 1.05

        # Boost if multi-timeframe confirmed
        strength = strength * 0.7 + confirmation * 0.3

        # Reduce in crisis regime
        if regime == Regime.CRISIS:
            strength = strength * 0.8

        return np.clip(strength, 0, 100)

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def _get_volatility_profile(self) -> VolatilityProfile:
        """Get current volatility snapshot"""
        current_iv = self.iv_history[-1]
        current_rv = self.rv_history[-1]

        # Percentiles
        iv_percentile = (self.iv_history <= current_iv).sum() / len(self.iv_history) * 100
        rv_percentile = (self.rv_history <= current_rv).sum() / len(self.rv_history) * 100

        # Placeholder for skew & term structure (integrate with your IV surface)
        skew = self._estimate_skew()
        term_slope = self._estimate_term_structure_slope()
        vol_of_vol = np.std(np.diff(self.iv_history[-20:]))

        return VolatilityProfile(
            current_iv=current_iv,
            current_rv=current_rv,
            iv_percentile=iv_percentile,
            rv_percentile=rv_percentile,
            skew=skew,
            term_structure_slope=term_slope,
            vol_of_vol=vol_of_vol,
        )

    def _estimate_skew(self) -> float:
        """
        Estimate skew from IV surface if available.
        Placeholder: returns 0. Replace with your skew calculation.
        """
        # TODO: Integrate with vol/iv_surface.py
        return 0.0

    def _estimate_term_structure_slope(self) -> float:
        """
        Estimate term structure slope from IV surface.
        Placeholder: returns 0. Replace with your IV surface term structure.
        """
        # TODO: Integrate with vol/iv_surface.py
        return 0.0

    def signal_to_dict(self, signal: Signal) -> Dict:
        """Convert Signal to dictionary for storage/logging"""
        return {
            'timestamp': signal.timestamp,
            'signal_type': signal.type.value,
            'strength': signal.strength,
            'iv_rv_spread': signal.iv_rv_spread,
            'regime': signal.regime.value,
            'components': signal.components,
            'reasons': ' | '.join(signal.reasons),
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_signal_engine(config: Optional[Dict] = None) -> VolatilitySignalEngine:
    """Factory function to create signal engine with defaults or config"""
    if config is None:
        config = {}

    return VolatilitySignalEngine(
        iv_window=config.get('iv_window', 20),
        rv_window=config.get('rv_window', 20),
        regime_lookback=config.get('regime_lookback', 60),
        percentile_thresholds=config.get('percentile_thresholds', (25, 75)),
    )


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Example: Generate signal with sample data.
    Replace with your yfinance data pipeline.
    """

    # Create sample IV and RV data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    iv_data = pd.DataFrame({
        'timestamp': dates,
        'atm_iv': np.random.uniform(15, 35, 100),  # Your IV from surface
    })
    rv_data = pd.DataFrame({
        'timestamp': dates,
        'rv': np.random.uniform(12, 30, 100),  # Your realized vol
    })
    prices = pd.Series(
        np.cumsum(np.random.randn(100) * 0.5) + 100,
        index=dates
    )

    # Generate signal
    engine = create_signal_engine()
    signal = engine.generate_signal(iv_data, rv_data, prices)

    print(f"\n{'='*70}")
    print(f"VOLATILITY TRADING SIGNAL")
    print(f"{'='*70}")
    print(f"Signal Type:       {signal.type.value}")
    print(f"Strength:          {signal.strength:.1f} / 100")
    print(f"IV-RV Spread:      {signal.iv_rv_spread:+.2f}%")
    print(f"Regime:            {signal.regime.value}")
    print(f"\nComponent Breakdown:")
    for key, value in signal.components.items():
        print(f"  {key:.<40} {value:>8.2f}")
    print(f"\nSignal Drivers:")
    for reason in signal.reasons:
        print(f"  • {reason}")
    print(f"{'='*70}\n")
