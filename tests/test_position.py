import pandas as pd
import pytest

from strategy.position import StraddlePosition


def test_mark_to_market_updates_greeks_from_series_row():
    pos = StraddlePosition(
        ticker="SPY",
        entry_date=pd.Timestamp("2024-01-01"),
        expiry=pd.Timestamp("2024-01-05"),
        strike=500.0,
        side=1,
        quantity=1.0,
        entry_price=10.0,
        entry_spot=500.0,
    )
    pos.current_iv = 0.2

    row = pd.Series(
        {
            "c_bid": 1.0,
            "c_ask": 3.0,
            "p_bid": 1.0,
            "p_ask": 3.0,
            "underlying_last": 505.0,
            "C_DELTA": 0.25,
            "P_DELTA": 0.15,
            "C_VEGA": 2.0,
            "P_VEGA": 3.0,
            "C_THETA": -1.0,
            "P_THETA": -0.5,
            "C_GAMMA": 0.1,
            "P_GAMMA": 0.2,
            "c_iv": 0.2,
            "p_iv": 0.2,
        }
    )

    pnl = pos.mark_to_market(row)

    assert pnl == pytest.approx(-600.0)
    assert pos.delta == pytest.approx(0.4)
    assert pos.vega == pytest.approx(5.0)
    assert pos.theta == pytest.approx(-1.5)
    assert pos.gamma == pytest.approx(0.3)
    assert pos.current_price == pytest.approx(4.0)
