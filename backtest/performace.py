import numpy as np
import pandas as pd



def compute_performance(results:pd.DataFrame, risk_free_rate:float=0.25, trading_days:int=252)->pd.Series:

    if len(results) == 0:
        return pd.Series(dtype=float)
    
    daily_returns= results['nav'].pct_change().dropna()
    nav=results['nav'].to_numpy()

    total_days=len(results)


    years=total_days/trading_days

    #Return Metrics

    total_return = (nav[-1] - nav[0])/nav[0]

    cagr=(1+total_return) ** (1/max(years,1e-9))-1 if years > 0 else 0.0

    ann_vol=daily_returns.std() * np.sqrt(trading_days)

    daily_rf=(1+risk_free_rate) ** (1/trading_days)-1

    excess=daily_returns-daily_rf

    sharpe=(excess.mean()/excess.std() * np.sqrt(trading_days)) if excess.std()>0 else 0.0
    downside=daily_returns[daily_returns<0]
    sortino=(excess.mean()/downside.std()*np.sqrt(trading_days)) if len(downside)>0 and downside.std()>0 else 0.0


    # Drawdown Metrics
    dd_series=results['drawdown'].to_numpy()
    max_dd=(dd_series.min())
    calmar=cagr/abs(max_dd) if max_dd < -1e-9 else np.nan
    avg_dd=float(dd_series[dd_series<0].mean()) if np.any(dd_series <0) else np.nan

    #Trade Stats
    trade_stats=_trade_statistics(results)

    #Pnl Attribution
    final_opt_pnl   = float(results["cumulative_option_pnl"].iloc[-1])
    final_hedge_pnl = float(results["cumulative_hedge_pnl"].iloc[-1])
    final_costs     = float(results["cumulative_costs"].iloc[-1])

    gross_pnl = final_hedge_pnl + final_opt_pnl
    cost_drag= final_costs/gross_pnl if abs(gross_pnl)>0 else np.nan
    hedge_efficiency=abs(final_hedge_pnl/final_opt_pnl) if final_opt_pnl!=0 else np.nan

    metrics = pd.Series({
        "total_return_pct":    round(total_return * 100, 2),
        "cagr_pct":            round(cagr * 100, 2),
        "ann_vol_pct":         round(ann_vol * 100, 2),
        "sharpe":              round(sharpe, 3),
        "sortino":             round(sortino, 3),
        "calmar":              round(calmar, 3) if not np.isnan(calmar) else np.nan,
        "max_drawdown_pct":    round(max_dd * 100, 2),
        "avg_drawdown_pct":    round(avg_dd * 100, 2),
        "total_option_pnl":    round(final_opt_pnl, 0),
        "total_hedge_pnl":     round(final_hedge_pnl, 0),
        "total_costs":         round(final_costs, 0),
        "cost_drag_pct":       round(cost_drag * 100, 2) if not np.isnan(cost_drag) else np.nan,
        "hedge_efficiency_x":  round(hedge_efficiency, 2) if not np.isnan(hedge_efficiency) else np.nan,
        **trade_stats,
    })

    return metrics


def _trade_statistics(results:pd.DataFrame)->dict:
    if "has_position" not in results.columns:
        return {}
    
    trades=[]
    entry_nav,entry_date=None,None
    inTrade=False

    pos=results["has_position"].astype(int).to_numpy()

    pos_prev=np.concatenate([[0],pos[:-1]])

    entry_flags=(pos==1) & (pos_prev==0)
    exit_flags=(pos_prev==1) & (pos==0)

    for i in range(len(results)):
        row=results.iloc[i]
        if entry_flags[i]==1:
            inTrade=True
            entry_nav=results.iloc[i]['nav']
            entry_date=results.iloc[i]['date']

        if exit_flags[i]==1:
            inTrade=False
            trades.append({
                "pnl":row['nav']-entry_nav,
                "holding_days":(row['date']-entry_date).days
            })

    if inTrade==True:
        #Forcing exit
        trades.append({
            "pnl":results.iloc[-1]['nav']-entry_nav,
            "holding_days":(results.iloc[-1]['date']-entry_date).days
        })    

    if not trades:
        return {
            "num_trades":       0,
            "win_rate_pct":     np.nan,
            "avg_win":          np.nan,
            "avg_loss":         np.nan,
            "profit_factor":    np.nan,
            "avg_holding_days": np.nan,
        }

    trades_df=pd.DataFrame(trades)

    wins=trades_df[trades_df['pnl']>0]['pnl']
    losses=trades_df[trades_df['pnl']<=0]['pnl']

    win_rate=len(wins)/len(trades_df) if len(trades_df)>0 else 0.0
    gross_wins=wins.sum()
    gross_losses=abs(losses.sum())
    pf=gross_wins/gross_losses if gross_losses>0 else np.nan

    return {
        "num_trades":        len(trades_df),
        "win_rate_pct":      round(win_rate * 100, 1),
        "avg_win":           round(float(wins.mean()) if len(wins) > 0 else 0, 0),
        "avg_loss":          round(float(losses.mean()) if len(losses) > 0 else 0, 0),
        "profit_factor":     round(pf, 2) if not np.isnan(pf) else np.nan,
        "avg_holding_days":  round(float(trades_df["holding_days"].mean()), 1),
    }