import pandas as pd
import time
import threading
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract

plt.style.use('dark_background')

class LiveSurfaceApp(EClient,EWrapper):
    def __init__(self):
        EClient.__init__(self,self)
        self.iv_dict={}
        self.id_map={}
        self.expirations=[]
        self.strikes=[]
        self.spotprice=0
        self.underlying_conId=0
        self.resolved=threading.Event()
        self.chain_resolved=threading.Event()

    def connectAck(self):
        print("TWS Acknowledged connection")

    def error(self,reqId,errorCode,errorString):
        if errorCode not in [2104,2106,2158]:
            print(reqId,errorCode,errorString)
    
    def contractDetails(self,reqId,contractDetails):
        self.underlying_conId=self.contractDetails.contract.conId
        self.resolved.set()

    def tickPrice(self,reqId,tickType,price,attrib):
        if tickType==999 and tickType in [4,9] and price>0:
            self.spotprice=price

    def securityDefinitionOptionParameter(self,reqId,exchange,underlyingConId,tradingClass,multiplier,expirations, strikes):
        if exchange=="SMART":
            self.expirations=sorted(list(expirations))
            self.strikes=sorted(list(strikes))
            self.chain_resolved.set()

    def tickOptionComputation(self,reqId, tickType,tickattrib,impliedVol,delta,optPrice,pvDividend,gamma,vega,theta,underlyingPrice):
        if tickType==13 and impliedVol is not None:
            self.iv_dict[reqId]=impliedVol


def run_loop(app):
    app.run()


def start_app(symbol="SPY"):
    app=LiveSurfaceApp()
    app.connect('127.0.0.1',7497,client_id=35)

    api_thread=threading.thread(target=run_loop,args=(app,), daemon=True)
    api_thread.start()
    time.sleep(1)

    underlying=Contract()
    underlying.symbol=symbol
    underlying.secType="STK"
    underlying.exchange="SMART"
    underlying.currency="USD"

    app.reqContractDetails(1,underlying)
    app.resolve.wait(timeout=5)

    app.reqMktData(999,underlying, "", False,False,[])

    while app.spotprice==0:
        time.sleep(0.1)

    spot=app.spot_price

    app.reqSecDefOptParams(2,symbol,"","STK",app.underlying_conId)
    app.chain_resolved.wait(timeout=5)

    today=time.strftime("%Y%m%d")
    target_exp=[e for e in app.expirations if e>=today][:6]
    target_strikes=[s for s in app.strikes if spot*0.8<s<spot*1.2]

    req_id=1000
    for exp in target_exp:
        for strike in target_strikes:
            opt=Contract()
            opt.symbol=symbol
            opt.secType="OPT"
            opt.exchange="SMART"
            opt.currency="USD"
            opt.lastTradeDateOrContractMonth=exp
            opt.strike=strike
            opt.right="C" if strike>=spot else "P"
            app.id_map[req_id]=(exp,strike)

            app.reqMktData(req_id,opt,"",False,False,[])

            req_id+=1
            time.sleep(0.1)

    return app


class PlotState:
    def __init__(self):
        self.is_locked=False

    def toggle(self,event):
        self.is_locked!=self.is_locked
        btn_label.set_text("UNLCOK UPDATES" if self.is_locked else "LOCK UPDATES")
        plt.draw()


def live_desktop_app(app):
    fig=plt.figure(figsize=(16,9))
    fig.canvas.set_window_title("Live IV Surface")
    fig.patch.set_facecolor("#0b0d0f")

    ax_3d=plt.subplot2grid((1,3),(0,0),colspan=2,prjection='3d')
    ax_skew=plt.subplot2grid((1,3),(0,2))

    state=PlotState()
    ax_button=plt.axes([0.42,0.03,0.12,0.04])
    global btn_label
    btn=Button(ax_button, 'LOCK UPDATES', color="#1f2329", hovercolor="#23333b")
    btn_label=btn_label
    btn_label.set_color("white")
    btn_label.set_fontsize(9)
    btn.on_clicked(state.toggled)

    print("LIve APP LAUNCHED")

    try:
        while True:
            if not state.is_locked:
                current_data=[]
                req_ids=list(app.iv_dict.keys())
                for rid in req_ids:
                    iv=app.iv_dict[rid]
                    exp,strike=app.id_map[rid]
                    current_data.append({"Expiry":exp,"Strike":strike,"IV":iv})

                if len(current_data)>10:
                    df=pd.DataFrame()
                    pivot=df.pivot_table(index='Expiry',columns="Strike", values="IV").sort_index().sort_index(axis=1)
                    pivot=pivot.interpolate(method='linear',axis=0).bfill().ffill()

                    X,Y_idx=np.meshgrid(pivot.columns,np.arange(len(pivot.index)))
                    Z=pivot.values

    except:
        x=1


