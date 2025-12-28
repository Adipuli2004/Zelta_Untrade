import pandas as pd
import numpy as np
import warnings
import sys
sys.path.append("../")
sys.path.append("./")

from backtest_engine import perform_backtest
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None


def process_data(data:pd.DataFrame):

    data['datetime'] = pd.to_datetime(data['datetime'])
    data['pred_buy'] = data['pred_close']>=data['open']*1.005
    data['pred_sell']=data['pred_close']<=data['open']*0.995
    data['pred_buy']=data['pred_buy'].apply(lambda x: 1 if x else 0)
    data['pred_sell']=data['pred_sell'].apply(lambda x: -1 if x else 0)
    data['trade']=data['pred_buy']+data['pred_sell'] 
    return data

def strat(data:pd.DataFrame, data_3m: pd.DataFrame):

    trade_data = data[data['trade']!=0]
    data_3m['datetime'] = pd.to_datetime(data_3m['datetime'])
    data_3m.set_index('datetime', drop=True, inplace=True)

    #transposing trades from hourly data to 3minute data for better volatility tracking
    for i in range(len(trade_data)):
        data_3m['signals'].loc[trade_data['datetime'].iloc[i]] = trade_data['trade'].iloc[i]
        data_3m['signals'].loc[trade_data['datetime'].iloc[i]+pd.DateOffset(minutes=57)] = -trade_data['trade'].iloc[i]

    data_3m.reset_index(inplace=True)
    logs = data_3m[['datetime','signals','open','high','low','close','volume']]
    return logs


def main():
    data = pd.read_csv("deep_learning_approach/predicted_data.csv").dropna()
    data_3m = pd.read_csv("deep_learning_approach/data/btcusdt_3m_zelta.csv")
    data = process_data(data=data)
    final_logs = strat(data=data, data_3m=data_3m)
    final_logs.to_csv("deep_learning_approach/final_logs.csv")
    perform_backtest("deep_learning_approach/final_logs.csv") 

    