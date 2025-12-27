import pandas_ta as ta
import talib as pta
import pandas as pd
import numpy as np
import warnings
import os
import sys
sys.path.append("../")
sys.path.append("./")
from backtest_engine import perform_backtest
from scipy.signal import cheby1, filtfilt
warnings.filterwarnings('ignore')


def process_data(data: pd.DataFrame):
    """
    Function to process and compute all necessary indicators and columns required to generate trade signals

    Args: 
        data (pandas dataframe) : input data with OHLCV columns

    Returns:
        pandas dataframe : data frame with necessary columns appended
    """

    # Heiken Ashi Calculations
    data['Heiken_Close'] = (data['open'] + data['close'] + data['high'] + data['low']) / 4
    data['Heiken_Open'] = data['open']
    for i in range(1, len(data)+1):
        data['Heiken_Open'][i] = (data['Heiken_Open'][i-1] + data['Heiken_Close'][i-1]) / 2
    data['Heiken_High'] = data[['high', 'Heiken_Open', 'Heiken_Close']].max(axis=1)
    data['Heiken_Low'] = data[['low', 'Heiken_Open', 'Heiken_Close']].min(axis=1)
    
    # Exponential Moving Averages (EMAs)
    data["EMA20"] = ta.ema(data['close'], length=20)
    data["EMA12"] = ta.ema(data['close'], length=12)
    data["EMA26"] = ta.ema(data['close'], length=26)
    
    #MACD
    data['MACD'] = data['EMA12'] - data['EMA26']
    data["MACD_Signal_Line"] = ta.ema(data['MACD'], length=9)
    
    #Chaikin Volatility
    High_Low_Range = data['high'] - data['low']
    data['Chaikin_volatility'] = ta.ema(High_Low_Range, timeperiod=50).pct_change(periods=50) * 100
    
    # directional movement index
    data['Plus_DI_s'] = pta.PLUS_DI(data['high'], data['low'], data['close'], timeperiod=63)#importing talib as pta for specifically PLUS_DI
    data['Minus_DI_s'] = pta.MINUS_DI(data['high'], data['low'], data['close'], timeperiod=63)
    
    # Chande Momentum Oscillator
    Change = data['close'].diff()
    Gain = Change.apply(lambda x: x if x > 0 else 0)
    Loss = Change.apply(lambda x: -x if x < 0 else 0)    
    Sum_Gain = Gain.rolling(window=14).sum()
    Sum_Loss = Loss.rolling(window=14).sum()
    data['CMO'] = 100 * ((Sum_Gain - Sum_Loss) / (Sum_Gain + Sum_Loss))
    
    # Donchian Channel
    data['Donchian_High'] = data['high'].rolling(window=20).max()
    data['Donchian_Low'] = data['low'].rolling(window=20).min()
    data['Donchian_Middle'] = (data['Donchian_High'] + data['Donchian_Low']) / 2
    
    # Aroon Oscillator
    data['Aroon_Oscillator'] = pta.AROONOSC(data['high'], data['low'], timeperiod=25)
    
    # Commodity Channel Index
    data['CCI'] = pta.CCI(data['high'], data['low'], data['close'], timeperiod=20)
    
    # Calculate the rolling threshold for pct_change_2
    data['pct_change_2'] = data['close'].pct_change(2)
    data['rolling_threshold'] = data['pct_change_2'].rolling(10).quantile(0.9)
    
    # Relative Strength Index
    data['RSI'] = ta.rsi(data['close'], timeperiod=14)
    # Smoothened RSI
    data['RSI_smoothed'] = data['RSI'].rolling(window=14).mean()
    
    # Chebyshev_filter
    order=2
    ripple=0.03
    cutoff=0.1
    b, a = cheby1(N=order, rp=ripple, Wn=cutoff, btype='low', analog=False)
    data['Chebyshev_Filtered']=filtfilt(b, a, data['close'])

    return data

def strat(data: pd.DataFrame):
    """
    Defines the trade signal generation strategy

    Args:
        data (pandas dataframe) : data frame with required columns to build trade signals

    Returns:
        pandas dataframe :  Dataframe with OHLCV data and signals and trade type
    """

    flag = 0  # 0 = no trade, 1 = long trade active, -1 = short trade active
    data['signals'] = 0
    data['trade_type'] = [''] * len(data)
    
    time_calculator = 0  # To track time in active position
    timeout_flag = False  # To ensure stricter rules apply after timeout
    strict_conditions_applied = False  # Flag to apply stricter rules
    
    data['time_calculator'] = 0
    
    for i in range(len(data)):

        if pd.isna(data.iloc[i]).sum()>0: #disregarding rows with null values
            continue
            
        # Skip new trade entry checks if a trade is already active
        if flag != 0:
            time_calculator += 1
            data.loc[i, 'time_calculator'] = time_calculator
    
            # Long Exit or Timeout Exit
            if  (flag == 1 
                 and ((data['EMA12'][i]      < data['EMA26'][i] 
                 and data['CMO'][i-1]        >= 20 
                 and data['Heiken_Open'][i]  > data['EMA20'][i] 
                 and data['Heiken_Close'][i] < data['EMA20'][i]) 
                 or time_calculator        >= 400)):
                
                data.loc[i, "signals"] = -1
                flag = 0
                timeout_flag = time_calculator >= 400
                data.loc[i, 'trade_type'] = "close"
    
            # Short Exit or Timeout Exit
            elif (flag == -1 
                   and ((data['CCI'][i]            > -20 
                   and data['Aroon_Oscillator'][i] > -50 
                   and data['CMO'][i-1]            <= 30 
                   and data['close'][i]            > data['Donchian_Middle'][i-1] 
                   and data['pct_change_2'][i]     <= data['rolling_threshold'][i]) 
                   or time_calculator            >= 400)):
                
                data.loc[i, "signals"] = 1
                flag = 0
                timeout_flag = time_calculator >= 400
                data.loc[i, 'trade_type'] = "close"
    
            # Reset stricter conditions after timeout exit
            if timeout_flag:
                strict_conditions_applied = True
                timeout_flag = False
    
            # Skip further processing for this iteration
            continue
    
        # Reset time_calculator if no trade is active
        time_calculator = 0
    
        # Long Entry
        if (flag == 0 
            and not strict_conditions_applied 
            and data['EMA12'][i]               > data['EMA26'][i] 
            and data['Heiken_Open'][i]         < data['EMA20'][i] 
            and data['Chaikin_volatility'][i]  < 55 
            and data['Heiken_Close'][i]        > data['EMA20'][i]):
            
            data.loc[i, "signals"] = 1
            flag = 1
            data.loc[i, 'trade_type'] = "long"
    
        # Short Entry
        elif (flag == 0 
              and not strict_conditions_applied 
              and data['Chebyshev_Filtered'][i]   < data['close'][i] 
              and data['Minus_DI_s'][i]           > 0 
              and data['EMA12'][i]                < data['EMA26'][i] 
              and data['Minus_DI_s'][i]           > data['Plus_DI_s'][i]):
            
            data.loc[i, "signals"] = -1
            flag = -1
            data.loc[i, 'trade_type'] = "short"
    
        # Stricter Long Entry
        elif (flag == 0 
              and strict_conditions_applied
              and data['EMA12'][i]              > data['EMA26'][i] 
              and data['Heiken_Open'][i]        < data['EMA20'][i] 
              and data['Chaikin_volatility'][i] < 50 
              and data['Heiken_Close'][i]       > data['EMA20'][i] #strict conditions
              and data['RSI_smoothed'][i]       > 60):           # Additional strict condition
            
            data.loc[i, "signals"] = 1
            flag = 1
            data.loc[i, 'trade_type'] = "long"
    
        # Stricter Short Entry
        elif (flag == 0 
              and strict_conditions_applied 
              and data['Chebyshev_Filtered'][i] < data['close'][i] 
              and data['EMA12'][i]              < data['EMA26'][i] 
              and data['Minus_DI_s'][i]          > data['Plus_DI_s'][i] 
              and data['RSI_smoothed'][i]          < 30):  # Additional strict condition
            
            data.loc[i, "signals"] = -1
            flag = -1
            data.loc[i, 'trade_type'] = "short"
    
        
        if(strict_conditions_applied):
            strict_conditions_applied=False #Resetting flag to check for normal conditions

    if flag == -1:
        data.loc[i, "signals"] = 1
        flag = 0
        data.loc[i, 'trade_type'] = "close"
    
    elif flag == 1:
        data.loc[i, "signals"] = -1
        flag = 0
        data.loc[i, 'trade_type'] = "close"

    signals = data[['datetime','open','high','low','close','volume','signals','trade_type']]
    return signals

#-------------- Main ------------------#
def main():
    data=pd.read_csv("technial_indicators_approach/eth_data/ETHUSDT_6h.csv")                   
    data=process_data(data)                                       
    data=strat(data)                                              
    data.to_csv('technial_indicators_approach/final_logs_eth.csv',index=False)                
    perform_backtest('technial_indicators_approach/final_logs_eth.csv')                       
    
if __name__=='__main__':
    main()
