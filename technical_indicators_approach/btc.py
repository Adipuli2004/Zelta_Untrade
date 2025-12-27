import pandas as pd
import numpy as np
import talib as pta
import pandas_ta as ta
import os 
import warnings
import sys
sys.path.append("../")
sys.path.append("./")
from backtest_engine import perform_backtest
from scipy.signal import cheby1, filtfilt
warnings.filterwarnings('ignore')

class KalmanFilter:
    """
    Class based implementation of Kalman Filtering"
    """
    def __init__(self, Q : float = 1e-7, R : float = 1e-2):
        """
        Class constructor for KalmanFilter class
        Args:
            Q (float) : Process Variance
            R (float) : Measurement Variance
        """
        self.Q = Q
        self.R = R
        self.P = 1.0  # Initial estimate error covariance
        self.X = 0.0  # Initial state estimate

    def predict(self):
        """
        Computes error covariance and returns the next predicted state according to current state and system model

        Returns
            float :  Predicted state 
        """
        self.P = self.P + self.Q 
        return self.X

    def update(self, measurement : float):
        """
        Updates the next state estimation (X) and Error Covariance

        Args: 
            measurement (float) :  Measured value of observation
        """
        K = self.P / (self.P + self.R)
        self.X = self.X + K * (measurement - self.X)
        self.P = (1 - K) * self.P


def hawkes_process(data: pd.Series, kappa: float):
    """
    Function to implement hawkes process on the given data
    Args:
        data (pandas series) : Series of containing data points
        kappa (float) : Excitation parameter

    Returns:
        pandas series : 
    """
    assert kappa > 0.0 , "Kappa should be non-negative"

    alpha = np.exp(-kappa)
    arr = data.to_numpy()
    output = np.zeros(len(data))
    output[:] = np.nan
    
    for i in range(1, len(data)):
        if np.isnan(output[i - 1]):
            output[i] = arr[i]
        else:
            output[i] = output[i - 1] * alpha + arr[i]
    
    return pd.Series(output, index=data.index) * kappa
    

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
    
    # Exponential Moving Averages
    data["EMA6"] = ta.ema(data['close'], length=6)
    data["EMA10"] = ta.ema(data['close'], length=10)
    data["EMA20"] = ta.ema(data['close'], length=20)
    data["EMA50"] = ta.ema(data['close'], length=50)
    
    # Moving Average Convergence Divergence
    EMA12=ta.ema(data['close'], length=12)
    EMA26=ta.ema(data['close'], length=26)
    
    data['MACD'] = EMA12 - EMA26
    data["MACD_Signal_Line"] = ta.ema(data['MACD'], length=9)
    
    # Relative Strength Index
    RSI = ta.rsi(data['close'], timeperiod=14)
    #Smoothened RSI
    data['RSI_smoothed'] = RSI.rolling(window=5).mean()
    
    # Chaikin Volatility
    High_Low_Range = data['high'] - data['low']
    data['Chaikin_volatility'] = ta.ema(High_Low_Range, timeperiod=10).pct_change(periods=10) * 100
    
    # Know Sure Thing(KST)
    roc1 = ta.roc(data['close'], timeperiod=10)
    roc2 = ta.roc(data['close'], timeperiod=15)
    roc3 = ta.roc(data['close'], timeperiod=20)
    roc4 = ta.roc(data['close'], timeperiod=30)
    
    data['KST'] = (
        ta.sma(roc1, timeperiod=10) +
        2 * ta.sma(roc2, timeperiod=10) +
        3 * ta.sma(roc3, timeperiod=10) +
        4 * ta.sma(roc4, timeperiod=15))
    data['KST_signal'] = ta.sma(data['KST'], timeperiod=9)  # KST Signal Line
    
    # Apply the Hawkes process
    data['Hawkes_close'] = hawkes_process(data['close'], kappa=0.06)
    
    # Elder Ray Index
    EMA13 = ta.ema(data['close'], length=13)
    data['Bull_Power'] = data['high'] - EMA13
    data['Bear_Power'] = data['low'] - EMA13
    
    # Calculate Directional Movement Index (DMI) components: +DI and -DI
    data['Plus_DI'] = pta.PLUS_DI(data['high'], data['low'], data['close'], timeperiod=14)
    data['Minus_DI'] = pta.MINUS_DI(data['high'], data['low'], data['close'], timeperiod=14)
    data['Plus_DI_s'] = pta.PLUS_DI(data['high'], data['low'], data['close'], timeperiod=63)
    data['Minus_DI_s'] = pta.MINUS_DI(data['high'], data['low'], data['close'], timeperiod=63)

    # Aroon Oscillator
    data['Aroon_Oscillator'] = pta.AROONOSC(data['high'], data['low'], timeperiod=25)
    
    # Calculate the rolling threshold for pct_change_5
    data['pct_change_5'] = data['close'].pct_change(5)
    data['rolling_threshold'] = data['pct_change_5'].rolling(10).quantile(0.9)
    
    # Kalman filtering
    filtered_price=[]
    kf=KalmanFilter() #filter intialisation
    for i in range(len(data)):
        kf.update(data['close'].iloc[i])
        filtered_price.append(kf.predict())
    data['filtered_price']=filtered_price

    # Chebyshev filter
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
    data['signals']=0
    data['trade_type'] = ['']*len(data)
    
    # Iterate over rows to evaluate conditions
    for i in range(len(data)):
        
        if pd.isna(data.iloc[i]).sum()>0: #disregarding rows with null values
            continue
            
        # Long Entry
        if (flag == 0 
            and data['EMA20'].iloc[i]              > data['EMA50'].iloc[i]
            and data['Heiken_Open'].iloc[i]        < data['EMA20'].iloc[i]
            and data['Heiken_Close'].iloc[i]       > data['EMA20'].iloc[i]
            and data['MACD'].iloc[i]               < data['MACD_Signal_Line'].iloc[i]
            and data['RSI_smoothed'].iloc[i]       > 50
            and data['Hawkes_close'].iloc[i]       > data['close'].iloc[i]
            and data['Chaikin_volatility'].iloc[i] < 25
            and data['KST'].iloc[i]                < data['KST_signal'].iloc[i]
            and data['Bull_Power'].iloc[i]         > 0  # Bull Power positive
            and data['Bear_Power'].iloc[i]         < 0  # Bear Power negative
            and data['Plus_DI'].iloc[i]            > data['Minus_DI'].iloc[i]  # +DI higher than -DI for upward trend
            and data['Plus_DI'].iloc[i]            > 10):  # Significant positive DI
            
            data["signals"].iloc[i] = 1  # Signal to go long
            flag = 1  # Long trade active
            data.loc[i, 'trade_type']="long"
            
    
        # Long Exit
        elif (flag == 1 
              and data['EMA20'].iloc[i]        < data['EMA50'].iloc[i]
              and data['filtered_price'][i]    < data['close'].iloc[i]
              and data['Heiken_Open'].iloc[i]  > data['EMA20'].iloc[i]
              and data['Heiken_Close'].iloc[i] < data['EMA20'].iloc[i]):
            
            data["signals"].iloc[i] = -1  # Signal to exit long trade
            flag = 0  # No trade active
            data.loc[i, 'trade_type']="close"
              
    
        # Short Entry
        elif (flag == 0 
              and data['EMA6'].iloc[i]         < data['EMA10'].iloc[i]
              and data['Heiken_Open'].iloc[i]  > data['EMA6'].iloc[i]
              and data['Heiken_Close'].iloc[i] < data['EMA6'].iloc[i]
              and data['MACD'].iloc[i]         > data['MACD_Signal_Line'].iloc[i]
              and data['Minus_DI_s'].iloc[i]   > data['Plus_DI_s'].iloc[i]
              and data['Chebyshev_Filtered'][i]   < data['close'][i]
              and data['RSI_smoothed'].iloc[i] < 40):
            data["signals"].iloc[i] = -1  # Signal to go short
            flag = -1  # Short trade active
            data.loc[i, 'trade_type']="short"
              
    
        # Short Exit
        elif (flag == -1 
              and data['EMA6'].iloc[i]         > data['EMA10'].iloc[i] 
              and data['Heiken_Open'].iloc[i]  < data['EMA6'].iloc[i]
              and data['Heiken_Close'].iloc[i] > data['EMA6'].iloc[i]
              and data['Aroon_Oscillator'][i] > -10
              and data['pct_change_5'][i] <= data['rolling_threshold'][i]):
                
            data.loc[i, "signals"] = 1  # Signal to exit short trade
            flag = 0  # No trade active
            data.loc[i, 'trade_type']="close"

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
    data=pd.read_csv('technial_indicators_approach/btc_data/BTC_2019_2023_2h.csv',index_col=0) 
    data=process_data(data)                                       
    data=strat(data)                                              
    data.to_csv('technial_indicators_approach/final_logs_btc.csv',index=False)                  
    perform_backtest('technial_indicators_approach/final_logs_btc.csv')                        
    
if __name__=='__main__':
    main()