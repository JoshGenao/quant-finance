# Josh Genao
# 
# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from clr import AddReference
AddReference("System")
AddReference("QuantConnect.Algorithm")
AddReference("QuantConnect.Common")

from System import *
from QuantConnect import *
from QuantConnect.Algorithm import *
from QuantConnect.Data.Market import *
from QuantConnect.Indicators import *
from QuantConnect.Data.Consolidators import *

import talib
import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.ensemble import RandomForestClassifier

class BitcoinRandomForestClassifierAlgorithm(QCAlgorithm):

    HOLD_MAX = 15
    # Model Object
    model = None
    # Model State Flag
    modelIsTraining = False
    DataIndex = [ 'Symbol', 'Time']
    DataColumns = [ 'Close', 'High', 'Low', 'Open', 'Volume', 'faststoch', 'stochastic', 'stochd', 'stochk', 'averagegain_rsi_7', 'averageloss_rsi_7', 'rsi_7', 
                    'averagegain_rsi_14', 'averageloss_rsi_14', 'rsi_14', 'cci', 'typicalpriceavg', 'typicalpricemeandev', 'ema_5', 'ema_12' ]

    def Initialize(self):
        self.SetStartDate(2020, 2, 1)  # Set Start Date
        self.SetEndDate(2020, 10, 1) # Set End Date
        self.hold = -1 # Variable to hold the amount of minutes holding bitcoin
        
        self.lookback = 4000 # number of previous min for training
        
        self.SetCash(10000)  # Set Strategy Cash
        self.btcusd = self.AddCrypto("BTCUSD", Resolution.Minute)
        self.SetBenchmark("BTCUSD")
        
        # 1 Hr consolidator
        hr_consolidator = TradeBarConsolidator(timedelta(minutes=60))
        hr_consolidator.DataConsolidated += self.HrBarHandler
        self.SubscriptionManager.AddConsolidator("BTCUSD", hr_consolidator)

        # 15min consolidator
        consolidator = TradeBarConsolidator(timedelta(minutes=15))
        consolidator.DataConsolidated += self.DataHandler
        self.SubscriptionManager.AddConsolidator("BTCUSD", consolidator)

        self.rolling_window = pd.DataFrame()
        self.rolling_window_hr = pd.DataFrame()

        self.window_size = 34

        # Train Immediately
        self.Train(self.TrainModel)

        self.Train(self.DateRules.MonthStart(0), self.TimeRules.At(12,0), self.TrainModel)
        # Warm up indicators
        self.SetWarmUp(self.window_size*15)

    def TrainModel(self):
        self.Log(f"TrainModel: Fired at : {self.Time}")
        self.modelIsTraining = True

        history = self.History(self.btcusd.Symbol, 60000, Resolution.Minute)
        history.reset_index(level=0, drop=True, inplace=True)
        history = pd.concat([history['close'].resample('15T').ohlc(), history['volume'].resample('15T').sum()], axis=1)

        history['slowk'], history['slowd'] = talib.STOCH(history['high'], history['low'], history['close'], fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
        history['fastk'], history['fastd'] = talib.STOCHF(history['high'], history['low'], history['close'], fastk_period=14, fastd_period=3, fastd_matype=0)
        history['rsi_7'] = talib.RSI(history['close'], timeperiod=7)
        history['rsi_14'] = talib.RSI(history['close'], timeperiod=14)
        history['wma_5'] = talib.WMA(history['close'], timeperiod=5)
        history['vol_sma30'] = talib.SMA(history['volume'], timeperiod=30)
        history['atr_14'] = talib.ATR(history['high'], history['low'], history['close'], timeperiod=14)
        history['macd'], history['macdsignal'], history['macdhist'] = talib.MACD(history['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        history['cci_14'] = talib.CCI(history['high'], history['low'], history['close'], timeperiod=14)
        history['ema_5'] = talib.EMA(history['close'], timeperiod=5)
        history['ema_12'] = talib.EMA(history['close'], timeperiod=12)

        hr_history = self.History(self.btcusd.Symbol, history.index[0], history.index[-1], Resolution.Hour)

        hr_history.reset_index(level=0, drop=True, inplace=True)
        df_hr_indicator = pd.DataFrame()
        df_hr_indicator['slowk_hr'], df_hr_indicator['slowd_hr'] = talib.STOCH(hr_history['high'], hr_history['low'], hr_history['close'], fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
        df_hr_indicator['fastk_hr'], df_hr_indicator['fastd_hr'] = talib.STOCHF(hr_history['high'], hr_history['low'], hr_history['close'], fastk_period=14, fastd_period=3, fastd_matype=0)
        df_hr_indicator['rsi_7_hr'] = talib.RSI(hr_history['close'], timeperiod=7)
        df_hr_indicator['rsi_14_hr'] = talib.RSI(hr_history['close'], timeperiod=14)
        df_hr_indicator['wma_5_hr'] = talib.WMA(hr_history['close'], timeperiod=5)
        df_hr_indicator['vol_sma30_hr'] = talib.SMA(hr_history['volume'], timeperiod=30)
        df_hr_indicator['atr_14_hr'] = talib.ATR(hr_history['high'], hr_history['low'], hr_history['close'], timeperiod=14)
        df_hr_indicator['macd_hr'], df_hr_indicator['macdsignal_hr'], df_hr_indicator['macdhist_hr'] = talib.MACD(hr_history['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        df_hr_indicator['cci_14_hr'] = talib.CCI(hr_history['high'], hr_history['low'], hr_history['close'], timeperiod=14)
        df_hr_indicator['ema_5_hr'] = talib.EMA(hr_history['close'], timeperiod=5)
        df_hr_indicator['ema_12_hr'] = talib.EMA(hr_history['close'], timeperiod=12)

        history = pd.merge(history, df_hr_indicator, left_index=True, right_index=True,how='outer')
        history.fillna(method='ffill', inplace=True)

        # Feeding 5 input past prices to predict the returns 5 min out
        lookback = 15

        # Offsetting close data by length of lookback
        history['Future Price'] = history['close'].shift(-lookback)
        history['Future Returns'] = ((history['Future Price'] - history['close'])/ history['close'])
        history['Prediction'] = history["Future Returns"].apply(lambda x: 1 if x > 0 else 0 )

        # Any row that has a 'NaN' value will be dropped
        history = history.dropna()

        X = history.drop(['Future Returns', 'Future Price', 'Prediction'], axis=1)
        y = history['Prediction']

        # Train a model
        self.model = RandomForestClassifier(n_estimators=2587, max_depth=100, max_features=1,
                                            min_samples_split=5, min_samples_leaf=1, bootstrap=False, n_jobs = -1)
        self.model.fit(X, y)

        self.modelIsTraining = False

    def HrBarHandler(self, sender, bar):
        if self.rolling_window_hr.shape[0] < self.window_size:
            # Add latest close to rolling window
            row = pd.DataFrame({"close": [bar.Close], "high":[bar.High], "low": [bar.Low], "open": [bar.Open], "volume":[bar.Volume]}, index=[bar.Time])
            self.rolling_window_hr = self.rolling_window_hr.append(row).iloc[-self.window_size:]
            # If we have enough closing data to start calculating indicators...
            if self.rolling_window_hr.shape[0] == self.window_size:
                opens = self.rolling_window_hr['open'].values
                highs = self.rolling_window_hr['high'].values
                lows = self.rolling_window_hr['low'].values
                closes = self.rolling_window_hr['close'].values
                volumes = self.rolling_window_hr['volume'].values
                
                # Add indicator columns to DataFrame
                self.rolling_window_hr['slowk_hr'], self.rolling_window_hr['slowd_hr'] = talib.STOCH(highs, lows, closes, fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
                self.rolling_window_hr['fastk_hr'], self.rolling_window_hr['fastd_hr'] = talib.STOCHF(highs, lows, closes, fastk_period=14, fastd_period=3, fastd_matype=0)
                self.rolling_window_hr['rsi_7_hr'] = talib.RSI(closes, timeperiod=7)
                self.rolling_window_hr['rsi_14_hr'] = talib.RSI(closes, timeperiod=14)
                self.rolling_window_hr['wma_5_hr'] = talib.WMA(closes, timeperiod=5)
                self.rolling_window_hr['vol_sma30_hr'] = talib.SMA(volumes, timeperiod=30)
                self.rolling_window_hr['atr_14_hr'] = talib.ATR(highs, lows, closes, timeperiod=14)
                self.rolling_window_hr['macd_hr'], self.rolling_window_hr['macdsignal_hr'], self.rolling_window_hr['macdhist_hr'] = talib.MACD(closes, fastperiod=12, slowperiod=26, signalperiod=9)
                self.rolling_window_hr['cci_14_hr'] = talib.CCI(highs, lows, closes, timeperiod=14)
                self.rolling_window_hr['ema_5_hr'] = talib.EMA(closes, timeperiod=5)
                self.rolling_window_hr['ema_12_hr'] = talib.EMA(closes, timeperiod=12)
            return

        opens = np.append(self.rolling_window_hr['open'].values, bar.Open)[-self.window_size:]
        highs = np.append(self.rolling_window_hr['high'].values, bar.High)[-self.window_size:]
        lows = np.append(self.rolling_window_hr['low'].values, bar.Low)[-self.window_size:]
        closes = np.append(self.rolling_window_hr['close'].values, bar.Close)[-self.window_size:]
        volumes = np.append(self.rolling_window_hr['volume'].values, bar.Volume)[-self.window_size:]

        slowk, slowd = talib.STOCH(highs, lows, closes, fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
        fastk, fastd = talib.STOCHF(highs, lows, closes, fastk_period=14, fastd_period=3, fastd_matype=0)
        macd, macdsignal, macdhist = talib.MACD(closes, fastperiod=12, slowperiod=26, signalperiod=9)

        # Update talib indicators time series with the latest close
        row = pd.DataFrame({"open"          : bar.Open,
                            "high"          : bar.High,
                            "low"           : bar.Low,
                            "close"         : bar.Close,
                            "volume"        : bar.Volume,
                            "slowk_hr"      : slowk[-1],
                            "slowd_hr"      : slowd[-1],
                            "fastk_hr"      : fastk[-1], 
                            "fastd_hr"      : fastd[-1],
                            "rsi_7_hr"      : talib.RSI(closes, timeperiod=7)[-1],
                            "rsi_14_hr"     : talib.RSI(closes, timeperiod=14)[-1],
                            "wma_5_hr"      : talib.WMA(closes, timeperiod=5)[-1],
                            "vol_sma30_hr"  : talib.SMA(volumes, timeperiod=30)[-1],
                            "atr_14_hr"     : talib.ATR(highs, lows, closes, timeperiod=14)[-1],
                            "macd_hr"       : macd[-1],
                            "macdsignal_hr" : macdsignal[-1], 
                            "macdhist_hr"   : macdhist[-1],
                            "cci_14_hr"     : talib.CCI(highs, lows, closes, timeperiod=14)[-1],
                            "ema_5_hr"      : talib.EMA(closes, timeperiod=5)[-1],
                            "ema_12_hr"     : talib.EMA(closes, timeperiod=12)[-1]},
                            index=[bar.Time])

        self.rolling_window_hr = self.rolling_window_hr.append(row).iloc[-self.window_size:]
          
    def OnData(self, data):
        pass  

    def DataHandler(self, sender, data):
        '''OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.
            Arguments:
                data: Slice object keyed by symbol containing the stock data
        '''
        # Do not use model while its being trained.

        if self.hold == self.HOLD_MAX and self.Portfolio.Invested:
            self.Log(f"OnData: SELL >> : {self.Time}")
            self.SetHoldings("BTCUSD", 0)
            self.hold = -1
       
        if self.modelIsTraining:
            self.Log(f"OnData: Model is Training : {self.Time}")
            return

        if self.rolling_window.shape[0] < self.window_size:
            # Add latest close to rolling window
            row = pd.DataFrame({"close": [data.Close], "high":[data.High], "low": [data.Low], "open": [data.Open], "volume":[data.Volume]}, index=[data.Time])
            self.rolling_window = self.rolling_window.append(row).iloc[-self.window_size:]
            return

        opens = np.append(self.rolling_window['open'].values, data.Open)[-self.window_size:]
        highs = np.append(self.rolling_window['high'].values, data.High)[-self.window_size:]
        lows = np.append(self.rolling_window['low'].values, data.Low)[-self.window_size:]
        closes = np.append(self.rolling_window['close'].values, data.Close)[-self.window_size:]
        volumes = np.append(self.rolling_window['volume'].values, data.Volume)[-self.window_size:]

        slowk, slowd = talib.STOCH(highs, lows, closes, fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
        fastk, fastd = talib.STOCHF(highs, lows, closes, fastk_period=14, fastd_period=3, fastd_matype=0)
        macd, macdsignal, macdhist = talib.MACD(closes, fastperiod=12, slowperiod=26, signalperiod=9)

        # Update talib indicators time series with the latest close
        row = pd.DataFrame({"open"       : data.Open,
                            "high"       : data.High,
                            "low"        : data.Low,
                            "close"      : data.Close,
                            "volume"     : data.Volume,
                            "slowk"      : slowk[-1],
                            "slowd"      : slowd[-1],
                            "fastk"      : fastk[-1], 
                            "fastd"      : fastd[-1],
                            "rsi_7"      : talib.RSI(closes, timeperiod=7)[-1],
                            "rsi_14"     : talib.RSI(closes, timeperiod=14)[-1],
                            "wma_5"      : talib.WMA(closes, timeperiod=5)[-1],
                            "vol_sma30"  : talib.SMA(volumes, timeperiod=30)[-1],
                            "atr_14"     : talib.ATR(highs, lows, closes, timeperiod=14)[-1],
                            "macd"       : macd[-1],
                            "macdsignal" : macdsignal[-1], 
                            "macdhist"   : macdhist[-1],
                            "cci_14"     : talib.CCI(highs, lows, closes, timeperiod=14)[-1],
                            "ema_5"      : talib.EMA(closes, timeperiod=5)[-1],
                            "ema_12"     : talib.EMA(closes, timeperiod=12)[-1]},
                            index=[data.Time])

        self.rolling_window = self.rolling_window.append(row).iloc[-self.window_size:]

        if self.rolling_window_hr.shape[0] == self.window_size:
            if not self.Portfolio.Invested:
                rolling_window_hr = self.rolling_window_hr.drop(['open', 'high', 'low', 'close', 'volume'], axis=1)
                row = pd.merge(self.rolling_window, rolling_window_hr, left_index=True, right_index=True,how='outer')
                row.fillna(method='ffill', inplace=True)

                # Calculate predictions
                y_hat = self.model.predict(row.tail(1))
                y_score = self.model.predict_proba(row.tail(1))[:,1]

                if y_hat[-1] and y_score[-1] > 0.7:
                    # Get current USD available, subtracting amount reserved for buy open orders
                    usdTotal = self.Portfolio.CashBook["USD"].Amount
                    self.Log(f"OnData: {self.Time} usdTotal: {usdTotal}")
                    # Allocate 100% of portfolio to BTC
                    self.SetHoldings("BTCUSD", 1)
                    self.hold = 1
            else:
                self.hold += 1
                self.Log(f"OnData {self.Time} : Hold: {self.hold}")
    
    def OnOrderEvent(self, orderEvent):
        print("{} {}".format(self.Time, orderEvent.ToString()))            