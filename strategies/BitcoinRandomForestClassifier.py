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

import talib
import numpy as np
import pandas as pd
from datetime import *
from sklearn.ensemble import RandomForestClassifier

class BitcoinRandomForestClassifierAlgorithm(QCAlgorithm):

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
        
        self.lookback = 4000 # number of previous min for training
        
        self.SetCash(10000)  # Set Strategy Cash
        self.btcusd = self.AddCrypto("BTCUSD", Resolution.Minute)
        self.SetBenchmark("BTCUSD")

        self.__sto = Stochastic(14, 3, 3)
        self.__rsi7 = RelativeStrengthIndex(7, MovingAverageType.Simple)
        self.__rsi14 = RelativeStrengthIndex(14, MovingAverageType.Simple)
        self.__cci_14 = CommodityChannelIndex(14)
        self.__ema_5 = ExponentialMovingAverage(5)
        self.__ema_12 = ExponentialMovingAverage(12)

        # Register minute data of 'BTCUSD' to automatically update indicators
        self.RegisterIndicator(self.btcusd.Symbol, self.__sto, Resolution.Minute)
        self.RegisterIndicator(self.btcusd.Symbol, self.__rsi7, Resolution.Minute)
        self.RegisterIndicator(self.btcusd.Symbol, self.__rsi14, Resolution.Minute)
        self.RegisterIndicator(self.btcusd.Symbol, self.__cci_14, Resolution.Minute)
        self.RegisterIndicator(self.btcusd.Symbol, self.__ema_5, Resolution.Minute)
        self.RegisterIndicator(self.btcusd.Symbol, self.__ema_12, Resolution.Minute)

        # Warm up indicators
#        self.SetWarmUp(timedelta(20))
        self.first = True
        # Train Immediately
        self.Train(self.TrainModel)

        # Set training every 3000 min
        self.Train(self.DateRules.Every(DayOfWeek.Tuesday, DayOfWeek.Thursday, DayOfWeek.Saturday), self.TimeRules.At(12,0), self.TrainModel)


    def TrainModel(self):
        self.Log(f"TrainModel: Fired at : {self.Time}")
        self.modelIsTraining = True

        # Initialize indicators
        sto = Stochastic(14, 3, 3)
        rsi7 = RelativeStrengthIndex(7, MovingAverageType.Simple)
        rsi14 = RelativeStrengthIndex(14, MovingAverageType.Simple)
        cci_14 = CommodityChannelIndex(14)
        ema_5 = ExponentialMovingAverage(5) 
        ema_12 = ExponentialMovingAverage(12)

        sto_df = pd.DataFrame(columns=['faststoch', 'stochastic', 'stochd', 'stochk']) 
        rsi7_df = pd.DataFrame(columns=['averagegain_rsi_7', 'averageloss_rsi_7', 'rsi_7'])
        rsi14_df = pd.DataFrame(columns=['averagegain_rsi_14', 'averageloss_rsi_14', 'rsi_14'])
        cci14_df = pd.DataFrame(columns=['cci', 'typicalpriceavg', 'typicalpricemeandev'])
        ema_5_df = pd.DataFrame(columns=['ema_5'])
        ema_12_df = pd.DataFrame(columns=['ema_12']) 

        history = self.History(self.btcusd.Symbol, 4000, Resolution.Minute)

        self.Log(f"TrainModel: Before FOR LOOP : {self.Time}")
        for index, row in history.loc["BTCUSD"].iterrows():
            tradeBar = TradeBar(index, "BTCUSD", row.open, row.high, row.low, row.close, row.volume, timedelta(1))
            sto.Update(tradeBar)
            sto_df.loc[index] = [sto.FastStoch.Current.Value, sto.Current.Value, sto.StochD.Current.Value, sto.StochK.Current.Value]

            rsi7.Update(index, row['close'])
            rsi7_df.loc[index] = [rsi7.AverageGain.Current.Value, rsi7.AverageLoss.Current.Value, rsi7.Current.Value]

            rsi14.Update(index, row['close'])
            rsi14_df.loc[index] = [rsi14.AverageGain.Current.Value, rsi14.AverageLoss.Current.Value, rsi14.Current.Value]

            cci_14.Update(tradeBar)
            cci14_df.loc[index] = [cci_14.Current.Value, cci_14.TypicalPriceAverage.Current.Value, cci_14.TypicalPriceMeanDeviation.Current.Value]

            ema_5.Update(index, row['close'])
            ema_5_df.loc[index] = [ema_5.Current.Value]

            ema_12.Update(index, row['close'])
            ema_12_df.loc[index] = [ema_12.Current.Value]
        
        self.Log(f"TrainModel: After FOR LOOP : {self.Time}")
        # Join to one dataframe
        df2 = sto_df.reindex(index=history.index, level=1)
        history = history.join(df2, how='inner')

        df2 = rsi7_df.reindex(index=history.index, level=1)
        history = history.join(df2, how='inner')

        df2 = rsi14_df.reindex(index=history.index, level=1)
        history = history.join(df2, how='inner')

        df2 = cci14_df.reindex(index=history.index, level=1)
        history = history.join(df2, how='inner')

        df2 = ema_5_df.reindex(index=history.index, level=1)
        history = history.join(df2, how='inner')

        df2 = ema_12_df.reindex(index=history.index, level=1)
        history = history.join(df2, how='inner')

        # Feeding 1 input past prices to predict the next price
        lookback = 1

        # Offsetting close data by length of lookback
        history['Future Price'] = history['close'].shift(-lookback)
        history['Future Returns'] = ((history['Future Price'] - history['close'])/ history['close'])
        history['Prediction'] = history["Future Returns"].apply(lambda x: 1 if x > 0 else 0 )

        # prints out the tail of the dataframe
        print(history.loc[self.btcusd.Symbol].tail())

        # Any row that has a 'NaN' value will be dropped
        history = history.dropna()

        X = history.drop(['Future Returns', 'Future Price', 'Prediction'], axis=1)
        y = history['Prediction']

        # Train a model
        self.model = RandomForestClassifier(n_estimators=2587, max_depth=100, max_features=1, class_weight='balanced_subsample',
                                            min_samples_split=5, min_samples_leaf=1, bootstrap=False)
        self.model.fit(X, y)

        self.modelIsTraining = False

       
    def OnData(self, data):
        '''OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.
            Arguments:
                data: Slice object keyed by symbol containing the stock data
        '''

        # Do not use model while its being trained.
        if self.modelIsTraining:
            self.Log(f"OnData: Model is Training : {self.Time}")
            return

        if self.__sto.IsReady and self.__rsi7.IsReady and self.__rsi14.IsReady and self.__cci_14.IsReady and self.__ema_5.IsReady and self.__ema_12.IsReady:
            idx = pd.MultiIndex.from_product([[self.btcusd.Symbol], [data['BTCUSD'].Time]], 
                                             names=self.DataIndex)

            # Get current bitcoin price and technical indicators and store in dataframe
            btc_data = pd.DataFrame([[data['BTCUSD'].Close, data['BTCUSD'].High, data['BTCUSD'].Low, data['BTCUSD'].Open, data['BTCUSD'].Volume, 
                                     self.__sto.FastStoch.Current.Value, self.__sto.Current.Value, self.__sto.StochD.Current.Value, self.__sto.StochK.Current.Value,
                                     self.__rsi7.AverageGain.Current.Value, self.__rsi7.AverageLoss.Current.Value, self.__rsi7.Current.Value,
                                     self.__rsi14.AverageGain.Current.Value, self.__rsi14.AverageLoss.Current.Value, self.__rsi14.Current.Value,
                                     self.__cci_14.Current.Value, self.__cci_14.TypicalPriceAverage.Current.Value, self.__cci_14.TypicalPriceMeanDeviation.Current.Value,
                                     self.__ema_5.Current.Value, self.__ema_12.Current.Value]], idx, columns=self.DataColumns)
            # Calculate predictions
            y_hat = self.model.predict(btc_data)

            if not self.Portfolio.Invested and y_hat[-1]:
                # Get current USD available, subtracting amount reserved for buy open orders
                usdTotal = self.Portfolio.CashBook["USD"].Amount
                self.Log(f"OnData: {self.Time} usdTotal: {usdTotal}")
                # Allocate 100% of portfolio to BTC
                self.SetHoldings("BTCUSD", 1)

            elif self.Portfolio.Invested and y_hat[-1] == 0:
                # Liquidate our BTC holdings (including the initial holding)
                self.SetHoldings("BTCUSD", 0)
    
    def OnOrderEvent(self, orderEvent):
        print("{} {}".format(self.Time, orderEvent.ToString()))            
  
    def Regression(self):
        # Daily historical data is used to train the machine learning model
        history = self.History(self.symbols, self.lookback, Resolution.Daily)

        # price dictionary:    key: symbol; value: historical price
        self.prices = {}
        # slope dictionary:    key: symbol; value: slope
        self.slopes = {}
        
        for symbol in self.symbols:
            if not history.empty:
                # get historical open price
                self.prices[symbol] = list(history.loc[symbol.Value]['open'])

        # A is the design matrix
        A = range(self.lookback + 1)
        
        for symbol in self.symbols:
            if symbol in self.prices:
                # response
                Y = self.prices[symbol]
                # features
                X = np.column_stack([np.ones(len(A)), A])
                
                # data preparation
                length = min(len(X), len(Y))
                X = X[-length:]
                Y = Y[-length:]
                A = A[-length:]
                
                # fit the linear regression
                reg = LinearRegression().fit(X, Y)
                
                # run linear regression y = ax + b
                b = reg.intercept_
                a = reg.coef_[1]
                
                # store slopes for symbols
                self.slopes[symbol] = a/b
                
    
    def Trade(self):
        # if there is no open price
        if not self.prices:
            return 
        
        thod_buy = 0.001 # threshold of slope to buy
        thod_liquidate = -0.001 # threshold of slope to liquidate
        
        for holding in self.Portfolio.Values:
            slope = self.slopes[holding.Symbol] 
            # liquidate when slope smaller than thod_liquidate
            if holding.Invested and slope < thod_liquidate:
                self.Liquidate(holding.Symbol)
        
        for symbol in self.symbols:
            # buy when slope larger than thod_buy
            if self.slopes[symbol] > thod_buy:
                self.SetHoldings(symbol, 1 / len(self.symbols))