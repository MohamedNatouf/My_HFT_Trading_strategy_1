
import pandas as pd
import numpy as np
import sys
import os
import ast
import re
import time
import math
import matplotlib.pyplot as plt
from dateutil.parser import parse
from datetime import datetime, date
from operator import itemgetter
from pandas.core import series

import yfinance
import yahoo_fin.stock_info as si

import ta.momentum as Technical_Analysis_1_Momentum
import ta.volatility as Technical_Analysis_1_Volatility
import ta.utils as ta_ut
import ta.trend as Technical_Analysis_1_Trend
import ta.volume as Technical_Analysis_1_Volume
import pywt

from tindicators import ti as Technical_Analysis_2 

import pypfopt
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

import btalib as bta_lib

import Functions as Functions
import Model_Input_Handler as Model_Input
import Parameters_Input_Handler as Parameters_Input
import Active_Trades_Handler as Active_Trades_Handler

import blankly

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn import metrics

#from blueshift import blueshift
#from backtesting import backtesting
#from tensorflow.python.keras.models import sequential
#.discrete_allocation import DiscreteAllocation, get_latest_prices
#import scipy

separator = ", "

# Assets Momentum Measuring Unit:
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def Retrieve_assets_Data(Request_Type, Asset_Col, Date_Row, Previous_price_Row, Current_Date, Lookback_Period, First_Date, Open_prices_Array, High_prices_Array, Low_prices_Array, Close_prices_Array, Adj_Close_prices_Array, Volume_Array, Asset_Symbols, Date_Array):
   
    Quant_Method = "Momentum_MultiFactor_Maximizer"
   
    if Request_Type == "Get Metric Data":

        Asset_Inception_Date_Row = 0
        Combined_Metric_Value = 0
        Asset_parameters_Array = []
        Detailed_MetricsData_Array = []

        Combined_Metric_Array_Dim = 0

        # Current Measured Asset's Price :
        Currnet_date_price =  Adj_Close_prices_Array[Date_Row][Asset_Col]

        # Previous lookback Measured Asset's Price :
        LookBack_date_Price = Adj_Close_prices_Array[Date_Row - Lookback_Period][Asset_Col]

        #if  LookBack_date_Price== 0 or np.isnan(LookBack_date_Price) == True:
        #    LookBack_date_Price = Adj_Close_prices_Array[Date_Row - Lookback_Period-1][Asset_Col]

        if np.isnan(LookBack_date_Price) == False and np.isnan(Currnet_date_price) == False:

            Asset_Inception_Date_Row= Adj_Close_prices_Array.index(next(filter(lambda x: not np.isnan(x[:][Asset_Col]), Adj_Close_prices_Array)))
        
            if Date_Row - int(Asset_Inception_Date_Row)>10:
                Assets_Prices_Range=(list(map(itemgetter(Asset_Col), Adj_Close_prices_Array))[Asset_Inception_Date_Row:])
                Assets_Prices_Dataframe = pd.DataFrame(Assets_Prices_Range)
                Asset_Volatility_Array = (Assets_Prices_Dataframe.pct_change())
                Asset_Volatility = (Asset_Volatility_Array.std().values.tolist())[0] * math.sqrt(252)
            else:
                Asset_Volatility = 0
            
            if Quant_Method == "Momentum_MultiFactor_Maximizer":
                
                #High_Lookback_Period_Array =pd.Series(list(map(itemgetter(Asset_Col), High_prices_Array))[:])
                #Low_Lookback_Period_Array = pd.Series(list(map(itemgetter(Asset_Col), Low_prices_Array))[:])
                #Open_Lookback_Period_Array = pd.Series(list(map(itemgetter(Asset_Col), Open_prices_Array))[:]) 
                #Close_Lookback_Period_Array = pd.Series(list(map(itemgetter(Asset_Col), Close_prices_Array))[:])
                Adj_Close_Lookback_Period_Array = pd.Series(list(map(itemgetter(Asset_Col), Adj_Close_prices_Array))[:])
                #Volume_Lookback_Period_Array = pd.Series(list(map(itemgetter(Asset_Col), Volume_Array))[:])
                #Adj_Close_Lookback_Period_Previous_Array = pd.Series(list(map(itemgetter(Asset_Col), Adj_Close_prices_Array))[:-1])
                
                Momentum_MultiFactor_Maximizer, MetricsData_Array = Functions.Momentum_MultiFactor_Maximizer(Lookback_Period, Date_Row, Adj_Close_Lookback_Period_Array)
                
                Checked_Current_Date = datetime.strptime(str(Current_Date),'%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d')  
                Checked_Current_Date = datetime.strptime(Checked_Current_Date, '%Y-%m-%d')

                Detailed_MetricsData_Array = [Checked_Current_Date, Asset_Symbols[Asset_Col], Momentum_MultiFactor_Maximizer]
                Detailed_MetricsData_Array.extend(MetricsData_Array)
                Detailed_MetricsData_Array.extend([Currnet_date_price])
                
                #print("Asset_Processed = " + str(Asset_Symbols[Asset_Col]) + "  --- Momentum_MultiFactor_Maximizer = " + str(Momentum_MultiFactor_Maximizer) + "  --- Current_Date = " + str(Current_Date) + "  --- Current_Price = " + str(Currnet_date_price))
                
                Combined_Metric_Value =  Momentum_MultiFactor_Maximizer                   
                Combined_Metric_Array = [Combined_Metric_Value]           
                Combined_Metric_Array_Dim = len(Combined_Metric_Array)

            elif Quant_Method == "Others":

                High_Lookback_Period_Array =pd.Series(list(map(itemgetter(Asset_Col), High_prices_Array))[Date_Row- Lookback_Period:Date_Row])
                Low_Lookback_Period_Array = pd.Series(list(map(itemgetter(Asset_Col), Low_prices_Array))[Date_Row- Lookback_Period:Date_Row])
                Open_Lookback_Period_Array = pd.Series(list(map(itemgetter(Asset_Col), Open_prices_Array))[Date_Row- Lookback_Period:Date_Row]) 
                Close_Lookback_Period_Array = pd.Series(list(map(itemgetter(Asset_Col), Close_prices_Array))[Date_Row- Lookback_Period:Date_Row])
                Adj_Close_Lookback_Period_Array = pd.Series(list(map(itemgetter(Asset_Col), Adj_Close_prices_Array))[Date_Row- Lookback_Period:Date_Row])
                Volume_Lookback_Period_Array = pd.Series(list(map(itemgetter(Asset_Col), Volume_Array))[Date_Row- Lookback_Period:Date_Row])
                Adj_Close_Lookback_Period_Previous_Array = pd.Series(list(map(itemgetter(Asset_Col), Adj_Close_prices_Array))[Date_Row- Lookback_Period:Date_Row-1])
                
                #High_Lookback_Period_Array = Functions.Data_exponential_smooth(High_Lookback_Period_Array, 0.65)
                #Low_Lookback_Period_Array = Functions.Data_exponential_smooth(Low_Lookback_Period_Array, 0.65)
                #Open_Lookback_Period_Array = Functions.Data_exponential_smooth(Open_Lookback_Period_Array, 0.65)
                #Low_Lookback_Period_Array = Functions.Data_exponential_smooth(Low_Lookback_Period_Array, 0.65)
                #Close_Lookback_Period_Array = Functions.Data_exponential_smooth(Close_Lookback_Period_Array, 0.65)
                #Adj_Close_Lookback_Period_Array = Functions.Data_exponential_smooth(Adj_Close_Lookback_Period_Array, 0.65)
                #Volume_Lookback_Period_Array = Functions.Data_exponential_smooth(Volume_Lookback_Period_Array, 0.65)

                Asset_Total_Return_RS=  ((Currnet_date_price/LookBack_date_Price)-1)*100
                MACD_Metric = (((Technical_Analysis_1_Trend.MACD(Adj_Close_Lookback_Period_Array,26,12,9,False)).macd().tolist())[-1] / ((Technical_Analysis_1_Trend.MACD(Adj_Close_Lookback_Period_Array,26,12,9,False)).macd_signal().tolist())[-1]) if (((Technical_Analysis_1_Trend.MACD(Adj_Close_Lookback_Period_Array,26,12,9,False)).macd_signal().tolist())[-1] != 0) else 0
                Commodity_Channel_Index_CCI_Metric =  ((Technical_Analysis_1_Trend.CCIIndicator(High_Lookback_Period_Array, Low_Lookback_Period_Array, Close_Lookback_Period_Array, 20, 0.015,False)).cci().tolist())[-1]
                RSI_Metric = (((Technical_Analysis_1_Momentum.RSIIndicator(Adj_Close_Lookback_Period_Array,14,False))).rsi().tolist())[-1]
                Stochastic_Oscillator_Metric = ((Technical_Analysis_1_Momentum.StochasticOscillator(High_Lookback_Period_Array, Low_Lookback_Period_Array, Close_Lookback_Period_Array, 14, 3, False)).stoch().tolist())[-1]
                Percentage_Price_Oscillator_Metric = ((Technical_Analysis_1_Momentum.PercentagePriceOscillator(Adj_Close_Lookback_Period_Array, 26, 12, 9, False)).ppo().tolist())[-1]
                ROC_Metric = (((Technical_Analysis_1_Momentum.ROCIndicator(Adj_Close_Lookback_Period_Array,14,False))).roc().tolist())[-1]
                AwesomeOscillator_Metric = (((Technical_Analysis_1_Momentum.AwesomeOscillatorIndicator(High_Lookback_Period_Array,Low_Lookback_Period_Array,5,34,False))).awesome_oscillator().tolist())[-1]
                Percentage_Volume_Oscillator_Metric = (((Technical_Analysis_1_Momentum.PercentageVolumeOscillator(Volume_Lookback_Period_Array,26,12,9,False))).pvo().tolist())[-1] 
                #Trend_ADX_Metric = (((Technical_Analysis_1_Trend.ADXIndicator(High_Lookback_Period_Array,Low_Lookback_Period_Array,Close_Lookback_Period_Array,14,False))).adx().tolist())[-1]
                Aroon_Metric = (((Technical_Analysis_1_Trend.AroonIndicator(Adj_Close_Lookback_Period_Array,25,False))).aroon_indicator().tolist())[-1]
                Stoch_RSI_Metric = (((Technical_Analysis_1_Momentum.StochRSIIndicator(Adj_Close_Lookback_Period_Array,14,1,1,False))).stochrsi_d().tolist())[-1]
                TSI_Metric = (((Technical_Analysis_1_Momentum.TSIIndicator(Adj_Close_Lookback_Period_Array,25,13,False))).tsi().tolist())[-1]
                Kaufman_Adaptive_Moving_Average_Metric = ((Technical_Analysis_1_Momentum.kama(Adj_Close_Lookback_Period_Array, 10, 2, 30, False)).tolist())[-1]
                Vortex_Metric = ((Technical_Analysis_1_Trend.VortexIndicator(High_Lookback_Period_Array, Low_Lookback_Period_Array, Close_Lookback_Period_Array, 14, False)).vortex_indicator_diff().tolist())[-1]
                SMA_Instantaneous_Metric = (((((Technical_Analysis_1_Trend.sma_indicator(Adj_Close_Lookback_Period_Array, 14, False)).tolist())[-1])/(((Technical_Analysis_1_Trend.sma_indicator(Adj_Close_Lookback_Period_Previous_Array, 14, False)).tolist())[-1]))-1)

                AverageTrueRange_Metric = ((Technical_Analysis_1_Volatility.AverageTrueRange(High_Lookback_Period_Array,Low_Lookback_Period_Array,Close_Lookback_Period_Array,14,False)).average_true_range().tolist())[-1]
                ChaikinMoneyFlow_Metric = (((Technical_Analysis_1_Volume.ChaikinMoneyFlowIndicator(High_Lookback_Period_Array,Low_Lookback_Period_Array,Close_Lookback_Period_Array,Volume_Lookback_Period_Array,14,False))).chaikin_money_flow().tolist())[-1]
                KST_Metric = ((Technical_Analysis_1_Trend.KSTIndicator(Adj_Close_Lookback_Period_Array,10, 15, 20, 30, 10, 10, 10, 15, 9, False)).kst().tolist())[-1]
                BollingerBands_Metric = (Adj_Close_Lookback_Period_Array.tolist())[-1]/(((Technical_Analysis_1_Volatility.BollingerBands(Adj_Close_Lookback_Period_Array,20,2,False)).bollinger_hband().tolist())[-1])
                Donchian_channel_Metric = (Adj_Close_Lookback_Period_Array.tolist())[-1]/(((Technical_Analysis_1_Volatility.donchian_channel_hband(High_Lookback_Period_Array, Low_Lookback_Period_Array, Close_Lookback_Period_Array,20,0,False)).tolist())[-1])
                keltner_channel_Metric = (Adj_Close_Lookback_Period_Array.tolist())[-1]/(((Technical_Analysis_1_Volatility.KeltnerChannel(High_Lookback_Period_Array, Low_Lookback_Period_Array, Close_Lookback_Period_Array, 14, 10, False, True, 2)).keltner_channel_hband().tolist())[-1])

                Combined_Metric_Value =  Asset_Total_Return_RS * abs(Percentage_Price_Oscillator_Metric)  * abs(Percentage_Volume_Oscillator_Metric) * abs(Vortex_Metric) * abs(TSI_Metric) * abs(SMA_Instantaneous_Metric) * abs(KST_Metric) * abs(keltner_channel_Metric) * abs(Donchian_channel_Metric) #           * abs(BollingerBands_Metric) 
                Combined_Metric_Array = [Asset_Total_Return_RS, Percentage_Price_Oscillator_Metric , Percentage_Volume_Oscillator_Metric, Vortex_Metric, TSI_Metric , SMA_Instantaneous_Metric, KST_Metric, keltner_channel_Metric , Donchian_channel_Metric] # , ChaikinMoneyFlow_Metric,  BollingerBands_Metric, Donchian_channel_Metric, keltner_channel_Metric, KST_Metric]
                Combined_Metric_Array_Dim = len(Combined_Metric_Array)

            Asset_parameters_Array = [Asset_Symbols[Asset_Col], Combined_Metric_Value, Volume_Array[Date_Row][Asset_Col], Asset_Volatility, Asset_Col, Current_Date]
            Asset_parameters_Array.extend(Combined_Metric_Array) 
           
            
            #print(f'Asset Symbol: {List_of_Ranked_Assets[-1][0]} ---- Metric Value = {List_of_Ranked_Assets[-1][1]}', sep='\n')

            return Asset_parameters_Array, Detailed_MetricsData_Array, Combined_Metric_Array_Dim
        
        return [], [], 0

    elif Request_Type == "Get Raw Data":
        
        Asset_Inception_Date_Row = 0
        Combined_Metric_Value = 0
        Asset_parameters_Array = []
        Detailed_MetricsData_Array = []

        Combined_Metric_Array_Dim = 0
        Asset_Volatility = 0
        
        Multiplied_Lookback_Period =  Lookback_Period  #3*

        # Current Measured Asset's Price :
        Currnet_date_price =  Adj_Close_prices_Array[Date_Row][Asset_Col]

        # Previous lookback Measured Asset's Price :
        LookBack_date_Price = Adj_Close_prices_Array[Date_Row - Multiplied_Lookback_Period][Asset_Col]

        if np.isnan(LookBack_date_Price) == False and np.isnan(Currnet_date_price) == False:

            Asset_Inception_Date_Row= Adj_Close_prices_Array.index(next(filter(lambda x: not np.isnan(x[:][Asset_Col]), Adj_Close_prices_Array)))
        
            if Date_Row - int(Asset_Inception_Date_Row)>10:
                Assets_Prices_Range=(list(map(itemgetter(Asset_Col), Adj_Close_prices_Array))[Asset_Inception_Date_Row:])
                Assets_Prices_Dataframe = pd.DataFrame(Assets_Prices_Range)
                Asset_Volatility_Array = (Assets_Prices_Dataframe.pct_change())
                Asset_Volatility = (Asset_Volatility_Array.std().values.tolist())[0] * math.sqrt(252)
            else:
                Asset_Volatility = 0
                
            High_Lookback_Period_Array =pd.Series(list(map(itemgetter(Asset_Col), High_prices_Array)))
            Low_Lookback_Period_Array = pd.Series(list(map(itemgetter(Asset_Col), Low_prices_Array)))
            Open_Lookback_Period_Array = pd.Series(list(map(itemgetter(Asset_Col), Open_prices_Array))) 
            Close_Lookback_Period_Array = pd.Series(list(map(itemgetter(Asset_Col), Close_prices_Array)))
            Adj_Close_Lookback_Period_Array = pd.Series(list(map(itemgetter(Asset_Col), Adj_Close_prices_Array)))
            Volume_Lookback_Period_Array = pd.Series(list(map(itemgetter(Asset_Col), Volume_Array)))
            Adj_Close_Lookback_Period_Previous_Array = pd.Series(list(map(itemgetter(Asset_Col), Adj_Close_prices_Array))).shift(-(int(round(1,0))))

            # calculate indicators features arrays:
            Total_Returns_Feature =  (((Adj_Close_Lookback_Period_Array/Adj_Close_Lookback_Period_Array.shift(-Multiplied_Lookback_Period))-1).fillna(0).tolist())[Date_Row- Multiplied_Lookback_Period:Date_Row]
            #MACD_Feature = (((Technical_Analysis_1_Trend.MACD(Adj_Close_Lookback_Period_Array,26,12,9,False)).macd().tolist())/ ((Technical_Analysis_1_Trend.MACD(Adj_Close_Lookback_Period_Array,26,12,9,False)).macd_signal().tolist())) if (((Technical_Analysis_1_Trend.MACD(Adj_Close_Lookback_Period_Array,26,12,9,False)).macd_signal().tolist()) != 0) else 0

            MACD_Feature = ((((Technical_Analysis_1_Trend.MACD(Adj_Close_Lookback_Period_Array,26,12,9,True)).macd())/ ((Technical_Analysis_1_Trend.MACD(Adj_Close_Lookback_Period_Array,26,12,9,True)).macd_signal())).fillna(0).tolist())[Date_Row- Multiplied_Lookback_Period:Date_Row] #if (((Technical_Analysis_1_Trend.MACD(Adj_Close_Lookback_Period_Array,26,12,9,False)).macd_signal()) != 0) else 0
            # MACD_Feature = MACD_Feature.tolist()

            Commodity_Channel_Index_CCI_Feature =  ((Technical_Analysis_1_Trend.CCIIndicator(High_Lookback_Period_Array, Low_Lookback_Period_Array, Close_Lookback_Period_Array, 20, 0.015, True)).cci().fillna(0).fillna(0).tolist())[Date_Row- Multiplied_Lookback_Period:Date_Row]
            RSI_Feature = (((Technical_Analysis_1_Momentum.RSIIndicator(Adj_Close_Lookback_Period_Array,14, True))).rsi().fillna(0).tolist())[Date_Row- Multiplied_Lookback_Period:Date_Row]
            Stochastic_Oscillator_Feature = ((Technical_Analysis_1_Momentum.StochasticOscillator(High_Lookback_Period_Array, Low_Lookback_Period_Array, Close_Lookback_Period_Array, 14, 3, True)).stoch().fillna(0).tolist())[Date_Row- Multiplied_Lookback_Period:Date_Row]
            Percentage_Price_Oscillator_Feature = ((Technical_Analysis_1_Momentum.PercentagePriceOscillator(Adj_Close_Lookback_Period_Array, 26, 12, 9, True)).ppo().fillna(0).tolist())[Date_Row- Multiplied_Lookback_Period:Date_Row]
            ROC_Feature = (((Technical_Analysis_1_Momentum.ROCIndicator(Adj_Close_Lookback_Period_Array,14,True))).roc().fillna(0).tolist())[Date_Row- Multiplied_Lookback_Period:Date_Row]
            AwesomeOscillator_Feature = (((Technical_Analysis_1_Momentum.AwesomeOscillatorIndicator(High_Lookback_Period_Array,Low_Lookback_Period_Array,5,34,True))).awesome_oscillator().fillna(0).tolist())[Date_Row- Multiplied_Lookback_Period:Date_Row]
            Percentage_Volume_Oscillator_Feature = (((Technical_Analysis_1_Momentum.PercentageVolumeOscillator(Volume_Lookback_Period_Array,26,12,9,True))).pvo().fillna(0).tolist())[Date_Row- Multiplied_Lookback_Period:Date_Row]

            Aroon_Feature = (((Technical_Analysis_1_Trend.AroonIndicator(Adj_Close_Lookback_Period_Array,25,True))).aroon_indicator().fillna(0).tolist())[Date_Row- Multiplied_Lookback_Period:Date_Row]
            Stoch_RSI_Feature = (((Technical_Analysis_1_Momentum.StochRSIIndicator(Adj_Close_Lookback_Period_Array,14,1,1,True))).stochrsi_d().fillna(0).tolist())[Date_Row- Multiplied_Lookback_Period:Date_Row]
            TSI_Feature = (((Technical_Analysis_1_Momentum.TSIIndicator(Adj_Close_Lookback_Period_Array,25,13,True))).tsi().fillna(0).tolist())[Date_Row- Multiplied_Lookback_Period:Date_Row]
            Kaufman_Adaptive_Moving_Average_Feature = ((Technical_Analysis_1_Momentum.kama(Adj_Close_Lookback_Period_Array, 10, 2, 30, True)).fillna(0).tolist())[Date_Row- Multiplied_Lookback_Period:Date_Row]
            Vortex_Feature = ((Technical_Analysis_1_Trend.VortexIndicator(High_Lookback_Period_Array, Low_Lookback_Period_Array, Close_Lookback_Period_Array, 14, True)).vortex_indicator_diff().fillna(0).tolist())[Date_Row- Multiplied_Lookback_Period:Date_Row]

            #SMA_Instantaneous_Feature = (((((Technical_Analysis_1_Trend.sma_indicator(Adj_Close_Lookback_Period_Array, 14, False)).tolist()))/(((Technical_Analysis_1_Trend.sma_indicator(Adj_Close_Lookback_Period_Previous_Array, 14, False)).tolist())))-1)
            SMA_Instantaneous_Feature = (((((Technical_Analysis_1_Trend.sma_indicator(Adj_Close_Lookback_Period_Array, 14, True))))/(((Technical_Analysis_1_Trend.sma_indicator(Adj_Close_Lookback_Period_Previous_Array, 14, True)))))-1)
            SMA_Instantaneous_Feature = SMA_Instantaneous_Feature.fillna(0).tolist()[Date_Row- Multiplied_Lookback_Period:Date_Row]

            AverageTrueRange_Feature = ((Technical_Analysis_1_Volatility.AverageTrueRange(High_Lookback_Period_Array,Low_Lookback_Period_Array,Close_Lookback_Period_Array,7,True)).average_true_range().fillna(0).tolist())[Date_Row- Multiplied_Lookback_Period:Date_Row]
            ChaikinMoneyFlow_Feature = (((Technical_Analysis_1_Volume.ChaikinMoneyFlowIndicator(High_Lookback_Period_Array,Low_Lookback_Period_Array,Close_Lookback_Period_Array,Volume_Lookback_Period_Array,14,True))).chaikin_money_flow().fillna(0).tolist())[Date_Row- Multiplied_Lookback_Period:Date_Row]
            #KST_Feature = (((Technical_Analysis_1_Trend.KSTIndicator(Adj_Close_Lookback_Period_Array,10, 15, 20, 30, 10, 10, 10, 15, 9, False)).kst().tolist()) / ((Technical_Analysis_1_Trend.KSTIndicator(Adj_Close_Lookback_Period_Array,10, 15, 20, 30, 10, 10, 10, 15, 9, False)).kst_sig().tolist())) if (((Technical_Analysis_1_Trend.KSTIndicator(Adj_Close_Lookback_Period_Array,10, 15, 20, 30, 10, 10, 10, 15, 9, False)).kst_sig().tolist()) != 0) else 0
            
            KST_Feature = (((Technical_Analysis_1_Trend.KSTIndicator(Adj_Close_Lookback_Period_Array,10, 15, 20, 30, 10, 10, 10, 15, 9, True)).kst()) / ((Technical_Analysis_1_Trend.KSTIndicator(Adj_Close_Lookback_Period_Array,10, 15, 20, 30, 10, 10, 10, 15, 9, True)).kst_sig())) 
            KST_Feature = KST_Feature.fillna(0).tolist()[Date_Row- Multiplied_Lookback_Period:Date_Row]

            BollingerBands_Feature = (Adj_Close_Lookback_Period_Array)/(((Technical_Analysis_1_Volatility.BollingerBands(Adj_Close_Lookback_Period_Array,20,2,True)).bollinger_hband()))
            BollingerBands_Feature = BollingerBands_Feature.fillna(0).tolist()[Date_Row- Multiplied_Lookback_Period:Date_Row]

            Donchian_channel_Feature = (Adj_Close_Lookback_Period_Array)/(((Technical_Analysis_1_Volatility.donchian_channel_hband(High_Lookback_Period_Array, Low_Lookback_Period_Array, Close_Lookback_Period_Array,20,0,True))))
            Donchian_channel_Feature = Donchian_channel_Feature.fillna(0).tolist()[Date_Row- Multiplied_Lookback_Period:Date_Row]

            keltner_channel_Feature = (Adj_Close_Lookback_Period_Array.tolist())/(((Technical_Analysis_1_Volatility.KeltnerChannel(High_Lookback_Period_Array, Low_Lookback_Period_Array, Close_Lookback_Period_Array, 14, 10, True, True, 2)).keltner_channel_hband()))
            keltner_channel_Feature = keltner_channel_Feature.fillna(0).tolist()[Date_Row- Multiplied_Lookback_Period:Date_Row]

            AverageTrueRange_Feature = ((Technical_Analysis_1_Volatility.AverageTrueRange(High_Lookback_Period_Array,Low_Lookback_Period_Array,Close_Lookback_Period_Array,7,True)).average_true_range().fillna(0).tolist())[Date_Row- Multiplied_Lookback_Period:Date_Row]
            Trend_ADX_Feature = (((Technical_Analysis_1_Trend.ADXIndicator(High_Lookback_Period_Array,Low_Lookback_Period_Array,Close_Lookback_Period_Array,7,True))).adx().fillna(0).tolist())[Date_Row- Multiplied_Lookback_Period:Date_Row]
            
            #Technical_Analysis_2.pfe()
#           ti.pfe(series: np.ndarray,period, ema_period) ---> NamedTuple(..., pfe = np.ndarray)

            # Also calculate EWMA moving averages for features arrays:
            Adj_Close_Lookback_Period_50D_EWMA_Feature = ((Adj_Close_Lookback_Period_Array / Adj_Close_Lookback_Period_Array.ewm(50).mean()).fillna(0).tolist())[Date_Row- Multiplied_Lookback_Period:Date_Row]
            Adj_Close_Lookback_Period_21D_EWMA_Feature = ((Adj_Close_Lookback_Period_Array / Adj_Close_Lookback_Period_Array.ewm(21).mean()).fillna(0).tolist())[Date_Row- Multiplied_Lookback_Period:Date_Row]
            Adj_Close_Lookback_Period_15D_EWMA_Feature = ((Adj_Close_Lookback_Period_Array / Adj_Close_Lookback_Period_Array.ewm(14).mean()).fillna(0).tolist())[Date_Row- Multiplied_Lookback_Period:Date_Row]
            Adj_Close_Lookback_Period_5D_EWMA_Feature = ((Adj_Close_Lookback_Period_Array / Adj_Close_Lookback_Period_Array.ewm(5).mean()).fillna(0).tolist())[Date_Row- Multiplied_Lookback_Period:Date_Row]
            Volume_Lookback_Period_5D_EWMA_Feature = ((Volume_Lookback_Period_Array / Volume_Lookback_Period_Array.ewm(5).mean()).fillna(0).tolist())[Date_Row- Multiplied_Lookback_Period:Date_Row]

            prediction_shift = Multiplied_Lookback_Period/14 #Multiplied_Lookback_Period

            prediction_output_1 = pd.Series(Adj_Close_Lookback_Period_Array).shift(-(int(round(prediction_shift,0)))) <= pd.Series(Adj_Close_Lookback_Period_Array)
            prediction_output_2 = prediction_output_1.iloc[Date_Row - Multiplied_Lookback_Period:Date_Row]
            prediction_output = prediction_output_2.astype(int)

            #prediction_output = prediction_output.dropna() # Some indicators produce NaN values for the first few rows, we just remove them here
            #prediction_output.tail()

            features_tuple = list(zip(Adj_Close_Lookback_Period_50D_EWMA_Feature, Adj_Close_Lookback_Period_21D_EWMA_Feature, Adj_Close_Lookback_Period_15D_EWMA_Feature, Adj_Close_Lookback_Period_5D_EWMA_Feature, Volume_Lookback_Period_5D_EWMA_Feature, Total_Returns_Feature, MACD_Feature, Commodity_Channel_Index_CCI_Feature, RSI_Feature, Stochastic_Oscillator_Feature, Percentage_Price_Oscillator_Feature, ROC_Feature, AwesomeOscillator_Feature, Percentage_Volume_Oscillator_Feature, Trend_ADX_Feature, Aroon_Feature, Stoch_RSI_Feature, TSI_Feature, Kaufman_Adaptive_Moving_Average_Feature, Vortex_Feature, SMA_Instantaneous_Feature, AverageTrueRange_Feature, ChaikinMoneyFlow_Feature, KST_Feature, BollingerBands_Feature, Donchian_channel_Feature, keltner_channel_Feature, prediction_output))
            Asset_parameters_Array = pd.DataFrame(features_tuple, columns=['adj close 50D ewma', 'adj close 21D ewma', 'adj close 15D ewma', 'adj close 5D ewma', 'volume 5D ewma', 'total_returns', 'macd', 'commodity_channel_index_cci', 'rsi', 'stochastic_oscillator', 'percentage_price_oscillator' , 'roc',  'awesome oscillator' , 'percentage_volume_oscillator', 'trend_adx', 'aroon', 'stoch_rsi', 'tsi', ' kaufman_adaptive_moving_average', 'vortex', 'sma_instantaneous', 'averagetruerange', 'chaikinmoneyflow', 'kst', 'bollingerbands', 'donchian_channel', 'keltner_channel', 'prediction output'])
            Asset_parameters_Array.fillna(0)
            # print(Asset_parameters_Array.to_string())
            
        return Asset_parameters_Array, Detailed_MetricsData_Array, Asset_Volatility


def Assets_Selection_Handler(Date_Row, Previous_price_Row, Current_Date, First_Date, Type_of_Momentum_Method, Selected_method_of_Momuntum_Metric, Method_of_Assets_Selection, Lookback_Period, Number_of_Portfolio_Allocations, Asset_Prices_Array, Open_prices_Array, High_prices_Array, Low_prices_Array, Close_prices_Array, Adj_Close_prices_Array, Volume_Array, Asset_Symbols, Date_Array, Special_Filter_Status, Special_Filter_Type):
   
    if Method_of_Assets_Selection == "Price Momuntum Strategy":
        
        List_of_Ranked_Assets = []
        Detailed_MetricsData_Array = []
        Special_Asset_Filter_Returns_Reference = 0

        for Asset_Col in range(len(Asset_Symbols)):

            Asset_parameters_Array, Asset_Detailed_MetricsData_Array, Combined_Metric_Array_Dim   = Retrieve_assets_Data("Get Metric Data", Asset_Col, Date_Row, Previous_price_Row, Current_Date, Lookback_Period, First_Date, Open_prices_Array, High_prices_Array, Low_prices_Array, Close_prices_Array, Adj_Close_prices_Array, Volume_Array, Asset_Symbols, Date_Array)
  
            if len(Asset_parameters_Array) > 0 and Combined_Metric_Array_Dim > 0:
                List_of_Ranked_Assets.append(Asset_parameters_Array)
                Detailed_MetricsData_Array.append(Asset_Detailed_MetricsData_Array)
            #else:
            #    print(f'Asset Symbol OUTXXXXX: {Asset_Symbols[Asset_Col]}')

        #if len(List_of_Ranked_Assets) == 0:
        #    print(Date_Array[Date_Row - Lookback_Period])
        #    print("Error")

        if Special_Filter_Status == True:
            List_of_Ranked_Assets = Functions.Special_Asset_Filter(List_of_Ranked_Assets, Special_Filter_Type)
            #print(List_of_Ranked_Assets)

        if Selected_method_of_Momuntum_Metric == "Metrics Count": 

            List_of_Ranked_Assets =  Functions.Get_Top_reformers_ByMetric_Count(List_of_Ranked_Assets, Combined_Metric_Array_Dim, Number_of_Portfolio_Allocations)


        elif Selected_method_of_Momuntum_Metric == "Metrics Combind": 

             List_of_Ranked_Assets =  Functions.Get_Top_reformers_ByMetric_Value(List_of_Ranked_Assets, Number_of_Portfolio_Allocations)
             
             #print (str(Current_Date))
             #print (str(Date_Array[Date_Row]))
             #print (str(List_of_Ranked_Assets))

    elif Method_of_Assets_Selection == "Momuntum Method with AI":

        List_of_Ranked_Assets = []
        Detailed_MetricsData_Array = []
        Special_Asset_Filter_Returns_Reference = 0

        for Asset_Col in range(len(Asset_Symbols)):

            Asset_parameters_Array,Asset_Detailed_MetricsData_Array, Asset_Volatility   = Retrieve_assets_Data("Get Raw Data", Asset_Col, Date_Row, Previous_price_Row, Current_Date, Lookback_Period, First_Date, Open_prices_Array, High_prices_Array, Low_prices_Array, Close_prices_Array, Adj_Close_prices_Array, Volume_Array, Asset_Symbols, Date_Array)
            
            if len(Asset_parameters_Array) > 0:

                Detailed_MetricsData_Array += Asset_Detailed_MetricsData_Array
                
                Output_Prediction = Asset_parameters_Array['prediction output']
                Input_Features_Names = [x for x in Asset_parameters_Array.columns if x not in ['prediction output']]
                Input_Features = Asset_parameters_Array[Input_Features_Names]

                X_train, X_test, y_train, y_test = train_test_split(Input_Features, Output_Prediction, train_size= 2 * len(Input_Features) // 3,shuffle=False)
               
                y_train_0 = (y_train.loc[lambda x : x == 0])
                y_train_1 = (y_train.loc[lambda x : x == 1]) 
                
                if y_train.values.sum()>0 and y_train_1.values.sum()!=len(y_train): # and y_test.sum(axis=0)>0 and y_test.sum(axis=1)>0:
                    
                    print("y train sum IN = "+ str(y_train.values.sum()))
                    print("y train 1 sum IN = "+ str(y_train_1.values.sum()))
                    print("len y_train IN = "+ str(len(y_train)))
                    
                    Random_Forest_model = Functions._train_random_forest(X_train, y_train, X_test, y_test)
                    KNearestNeighbor_Model = Functions._train_KNN(X_train, y_train, X_test, y_test)
                    GBC_Model =Functions._train_GBC(X_train, y_train, X_test, y_test)
                
                    Ensemble_Model = Functions._ensemble_model(Random_Forest_model, KNearestNeighbor_Model, GBC_Model, X_train, y_train, X_test, y_test)
        
                    Random_Forest_prediction = Random_Forest_model.predict(X_test)
                    KNN_prediction = KNearestNeighbor_Model.predict(X_test.values)
                    GBC_prediction = GBC_Model.predict(X_test)
                
                    Ensemble_prediction = Ensemble_Model.predict(X_test)

                    #print(confusion_matrix(y_test, prediction))

                    Random_Forest_Accuracy = accuracy_score(y_test.values, Random_Forest_prediction)
                    Knn_Accuracy = accuracy_score(y_test.values, KNN_prediction)
                    Ensemble_Accuracy = accuracy_score(y_test.values, Ensemble_prediction)

                    # Declare dictionary
                    Accuracy_model_Dict = {"Random_Forest":float(Random_Forest_Accuracy), "Knn":float(Knn_Accuracy), "Ensemble": float(Ensemble_Accuracy)}
                    Max_Accuracy_Model = max(Accuracy_model_Dict, key=Accuracy_model_Dict.get)

                    if Max_Accuracy_Model == "Random_Forest":
                        Prediction_DataFrame = pd.Series(Random_Forest_prediction)
                    elif Max_Accuracy_Model == "Knn":
                        Prediction_DataFrame = pd.Series(KNN_prediction)
                    elif Max_Accuracy_Model == "Ensemble":
                        Prediction_DataFrame = pd.Series(KNN_prediction)

                    # print(Prediction_DataFrame)
                
                    Count_of_Positive_Days = len([i for i in Prediction_DataFrame.tolist() if i == 1])
                
                    List_of_Ranked_Assets.append([Asset_Symbols[Asset_Col],Count_of_Positive_Days* max(Accuracy_model_Dict.values()), Volume_Array[Date_Row][Asset_Col], Asset_Volatility, Asset_Col, Current_Date])
                
                else:
                    
                    print("y train sum OUT = "+ str(y_train.values.sum()))
                    print("y train 1 sum OUT = "+ str(y_train_1.values.sum()))
                    print("len y_train OUT = "+ str(len(y_train)))

        #if len(List_of_Ranked_Assets)>=int(Number_of_Portfolio_Allocations):

        if len(List_of_Ranked_Assets)>0:

            List_of_Ranked_Assets.sort(key=lambda a: a[1], reverse=True)
            del List_of_Ranked_Assets[int(Number_of_Portfolio_Allocations):]
            
        # print("work still in prgoress .. ", sep='\n')


    return List_of_Ranked_Assets, Detailed_MetricsData_Array, Special_Asset_Filter_Returns_Reference
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------