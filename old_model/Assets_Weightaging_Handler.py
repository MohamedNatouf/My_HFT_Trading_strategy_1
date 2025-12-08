
import pandas as pd
import numpy as np
from scipy.ndimage.interpolation import shift
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
from pandas.core.tools.numeric import to_numeric

import yfinance
import yahoo_fin.stock_info as si

from scipy.optimize import minimize

import ta.momentum as ta_mum
import ta.volatility as ta_Volatility
import ta.utils as ta_ut
import ta.trend as ta_trend
import ta.volume as ta_Vol

import pypfopt
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

import btalib as bta_lib

import Functions as Functions
import Model_Input_Handler as Model_Input
import Parameters_Input_Handler as Parameters_Input
import Dates_Handler as Dates_Handler
import Active_Trades_Handler as Active_Trades_Handler

#from blueshift import blueshift
#from backtesting import backtesting
#from tensorflow.python.keras.models import sequential
#.discrete_allocation import DiscreteAllocation, get_latest_prices
#import scipy

from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
# from pypfopt.plotting import plot_efficient_frontier
from pypfopt import objective_functions



Save_Total_ETFs_List_file = 'Total ETFs List.txt'
Save_Total_ETFs_performance_List_file = 'Total ETFs Performance List.txt' 
Total_ETFs_List = []
TF_Check_in_List = False
Asset_performanc_Data_List = []
separator = ", "
List_of_Ranked_Assets = []
Asset_Inception_Date_Row_Array = []

High_Lookback_Period_Array = []
Low_Lookback_Period_Array = []
Open_Lookback_Period_Array = []
Close_Lookback_Period_Array = []
Adj_Close_Lookback_Period_Array = []

Portfolio_Value_Array= []
Portfolio_Returns_Array= []
Portfolio_Running_High_Array= []
Portfolio_Maximum_DrawDown_Array = []
Start_Simuation = False
First_Month = True 
Previous_date_price = 0
Portfolio_Value = 0

Month_Days = 22

# Assets Portfolio Allocations Weightage Unit:
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def Assets_Weightage_Handler(Date_Row, Previous_price_Row, Current_Rebalancing_Date, Type_of_Momentum_Method, Selected_method_of_Momuntum_Metric, Method_of_Assets_Selection, Lookback_Period, Number_of_Portfolio_Allocations, Method_Of_Wieghting, Volatility_Weight_Strength, Volume_Weight_Strength, Volatility_Measuring_Lookback_Period, List_of_Filtered_Assets, Asset_Prices_Array, Open_prices_Array, High_prices_Array, Low_prices_Array, Close_prices_Array, Adj_Close_prices_Array, Volume_Array, Asset_Symbols, Date_Array):       
    Assets_Col_list = []
    #Assets_Col_list = ""
    Assets_Weights_Array= []
    Total_Assets_Volatility = 0 
    Total_Assets_Market_Cap = 0
    Assets_Volatility_Weights_Count = 0 
    Total_Assets_Weight = 0

    #if len(List_of_Filtered_Assets)==int(Number_of_Portfolio_Allocations):

    if len(List_of_Filtered_Assets)>0:

       if Method_Of_Wieghting == "Assets Volume":

            for Filtered_Asset in range(len(List_of_Filtered_Assets)):
                if List_of_Filtered_Assets[Filtered_Asset][3] >0:

                    Currnet_date_price = Adj_Close_prices_Array[Date_Row][int(List_of_Filtered_Assets[Filtered_Asset][4])]
                    Total_Assets_Market_Cap = Total_Assets_Market_Cap + ((List_of_Filtered_Assets[Filtered_Asset][2])*Currnet_date_price)

            for Filtered_Asset in range(len(List_of_Filtered_Assets)):

                    Currnet_date_price = Adj_Close_prices_Array[Date_Row][int(List_of_Filtered_Assets[Filtered_Asset][4])]
                    MarketCap_Asset_Weight = ((Currnet_date_price*(List_of_Filtered_Assets[Filtered_Asset][2]))/Total_Assets_Market_Cap) * Volume_Weight_Strength

                    Assets_Weights_Array.append((MarketCap_Asset_Weight))
                    Total_Assets_Weight=Total_Assets_Weight + Assets_Weights_Array[-1]


       elif Method_Of_Wieghting == "Assets Inverse Volatility":

            for Filtered_Asset in range(len(List_of_Filtered_Assets)):
                if List_of_Filtered_Assets[Filtered_Asset][3] >0:

                    Total_Assets_Volatility = Total_Assets_Volatility + (1/ ((List_of_Filtered_Assets[Filtered_Asset][3]/100)))

                    Assets_Volatility_Weights_Count =  Assets_Volatility_Weights_Count +1

            for Filtered_Asset in range(len(List_of_Filtered_Assets)):
                if Assets_Volatility_Weights_Count >0:
                                
                    Volatility_Asset_Weight = (1/Total_Assets_Volatility)/(List_of_Filtered_Assets[Filtered_Asset][3]/100) * Volatility_Weight_Strength

                    Assets_Weights_Array.append((Volatility_Asset_Weight))
                    Total_Assets_Weight=Total_Assets_Weight + Assets_Weights_Array[-1]

       elif Method_Of_Wieghting == "Volume & Inverse Volume":

            for Filtered_Asset in range(len(List_of_Filtered_Assets)):
                if List_of_Filtered_Assets[Filtered_Asset][3] >0:

                    Currnet_date_price = Adj_Close_prices_Array[Date_Row][int(List_of_Filtered_Assets[Filtered_Asset][4])]
                    Total_Assets_Market_Cap = Total_Assets_Market_Cap + ((List_of_Filtered_Assets[Filtered_Asset][2])*Currnet_date_price)

                    Total_Assets_Volatility = Total_Assets_Volatility + (1/ ((List_of_Filtered_Assets[Filtered_Asset][3]/100)))
                    Assets_Volatility_Weights_Count =  Assets_Volatility_Weights_Count +1


            for Filtered_Asset in range(len(List_of_Filtered_Assets)):
                if Assets_Volatility_Weights_Count>0:
                                
                    Volatility_Asset_Weight = (1/Total_Assets_Volatility)/(List_of_Filtered_Assets[Filtered_Asset][3]/100) * Volatility_Weight_Strength

                    Currnet_date_price = Adj_Close_prices_Array[Date_Row][int(List_of_Filtered_Assets[Filtered_Asset][4])]
                    MarketCap_Asset_Weight = ((Currnet_date_price*(List_of_Filtered_Assets[Filtered_Asset][2]))/Total_Assets_Market_Cap) * Volume_Weight_Strength

                    Assets_Weights_Array.append((Volatility_Asset_Weight+MarketCap_Asset_Weight))
                    Total_Assets_Weight=Total_Assets_Weight + Assets_Weights_Array[-1]

       elif Method_Of_Wieghting == "Efficient Frontier Optimization":

           Efficient_Frontier_Optimization_Method = 2

           for Filtered_Asset in range(len(List_of_Filtered_Assets)):
               if List_of_Filtered_Assets[Filtered_Asset][4] >=0:
                   Assets_Col_list.append(List_of_Filtered_Assets[Filtered_Asset][4])

           Adj_Close_Lookback_Period_Previous_Array = pd.DataFrame(list(map(itemgetter(*Assets_Col_list),Adj_Close_prices_Array))[:Date_Row])

           if Efficient_Frontier_Optimization_Method == 2:

               # Calculate historical returns:
                returns = np.log(Adj_Close_Lookback_Period_Previous_Array /Adj_Close_Lookback_Period_Previous_Array.shift(1))

                # Calculate the covariance matrix:
                returns_cov_matrix = returns.cov()

                # Calculate expected returns:
                Mean_returns = returns.mean()

                # #Calculate the efficient frontier:
                num_portfolios = 1000
                results = Functions.efficient_frontier(Mean_returns, returns_cov_matrix, num_portfolios)

                # Manimize the risk to find the optimal weights:

                if len(List_of_Filtered_Assets) < Number_of_Portfolio_Allocations:
                    Obtimized_Assets_Count = len(List_of_Filtered_Assets)
                elif len(List_of_Filtered_Assets) == Number_of_Portfolio_Allocations:
                     Obtimized_Assets_Count = Number_of_Portfolio_Allocations


                optimal_weights = Functions.optimal_portfolio(Obtimized_Assets_Count, Mean_returns, returns_cov_matrix, returns)

                Assets_Weights_Array = optimal_weights
                Total_Assets_Weight = sum(optimal_weights)

           elif Efficient_Frontier_Optimization_Method == 2:
                
                min_weight = 0.1
                max_weight= 1

                # Calculate expected returns
                Mean_returns = expected_returns.mean_historical_return(Adj_Close_Lookback_Period_Previous_Array)

                # Calculate the covariance matrix
                returns_cov_matrix = risk_models.sample_cov(Adj_Close_Lookback_Period_Previous_Array)

                # Create the Efficient Frontier object
                Efficient_Frontier = EfficientFrontier(Mean_returns, returns_cov_matrix, weight_bounds=(min_weight,max_weight))

                raw_assets_weights = Efficient_Frontier.nonconvex_objective(objective_functions.sharpe_ratio, objective_args= (Efficient_Frontier.expected_returns, Efficient_Frontier.cov_matrix), weights_sum_to_one=True, constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1},  {"type": "ineq", "fun": lambda w: w - min_weight}, {"type": "ineq", "fun": lambda w: max_weight - w} ], ) 

                # Maximize the Sharpe ratio to find the optimal weights
                #raw_assets_weights = Efficient_Frontier.max_sharpe()
                #raw_assets_weights = Efficient_Frontier.min_volatility()

                # Clean the raw weights:
                #cleaned_assets_weights = Efficient_Frontier.clean_weights()

                #Assets_Weights_Array = cleaned_assets_weights
                Assets_Weights_Array = raw_assets_weights
                Total_Assets_Weight = sum(Assets_Weights_Array)


    return Assets_Weights_Array, Total_Assets_Weight
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------








