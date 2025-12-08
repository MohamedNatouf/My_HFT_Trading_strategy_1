

import pandas as pd
import numpy as np
import sys
import os
import ast
import re
import time
import math
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
from dateutil.parser import parse
from datetime import datetime, date
from operator import itemgetter

import yfinance
import yahoo_fin.stock_info as si

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

#from blueshift import blueshift
#from backtesting import backtesting
#from tensorflow.python.keras.models import sequential
#.discrete_allocation import DiscreteAllocation, get_latest_prices
#import scipy

Month_Days = 22


def Output_Handler(Current_Rebalancing_Date, Date_Row, End_Row, Portfolio_Value_Array=[], Portfolio_Returns_Array=[], Portfolio_Running_High_Array=[], Portfolio_Maximum_DrawDown_Array=[], List_of_Filtered_Assets=[], Assets_Weights_Array=[], Total_Assets_Weight=1, Simulation_is_Running=True, Emergency_Status= False, Emergency_Assets_Selected_Array=[], Start_of_Emergency_Status= False, End_of_Emergency_Status= False, Emergency_Assets_Weights_Array=[], Backtest_Mode=False):

    Assets_Positions_Lists = []

    if Backtest_Mode == True:

        Portfolio_Allocation_Output = []

        Portfolio_Value_Dataframe = pd.DataFrame(Portfolio_Value_Array)
        Portfolio_Returns_Dataframe = pd.DataFrame(Portfolio_Returns_Array)
        Portfolio_Running_High_Dataframe = pd.DataFrame(Portfolio_Running_High_Array)
        Portfolio_Maximum_DrawDown_Dataframe = pd.DataFrame(Portfolio_Maximum_DrawDown_Array)

        if Simulation_is_Running == True:

            # Calculate Portfolios Metrics :

            Investment_Year_Counts=(Portfolio_Value_Array[-1][0].year - Portfolio_Value_Array[0][0].year) +1
            Portfolio_CAGR=pow((Portfolio_Value_Array[-1][1]/Portfolio_Value_Array[0][1]),(1/Investment_Year_Counts))-1
            Cumulative_Returns = ((Portfolio_Value_Array[-1][1] - Portfolio_Value_Array[0][1]) / Portfolio_Value_Array[0][1])
            Portfolio_Returns_Volatility = (Portfolio_Returns_Dataframe.std().values.tolist())[1] * math.sqrt(12)
            Portfolio_Maximum_DrawDown = min(Portfolio_Maximum_DrawDown_Array, key=lambda tup: tup[1])[1]


            # Add New Positions to Allocations:

            if Start_of_Emergency_Status == False:
                for Filtered_Asset in range(len(List_of_Filtered_Assets)):
                    if list(Assets_Weights_Array)[Filtered_Asset] >0:
                        Assets_Positions_Lists.append([List_of_Filtered_Assets[Filtered_Asset][0], list(Assets_Weights_Array)[Filtered_Asset]])
            
            elif Start_of_Emergency_Status == True:
                for Filtered_Asset in range(len(Emergency_Assets_Selected_Array)):
                    if Emergency_Assets_Weights_Array[Filtered_Asset]>0:
                        Total_Assets_Weight = Total_Assets_Weight + Emergency_Assets_Weights_Array[Filtered_Asset]
                        Assets_Positions_Lists.append([Emergency_Assets_Selected_Array[Filtered_Asset][0], Emergency_Assets_Weights_Array[Filtered_Asset]])
            
            Portfolio_Allocation_Output.append([Current_Rebalancing_Date.strftime('%Y-%m-%d'), Assets_Positions_Lists])


            if len(Portfolio_Value_Array) > 0:
                print(f'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX', sep='\n')  
                print(f'Trading/Rebalancing Date: {Portfolio_Value_Array[-1][0]}', sep='\n')
                print(f'Total Assets Weight= {round(Total_Assets_Weight, 3) * 100} %', sep='\n') 
                print(f'Portfolio Value: {Portfolio_Value_Array[-1][1]} ---- Portfolio Return : {round(Portfolio_Returns_Array[-1][1], 3) * 100} %', sep='\n') 
                print(f'Portfolio CAGR: {round(Portfolio_CAGR, 5) * 100}% ---- Portfolio Cumulative Returns: {round(Cumulative_Returns, 5) * 100}%  ---- Total number of Investment Years: {Investment_Year_Counts}', sep='\n') 
                print(f'Portfolio Returns Volatility : {round(Portfolio_Returns_Volatility, 3) * 100} % ---- Portfolio Maximum DrawDown : {round(Portfolio_Maximum_DrawDown*100, 3)  } %', sep='\n') 
                print(f'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX', sep='\n') 
        
        elif Simulation_is_Running == False:

            print(f'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX', sep='\n')  
            print(f'--------End of Simulation-----', sep='\n') 
            print(f'Simulation Start Date: {Portfolio_Value_Array[0][0]}', sep='\n')
            print(f'Simulation End Date: {Portfolio_Value_Array[-1][0]}', sep='\n')
            print(f'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX', sep='\n') 

           
            
        return Portfolio_Allocation_Output

    elif Backtest_Mode == False:

        # Add New Positions to Allocations:

        Portfolio_Allocation_Output = []

        if len(List_of_Filtered_Assets)>0:

            if Start_of_Emergency_Status == False:
                for Filtered_Asset in range(len(List_of_Filtered_Assets)):
                    if list(Assets_Weights_Array)[Filtered_Asset] >0:
                        Assets_Positions_Lists.append([List_of_Filtered_Assets[Filtered_Asset][0], list(Assets_Weights_Array)[Filtered_Asset]])
            
            elif Start_of_Emergency_Status == True:
                for Filtered_Asset in range(len(Emergency_Assets_Selected_Array)):
                    if Emergency_Assets_Weights_Array[Filtered_Asset]>0:
                        Total_Assets_Weight = Total_Assets_Weight + Emergency_Assets_Weights_Array[Filtered_Asset]
                        Assets_Positions_Lists.append([Emergency_Assets_Selected_Array[Filtered_Asset][0], Emergency_Assets_Weights_Array[Filtered_Asset]])
            
            Portfolio_Allocation_Output.append([Current_Rebalancing_Date.strftime('%Y-%m-%d'), Assets_Positions_Lists])


        else:
            Portfolio_Allocation_Output = []
    
        return Portfolio_Allocation_Output