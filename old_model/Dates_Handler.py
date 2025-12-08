
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
from datetime import datetime, date, timedelta
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
import Model_Input_Handler as Model_Input
import Parameters_Input_Handler as Parameters_Input
import Dates_Handler as Dates_Handler
import Active_Trades_Handler as Active_Trades_Handler
import Assets_Weightaging_Handler as Assets_Weightage
import Assets_Selection_Handler as Assets_Selection
import Output_Handler as Output_Generator
import Emergency_System_Handler as Emergency_Handler

Month_Days = 22
# 

# Parameters:
#-------------


def Dates_Handler(Request_Type, Rebalancing_Method, Current_Rebalancing_Date, Emergency_Status, Emergency_Status_Input_Code, Emergency_Asset_Input, Date_Row, Date_Array, Start_Row, End_Row, Start_Date, End_Date, Previous_Price_Row, First_Date, Lookback_Period, Equity_Lookback_Period, Bond_Lookback_Period, Number_of_Portfolio_Allocations, Asset_Prices_Array, Open_prices_Array, High_prices_Array, Low_prices_Array, Close_prices_Array, Adj_Close_prices_Array, Volume_Array, Asset_Symbols, Special_Asset_Filter_Returns_Reference, Total_Days_Pre_Allarm_ON_Equity, Total_Days_Pre_Allarm_ON_Bond, Total_Days_Allarm_ON, Equity_Signal_Wait_Period, Bond_Signal_Wait_Period, Equity_N_Bond_Hold_Period_Status, Equity_N_Bond_Hold_Period, Equity_Signal_Assets_List, Bond_Signal_Assets_List, BackUp_Emergency_Assets, Emergency_Assets, Reference_RiskFree_Asset, Emergency_Assets_Selected_Array,  Emergency_Assets_Weights_Array, Total_Emergnecy_Weight, Input_JSON, Backtest_Mode):
    
    if Request_Type == "Find Rebalancing Date":

        End_of_Emergency_Status = False
        Rebalancing_Date_Switch = False

        Emergency_Status_Previous =Emergency_Status

        Current_Date = datetime.strptime(str(Date_Array[Date_Row]), '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d')
        Current_Date = datetime.strptime(Current_Date, '%Y-%m-%d')

        Rebalancing_Date_Switch, Current_Rebalancing_Date, Emergency_Status, Emergency_Assets_Selected_Array, Total_Days_Pre_Allarm_ON_Equity, Total_Days_Pre_Allarm_ON_Bond, Total_Days_Allarm_ON, Emergency_Assets_Weights_Array, Total_Emergnecy_Weight, Emergency_Metric_Data_Array = Emergency_Handler.Emergency_Detector(Rebalancing_Method, Emergency_Status, Emergency_Status_Input_Code, Equity_Lookback_Period, Bond_Lookback_Period, Lookback_Period, Emergency_Asset_Input, Current_Date, Date_Row, Date_Array, Start_Row, End_Row, Start_Date, End_Date, Previous_Price_Row, First_Date, Number_of_Portfolio_Allocations, Asset_Prices_Array, Open_prices_Array, High_prices_Array, Low_prices_Array, Close_prices_Array, Adj_Close_prices_Array, Volume_Array, Asset_Symbols, Rebalancing_Date_Switch, Current_Rebalancing_Date, Special_Asset_Filter_Returns_Reference, Total_Days_Pre_Allarm_ON_Equity, Total_Days_Pre_Allarm_ON_Bond, Total_Days_Allarm_ON, Equity_Signal_Wait_Period, Bond_Signal_Wait_Period, Equity_N_Bond_Hold_Period_Status, Equity_N_Bond_Hold_Period, Equity_Signal_Assets_List, Bond_Signal_Assets_List, BackUp_Emergency_Assets, Emergency_Assets, Reference_RiskFree_Asset, Emergency_Assets_Selected_Array, Emergency_Assets_Weights_Array, Total_Emergnecy_Weight, Input_JSON, Backtest_Mode)

        if Emergency_Status != Emergency_Status_Previous and Emergency_Status == True:
            Start_of_Emergency_Status = True
            Current_Rebalancing_Date = Current_Date
        else:
            Start_of_Emergency_Status = False

        if Emergency_Status == False: 

            if Emergency_Status != Emergency_Status_Previous:
                End_of_Emergency_Status = True
            else:
                End_of_Emergency_Status = False

            LastMonth_Date = Functions.last_workday_of_month_New(Current_Date)

            if Date_Row <= len(Date_Array)-2:
                Next_Date = datetime.strptime(str(Date_Array[Date_Row+1]),'%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d') 
                Next_Date = datetime.strptime(Next_Date,'%Y-%m-%d')

                if Next_Date.month != Current_Date.month and LastMonth_Date != Current_Date:
                    LastMonth_Date = Current_Date
    
            if Rebalancing_Method == '2':    # Monthly Rebalance
                
                if  (Current_Date.day == LastMonth_Date.day or Emergency_Status !=Emergency_Status_Previous) and Current_Date.weekday() < 5: 

                    Rebalancing_Date_Switch = True
                    Current_Rebalancing_Date = Current_Date

            elif Rebalancing_Method == '1':  # Weekly Rebalance
        
                Week_Start_date = Functions.first_day_of_week(Current_Date)
                Weekly_Date = Week_Start_date + timedelta(days=4)

                print("Weekly Date = " + str(Weekly_Date) + " -- Current Date = " + str(Current_Date))

                if (Current_Date== Weekly_Date or Emergency_Status !=Emergency_Status_Previous) and Current_Date.weekday() < 5: # or Date_Row ==Start_Row or Date_Row ==End_Row'

                    Rebalancing_Date_Switch = True
                    Current_Rebalancing_Date = Current_Date

            elif Rebalancing_Method == '3':   # Active Rebalance

                Current_Active_Rebalancing_date = Active_Trades_Handler.Active_Trades_Dates_Generaters(Request_Type, Rebalancing_Method, Date_Row, Date_Array, Start_Row, End_Row, Start_Date, End_Date)
            
                if Current_Active_Rebalancing_date != "" and Date_Row ==Start_Row or Date_Row ==End_Row or Emergency_Status !=Emergency_Status_Previous:

                    Rebalancing_Date_Switch = True
                    Current_Rebalancing_Date = Current_Date
           

        return Rebalancing_Date_Switch, Current_Rebalancing_Date, Emergency_Status, Emergency_Assets_Selected_Array, Start_of_Emergency_Status, End_of_Emergency_Status, Total_Days_Pre_Allarm_ON_Equity, Total_Days_Pre_Allarm_ON_Bond, Total_Days_Allarm_ON, Emergency_Assets_Weights_Array, Total_Emergnecy_Weight, Emergency_Metric_Data_Array
    