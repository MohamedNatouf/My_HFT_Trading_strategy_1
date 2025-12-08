

import pandas as pd
import numpy as np
import sys
import os
import ast
import re
import time
import math
import matplotlib.pyplot as plt
import matplotlib.dates as mdates  
import json

from dateutil.parser import parse
from datetime import datetime, date
from operator import itemgetter
from collections.abc import Mapping

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




def Portfolio_Simulator(Input_JSON):

    TF_Check_in_List = False
    separator = ", "
    #List_of_Ranked_Assets = []
    Asset_Inception_Date_Row_Array = []

    High_Lookback_Period_Array = []
    Low_Lookback_Period_Array = []
    Open_Lookback_Period_Array = []
    Close_Lookback_Period_Array = []
    Adj_Close_Lookback_Period_Array = []

    Assets_Weights_Array = [] 
    Portfolio_Value_Array= []
    Portfolio_Returns_Array= []
    Portfolio_Running_High_Array= []
    Portfolio_Maximum_DrawDown_Array = []
    Portfolio_Allocation_Output_Array = []
    Month_Days = 22
    Start_Simuation = False
    First_Date = True 
    Previous_date_price = 0
    Portfolio_Value = 0
    Start_Row = 0  
    End_Row =  0

    Rebalancing_Date_Switch = False

    Emergnecy_Parameters = ""

    Strategy_Output = []
    List_of_Filtered_Assets = [] 
    Assets_Weights_Array = []

    Emergency_Assets_Selected_Array = []
    Emergency_Assets_Weights_Array= []
    
    Total_Emergnecy_Weight = 0 


    if Input_JSON["sinceDate"]==Input_JSON["untilDate"]:
        Backtest_Mode= False
    elif Input_JSON["sinceDate"]!=Input_JSON["untilDate"]:
        Backtest_Mode= True

    if Input_JSON["state"] is not None and Input_JSON["state"] != "":

        Emergnecy_previous_Parameters =  json.loads(Input_JSON["state"])

        Emergency_Status = Emergnecy_previous_Parameters.get("emergency status")
        Total_Days_Pre_Allarm_ON_Equity = Emergnecy_previous_Parameters.get("Total Days Pre Allarm Equity")
        Total_Days_Pre_Allarm_ON_Bond = Emergnecy_previous_Parameters.get("Total Days Pre Allarm Bond")
        Total_Days_Allarm_ON = Emergnecy_previous_Parameters.get("Total Days Allarm ON")
    else:
        Emergency_Status = False
        Total_Days_Pre_Allarm_ON_Equity = 0
        Total_Days_Pre_Allarm_ON_Bond = 0
        Total_Days_Allarm_ON = 0


    Special_Asset_Filter_Returns_Reference = 0

    if Backtest_Mode == True:

        # Initiate Performance Charts:
        #-------------------------------------
        #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        Portfolio_Value_Dataframe = pd.DataFrame({'Dates': np.arange(0), 'Portfolio Value': np.zeros(0)})
        Portfolio_Returns_Dataframe = pd.DataFrame({'Dates': np.arange(0), 'Portfolio Returns': np.zeros(0)})
        Portfolio_Running_High_Dataframe = pd.DataFrame({'Dates': np.arange(0), 'Portfolio Running High': np.zeros(0)})
        Portfolio_Maximum_DrawDown_Dataframe = pd.DataFrame({'Dates': np.arange(0), 'Portfolio Maximum DrawDown': np.zeros(0)})

        #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        #figure, ax = plt.subplots(figsize=(10, 8))
        #line1, = ax.plot(x, y)
 

        #plt.title("Geeks For Geeks", fontsize=20)

        plt.ion()
        figure, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,8))


        #plt.tight_layout()

         
        # First subplot Chart 1: Portfolio Value
        line1, = axes[0,0].plot(Portfolio_Value_Dataframe['Dates'], Portfolio_Value_Dataframe['Portfolio Value']) #, marker='', linestyle='-')
        axes[0,0].set_title("Chart 1: Portfolio Value")
        axes[0,0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # Formatting as Year-Month
        axes[0,0].xaxis.set_major_locator(mdates.MonthLocator())  # Set major ticks to month
        axes[0,0].set_xlabel("Date (Months)")
        axes[0,0].set_ylabel("Portfolio Value")
        axes[0,0].tick_params(axis='x', rotation=90)
        #axes[0,0].grid(True)

        # Second subplot Chart 2: Portfolio Returns
        line2, = axes[0,1].plot(Portfolio_Returns_Dataframe['Dates'], Portfolio_Returns_Dataframe['Portfolio Returns']) #, marker='', linestyle='-')
        axes[0,1].set_title("Chart 2: Portfolio Returns")
        axes[0,1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # Formatting as Year-Month
        axes[0,1].xaxis.set_major_locator(mdates.MonthLocator())  # Set major ticks to month
        axes[0,1].set_xlabel("Date (Months)")
        axes[0,1].set_ylabel("Portfolio Returns")
        axes[0,1].tick_params(axis='x', rotation=90)
        #axes[0,1].grid(True)

        # Third subplot Chart 3: Portfolio Running High
        line3, = axes[1,0].plot(Portfolio_Running_High_Dataframe['Dates'], Portfolio_Running_High_Dataframe['Portfolio Running High']) #, marker='', linestyle='-')
        axes[1,0].set_title("Chart 3: Portfolio Running High")
        axes[1,0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # Formatting as Year-Month
        axes[1,0].xaxis.set_major_locator(mdates.MonthLocator())  # Set major ticks to month
        axes[1,0].set_xlabel("Date (Months)")
        axes[1,0].set_ylabel("Portfolio Running High")
        axes[1,0].tick_params(axis='x', rotation=90)
        #axes[1,0].grid(True)

        # Forth subplot Chart 4: Portfolio Maximum DrawDown
        line4, = axes[1,1].plot(Portfolio_Maximum_DrawDown_Dataframe['Dates'], Portfolio_Maximum_DrawDown_Dataframe['Portfolio Maximum DrawDown']) #, marker='', linestyle='-')
        axes[1,1].set_title("Chart 4: Portfolio Maximum DrawDown")
        axes[1,1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # Formatting as Year-Month
        axes[1,1].xaxis.set_major_locator(mdates.MonthLocator())  # Set major ticks to month
        axes[1,1].set_xlabel("Date (Months)")
        axes[1,1].set_ylabel("Portfolio Maximum DrawDown")
        axes[1,1].tick_params(axis='x', rotation=90)
        #axes[1,1].grid(True)

        figure.autofmt_xdate()

        plt.show()


    # Input:
    #-------
    # Parameters:
    #-------------
    
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    Model_Input_List = Model_Input.input_Entry_Handler(Input_JSON)
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    if len(Model_Input_List)==0:
        Start_Simuation == False  
    elif len(Model_Input_List)>0:
        [Start_Simuation, Start_Date, End_Date, Start_Row, End_Row, Asset_Prices_Array, Open_prices_Array, High_prices_Array, Low_prices_Array, Close_prices_Array, Adj_Close_prices_Array, Volume_Array, Asset_Symbols, Date_Array, Data_Array, Emergency_Status_Input_Code, Asset_Prices_Array_Emergency, Equity_Signal_Assets_List, Bond_Signal_Assets_List, BackUp_Emergency_Assets, Emergency_Assets, Reference_RiskFree_Asset] = Model_Input_List
       
        #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        Parameters_Input_List = Parameters_Input.Parameters_Input_Handler(Emergency_Status_Input_Code, Input_JSON)
        #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if len(Parameters_Input_List)==0:
            Start_Simuation == False  
        elif len(Parameters_Input_List)>0:
            [Method_of_Assets_Selection, Type_of_Momentum_Method, Selected_method_of_Momuntum_Metric, Number_of_Portfolio_Allocations, Initial_Invesment_Value, Method_Of_Wieghting, Volatility_Weight_Strength, Volume_Weight_Strength, Volatility_Measuring_Lookback_Period, Equity_Lookback_Period, Bond_Lookback_Period, Lookback_Period, Rebalancing_Method, Emergency_Status_Input_Code, Special_Filter_Status, Equity_Signal_Wait_Period, Bond_Signal_Wait_Period, Equity_N_Bond_Hold_Period_Status, Equity_N_Bond_Hold_Period, Special_Filter_Type] = Parameters_Input_List

        
    if Start_Simuation == True:  

        Previous_Price_Row = 0
        Detailed_MetricsData_Array_CSVFIle= []
        Emergency_Metric_Data_Array_CSVFIle= []

        Current_Rebalancing_Date = Start_Date

        if Backtest_Mode == True:
            print("Step 3: Simulation Date and Algo commence:")
            print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

        for Date_Row in range(Start_Row, End_Row+1): 

            #---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            Rebalancing_Date_Switch, Current_Rebalancing_Date, Emergency_Status, Emergency_Assets_Selected_Array, Start_of_Emergency_Status, End_of_Emergency_Status, Total_Days_Pre_Allarm_ON_Equity, Total_Days_Pre_Allarm_ON_Bond, Total_Days_Allarm_ON, Emergency_Assets_Weights_Array, Total_Emergnecy_Weight, Emergency_Metric_Data_Array = Dates_Handler.Dates_Handler("Find Rebalancing Date", Rebalancing_Method, Current_Rebalancing_Date, Emergency_Status, Emergency_Status_Input_Code, Asset_Prices_Array_Emergency, Date_Row, Date_Array, Start_Row, End_Row, Start_Date, End_Date, Previous_Price_Row, First_Date, Lookback_Period, Equity_Lookback_Period, Bond_Lookback_Period, int(Number_of_Portfolio_Allocations), Asset_Prices_Array, Open_prices_Array, High_prices_Array, Low_prices_Array, Close_prices_Array, Adj_Close_prices_Array, Volume_Array, Asset_Symbols, Special_Asset_Filter_Returns_Reference, Total_Days_Pre_Allarm_ON_Equity, Total_Days_Pre_Allarm_ON_Bond, Total_Days_Allarm_ON, Equity_Signal_Wait_Period, Bond_Signal_Wait_Period, Equity_N_Bond_Hold_Period_Status, Equity_N_Bond_Hold_Period, Equity_Signal_Assets_List, Bond_Signal_Assets_List, BackUp_Emergency_Assets, Emergency_Assets, Reference_RiskFree_Asset, Emergency_Assets_Selected_Array, Emergency_Assets_Weights_Array, Total_Emergnecy_Weight, Input_JSON, Backtest_Mode)
            #---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            
            Simulation_is_Running = True
           
            Entry_Statement = Rebalancing_Date_Switch == True or Start_of_Emergency_Status == True 

            Emergency_Metric_Data_Array_CSVFIle.append(Emergency_Metric_Data_Array)

            Emergnecy_Parameters = Functions.create_json_string(Emergency_Status, Total_Days_Pre_Allarm_ON_Equity, Total_Days_Pre_Allarm_ON_Bond, Total_Days_Allarm_ON)
            
            #print(Emergnecy_Parameters)

            if Entry_Statement == True: 

                Previous_List_of_Filtered_Assets = List_of_Filtered_Assets
                Previous_Assets_Weights_Array = Assets_Weights_Array

                if Emergency_Status == False:

                    if Date_Row >= Lookback_Period:
                   
                        # Assets Momentum Measuring Unit:
                        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                        List_of_Filtered_Assets, Detailed_MetricsData_Array, Special_Asset_Filter_Returns_Reference = Assets_Selection.Assets_Selection_Handler(Date_Row, Previous_Price_Row, Current_Rebalancing_Date, First_Date, Type_of_Momentum_Method, Selected_method_of_Momuntum_Metric, Method_of_Assets_Selection, Lookback_Period, int(Number_of_Portfolio_Allocations), Asset_Prices_Array, Open_prices_Array, High_prices_Array, Low_prices_Array, Close_prices_Array, Adj_Close_prices_Array, Volume_Array, Asset_Symbols, Date_Array, Special_Filter_Status, Special_Filter_Type)
                        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

                        # Assets Portfolio Allocations Weightage Unit:
                        #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                        Assets_Weights_Array, Total_Assets_Weight = Assets_Weightage.Assets_Weightage_Handler(Date_Row, Previous_Price_Row, Current_Rebalancing_Date, Type_of_Momentum_Method, Selected_method_of_Momuntum_Metric, Method_of_Assets_Selection, Lookback_Period, int(Number_of_Portfolio_Allocations), Method_Of_Wieghting, Volatility_Weight_Strength, Volume_Weight_Strength, Volatility_Measuring_Lookback_Period, List_of_Filtered_Assets, Asset_Prices_Array, Open_prices_Array, High_prices_Array, Low_prices_Array, Close_prices_Array, Adj_Close_prices_Array, Volume_Array, Asset_Symbols, Date_Array)
                        #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                        

                        if len(List_of_Filtered_Assets)>0 and len(Assets_Weights_Array) > 0: 

                            print(f'-----------------------------------------------------------------------------------------------------------------------', sep='\n') 

                            for Detailed_MetricsData in Detailed_MetricsData_Array:
                                if isinstance(Detailed_MetricsData, list):
                                    Detailed_MetricsData_Array_CSVFIle.append(Detailed_MetricsData)

                            for Filtered_Asset in range(len(List_of_Filtered_Assets)):

                                Currnet_price =  Adj_Close_prices_Array[Date_Row][int(List_of_Filtered_Assets[Filtered_Asset][4])]

                                print(f'Trades Opening Date (Normal Trade): {List_of_Filtered_Assets[Filtered_Asset][5]}', sep='\n')
                                print(f'Asset Symbol: {List_of_Filtered_Assets[Filtered_Asset][0]} ---- Asset Weight : {round(Assets_Weights_Array[Filtered_Asset]* 100, 4) } % ---- Asset Volume Value: {round(List_of_Filtered_Assets[Filtered_Asset][2], 3)} ---- Asset Volatility Value: {round(List_of_Filtered_Assets[Filtered_Asset][3]*100, 4)} % ---- Momentum Strength metric value: {List_of_Filtered_Assets[Filtered_Asset][1]} ----  Asset Trade Open Price: {round(Currnet_price, 3)}$', sep='\n') 
                    
                            print(f'-----------------------------------------------------------------------------------------------------------------------', sep='\n') 
                        
                            if Backtest_Mode == False:

                                Strategy_Output = Output_Generator.Output_Handler(Current_Rebalancing_Date, Date_Row, End_Row, List_of_Filtered_Assets=List_of_Filtered_Assets, Assets_Weights_Array=Assets_Weights_Array, Emergency_Status=Emergency_Status, Emergency_Assets_Selected_Array=Emergency_Assets_Selected_Array, Start_of_Emergency_Status=Start_of_Emergency_Status, End_of_Emergency_Status=End_of_Emergency_Status, Emergency_Assets_Weights_Array=Emergency_Assets_Weights_Array, Backtest_Mode=Backtest_Mode)

                                Portfolio_Allocation_Output_Array = Strategy_Output
    
                elif Emergency_Status == True and Start_of_Emergency_Status == True:              
                
                    if len(Asset_Prices_Array_Emergency) > 0 and len(Emergency_Assets_Weights_Array) > 0: 

                        print(f'-----------------------------------------------------------------------------------------------------------------------', sep='\n') 

                            #for Detailed_MetricsData in Detailed_MetricsData_Array:
                            #    if isinstance(Detailed_MetricsData, list):
                            #        Detailed_MetricsData_Array_CSVFIle.append(Detailed_MetricsData)

                        for Filtered_Asset in range(len(Emergency_Assets_Selected_Array)):

                            Asset_Prices_Array_Emergency = pd.DataFrame(Asset_Prices_Array_Emergency)   
                            Adj_Close_Price_Emergency_Array =Asset_Prices_Array_Emergency.get("Adj Close").values.tolist()
                            Currnet_price = Adj_Close_Price_Emergency_Array[Date_Row][Emergency_Assets_Selected_Array[Filtered_Asset][4]]

                            print(f'Trades Opening Date (Emergency Trade): {Emergency_Assets_Selected_Array[Filtered_Asset][5]}', sep='\n') 
                            print(f'Asset Symbol: {Emergency_Assets_Selected_Array[Filtered_Asset][0]} ---- Asset Weight : {round(Emergency_Assets_Weights_Array[Filtered_Asset]* 100, 4) } % ---- Asset Volume Value: {round(Emergency_Assets_Selected_Array[Filtered_Asset][2], 3)} ---- Asset Volatility Value: {round(Emergency_Assets_Selected_Array[Filtered_Asset][3]*100, 4)} % ---- Momentum Strength metric value: {Emergency_Assets_Selected_Array[Filtered_Asset][1]} ----  Asset Trade Open Price: {round(Currnet_price, 3)}$', sep='\n') 
                    
                        print(f'-----------------------------------------------------------------------------------------------------------------------', sep='\n') 
                        
                        if Backtest_Mode == False:

                            Strategy_Output = Output_Generator.Output_Handler(Current_Rebalancing_Date, Date_Row, End_Row, List_of_Filtered_Assets=Asset_Prices_Array_Emergency, Assets_Weights_Array=Emergency_Assets_Weights_Array, Emergency_Status=Emergency_Status, Emergency_Assets_Selected_Array=Emergency_Assets_Selected_Array, Start_of_Emergency_Status=Start_of_Emergency_Status, End_of_Emergency_Status=End_of_Emergency_Status, Emergency_Assets_Weights_Array=Emergency_Assets_Weights_Array, Backtest_Mode=Backtest_Mode)
                    
                            Portfolio_Allocation_Output_Array = Strategy_Output


                # Evalute Portfolio value

                if Backtest_Mode == True:

                    #for Detailed_EmergencyMetricsData in Emergency_Metric_Data_Array:
                    #    if isinstance(Detailed_EmergencyMetricsData, list):

                    if (len(Previous_List_of_Filtered_Assets) >0  and len(Previous_Assets_Weights_Array) > 0) \
                        or ((Start_of_Emergency_Status == True or End_of_Emergency_Status == True)  and Emergency_Assets_Selected_Array !=[]) \
                        or (First_Date == True and len(List_of_Filtered_Assets) >0 and len(Assets_Weights_Array) > 0) : 
                    
                        Portfolio_Temp_Value = 0         
                     
                        if First_Date == True:

                            Asset_Date_Returns=0

                            Portfolio_Value= Initial_Invesment_Value
                            Portfolio_Value_Array.append((Current_Rebalancing_Date,Portfolio_Value))

                            Portfolio_Returns = 0
                            Portfolio_Returns_Array.append((Current_Rebalancing_Date, Portfolio_Returns))

                            Running_High_Value= Initial_Invesment_Value
                            Portfolio_Running_High_Array.append((Current_Rebalancing_Date, Running_High_Value))

                            Maximum_DrawDown = 0
                            Portfolio_Maximum_DrawDown_Array.append((Current_Rebalancing_Date, Maximum_DrawDown))

                            Previous_Price_Row = Date_Row

                            Simulation_is_Running = True

                            print(f'-----------------------------------------------------------------------------------------------------------------------', sep='\n') 
                        
                            if (Start_of_Emergency_Status == True and Emergency_Assets_Selected_Array !=[]):
                                Assets_Weights_Array = []
                                Total_Assets_Weight = 0


                        elif First_Date == False:

                            if (Emergency_Status == False and End_of_Emergency_Status == False) or Start_of_Emergency_Status == True:

                                print(f'-----------------------------------------------------------------------------------------------------------------------', sep='\n')      
                        
                                print(f'Trades Closing Date (Normal Trade): {Current_Rebalancing_Date}', sep='\n')

                                for Filtered_Asset in range(len(Previous_List_of_Filtered_Assets)):
                                    Asset_Symbol = Previous_List_of_Filtered_Assets[Filtered_Asset][0]

                                    Currnet_date_price =  Adj_Close_prices_Array[Date_Row][int(Previous_List_of_Filtered_Assets[Filtered_Asset][4])]
                                    Previous_date_price = Adj_Close_prices_Array[Previous_Price_Row][int(Previous_List_of_Filtered_Assets[Filtered_Asset][4])]

                                    Asset_Date_Returns = (Currnet_date_price/Previous_date_price)-1
                                    Portfolio_Temp_Value = Portfolio_Temp_Value + (Portfolio_Value*(Previous_Assets_Weights_Array[Filtered_Asset])*(1+Asset_Date_Returns))

                                    print(f'Asset Symbol: {Previous_List_of_Filtered_Assets[Filtered_Asset][0]} ---- Asset Weight : {round(Previous_Assets_Weights_Array[Filtered_Asset]* 100, 4) } % ---- Asset Volume Value: {round(Previous_List_of_Filtered_Assets[Filtered_Asset][2], 3)} ---- Asset Volatility Value: {round(Previous_List_of_Filtered_Assets[Filtered_Asset][3]*100, 4)} % ---- Momentum Strength metric value: {round(Previous_List_of_Filtered_Assets[Filtered_Asset][1], 4)} ---- Asset Trade Close Price: {round(Currnet_date_price, 3)}$ ---- Asset Trade Open Price: {round(Previous_date_price, 3)}$ ----  Asset Return: {round((((Currnet_date_price/Previous_date_price)-1)*100),4)}%', sep='\n') 
                    
                                print(f'-----------------------------------------------------------------------------------------------------------------------', sep='\n') 
                        
                            elif End_of_Emergency_Status == True and Emergency_Assets_Selected_Array !=[] and Start_of_Emergency_Status == False:

                                print(f'-----------------------------------------------------------------------------------------------------------------------', sep='\n')      
                        
                                print(f'Trades Closing Date (Emergency Trade): {Current_Rebalancing_Date}', sep='\n')

                                for Filtered_Asset in range(len(Emergency_Assets_Selected_Array)):
                                    Asset_Symbol = Emergency_Assets_Selected_Array[Filtered_Asset][0]

                                    Asset_Prices_Array_Emergency = pd.DataFrame(Asset_Prices_Array_Emergency)   
                                    Adj_Close_Price_Emergency_Array =Asset_Prices_Array_Emergency.get("Adj Close").values.tolist()
                                
                                    if np.isnan(Adj_Close_Price_Emergency_Array[Date_Row][int(Emergency_Assets_Selected_Array[Filtered_Asset][4])]) == False:
                                        Currnet_date_price =  Adj_Close_Price_Emergency_Array[Date_Row][int(Emergency_Assets_Selected_Array[Filtered_Asset][4])]
                                    else:
                                         Currnet_date_price =  Adj_Close_Price_Emergency_Array[Date_Row-1][int(Emergency_Assets_Selected_Array[Filtered_Asset][4])]

                                    #print(f'Currnet date price: {Currnet_date_price}')

                                    Asset_Prices_Array_Emergency = pd.DataFrame(Asset_Prices_Array_Emergency)   
                                    Adj_Close_Price_Emergency_Array =Asset_Prices_Array_Emergency.get("Adj Close").values.tolist()
                                
                                    if np.isnan(Adj_Close_Price_Emergency_Array[Previous_Price_Row][int(Emergency_Assets_Selected_Array[Filtered_Asset][4])]) == False:
                                        Previous_date_price = Adj_Close_Price_Emergency_Array[Previous_Price_Row][int(Emergency_Assets_Selected_Array[Filtered_Asset][4])]
                                    else:
                                        Previous_date_price = Adj_Close_Price_Emergency_Array[Previous_Price_Row-1][int(Emergency_Assets_Selected_Array[Filtered_Asset][4])]

                                    Previous_date_price = Adj_Close_Price_Emergency_Array[Previous_Price_Row][int(Emergency_Assets_Selected_Array[Filtered_Asset][4])]
                                
                                    Asset_Date_Returns = (Currnet_date_price/Previous_date_price)-1

                                    Portfolio_Temp_Value = Portfolio_Temp_Value + (Portfolio_Value*(Emergency_Assets_Weights_Array[Filtered_Asset]*(1+Asset_Date_Returns)))

                                    print(f'Asset Symbol: {Asset_Symbol} ---- Asset Weight : {round(Emergency_Assets_Weights_Array[Filtered_Asset]* 100, 4) } % ---- Asset Volume Value: {round(Emergency_Assets_Selected_Array[Filtered_Asset][2], 3)} ---- Asset Volatility Value: {round(Emergency_Assets_Selected_Array[Filtered_Asset][3]*100, 4)} % ---- Momentum Strength metric value: {Emergency_Assets_Selected_Array[Filtered_Asset][1]} ---- Asset Trade Close Price: {round(Currnet_date_price, 3)}$ ---- Asset Trade Open Price: {round(Previous_date_price, 3)}$ ----  Asset Return: {round((Asset_Date_Returns*100),4)}%', sep='\n') 
                    
                                print(f'-----------------------------------------------------------------------------------------------------------------------', sep='\n') 

                            Portfolio_Value=Portfolio_Temp_Value
                            Portfolio_Value_Array.append((Current_Rebalancing_Date,Portfolio_Value))

                            Portfolio_Returns = (Portfolio_Value/Portfolio_Value_Array[-2][1])-1
                            Portfolio_Returns_Array.append((Current_Rebalancing_Date, Portfolio_Returns))

                            if Portfolio_Value > Running_High_Value:
                                Running_High_Value= Portfolio_Value
                                Portfolio_Running_High_Array.append((Current_Rebalancing_Date, Running_High_Value))

                            Maximum_DrawDown =  - (Running_High_Value - Portfolio_Value_Array[-1][1])/Running_High_Value
                            Portfolio_Maximum_DrawDown_Array.append((Current_Rebalancing_Date, Maximum_DrawDown))
               
                            Previous_Price_Row = Date_Row

                            Simulation_is_Running = True

                        Strategy_Output = Output_Generator.Output_Handler(Current_Rebalancing_Date, Date_Row, End_Row, Portfolio_Value_Array, Portfolio_Returns_Array, Portfolio_Running_High_Array, Portfolio_Maximum_DrawDown_Array, List_of_Filtered_Assets, Assets_Weights_Array, Total_Assets_Weight, Simulation_is_Running, Emergency_Status, Emergency_Assets_Selected_Array, Start_of_Emergency_Status, End_of_Emergency_Status, Emergency_Assets_Weights_Array, Backtest_Mode)

                        Portfolio_Allocation_Output_Array.append(Strategy_Output) 

                        if First_Date == True:
                            First_Date = False

                        Portfolio_Value_Dataframe = pd.DataFrame(Portfolio_Value_Array)
                        Portfolio_Returns_Dataframe = pd.DataFrame(Portfolio_Returns_Array)
                        Portfolio_Running_High_Dataframe = pd.DataFrame(Portfolio_Running_High_Array)
                        Portfolio_Maximum_DrawDown_Dataframe = pd.DataFrame(Portfolio_Maximum_DrawDown_Array)

                        # Update Performance Charts:
                        #-------------------------------------
                        #--------------------------------------

                        line1.set_xdata(Portfolio_Value_Dataframe[0])
                        line1.set_ydata(Portfolio_Value_Dataframe[1])

                        line2.set_xdata(Portfolio_Returns_Dataframe[0])
                        line2.set_ydata(Portfolio_Returns_Dataframe[1])

                        line3.set_xdata(Portfolio_Running_High_Dataframe[0])
                        line3.set_ydata(Portfolio_Running_High_Dataframe[1])

                        line4.set_xdata(Portfolio_Maximum_DrawDown_Dataframe[0])
                        line4.set_ydata(Portfolio_Maximum_DrawDown_Dataframe[1])

                        # Rescale the plot based on the updated data

                        axes[0,0].relim()
                        axes[0,0].autoscale_view(True,True,True)

                        if len(axes[0,0].xaxis.get_minorticklocs()) >= 30:
                            axes[0,0].xaxis.set_minor_locator(mdates.MonthLocator(interval=3))


                        axes[0,1].relim()
                        axes[0,1].autoscale_view(True,True,True)

                        if len(axes[0,1].xaxis.get_minorticklocs()) >= 30:
                            axes[0,1].xaxis.set_minor_locator(mdates.MonthLocator(interval=3))


                        axes[1,0].relim()
                        axes[1,0].autoscale_view(True,True,True)

                        if len(axes[1,0].xaxis.get_minorticklocs()) >= 30:
                            axes[1,0].xaxis.set_minor_locator(mdates.MonthLocator(interval=3))


                        axes[1,1].relim()
                        axes[1,1].autoscale_view(True,True,True)

                        if len(axes[1,1].xaxis.get_minorticklocs()) >= 30:
                            axes[1,1].xaxis.set_minor_locator(mdates.MonthLocator(interval=3))


                        plt.draw()
                        plt.pause(0.1) 
                        ##plt.ioff() 
                        plt.show()

                        #figure.autofmt_xdate()
                        #x_major_ticks = axes[0,0].xaxis.get_majorticklocs()
                        #x_minor_ticks = axes[0,0].xaxis.get_minorticklocs()

                        #ax.xaxis.set_major_locator(mdates.YearLocator())
                        #ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

                        ## Optionally, set minor x-axis ticks to monthly or quarterly interval

                        #ax.xaxis.set_minor_formatter(mdates.DateFormatter('%b')) 

                        ## Rotate and align the tick labels so they look better
                        #fig.autofmt_xdate()

                        #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


                        #ax.xaxis.set_major_locator(mdates.YearLocator())
                        #ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

                        #ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=3))  # Every 3 months
                        #ax.xaxis.set_minor_formatter(mdates.DateFormatter('%b'))  # Month abbreviation

                        ## Rotate and align the tick labels so they look better
                        #fig.autofmt_xdate()


            if Date_Row + 1 > len(Date_Array) - 1:
                break


    else:
        print(" \n Error in model's inputs and parameters. Please review all inputs and run the software again. This run is terminated.")

        Portfolio_Allocation_Output_Array = []
        Emergnecy_Parameters = ""

       
    # print backtest output data to excel CSV:

    if Start_Simuation == True and Backtest_Mode == True:

        Simulation_is_Running = False

        Strategy_Output = Output_Generator.Output_Handler(Current_Rebalancing_Date, Date_Row, End_Row, Portfolio_Value_Array, Portfolio_Returns_Array, Portfolio_Running_High_Array, Portfolio_Maximum_DrawDown_Array, List_of_Filtered_Assets, Assets_Weights_Array, Total_Assets_Weight, Simulation_is_Running, Emergency_Assets_Weights_Array, Backtest_Mode=Backtest_Mode)

        Input_parameters_record_list = [['Start Date:', Start_Date], ['End Date:', End_Date], ['Method of Assets Selection:', Method_of_Assets_Selection], ['Type of Momentum Method:', Type_of_Momentum_Method], ['Selected method of Momuntum Metric:', Selected_method_of_Momuntum_Metric], ['Number of Portfolio Allocations:', Number_of_Portfolio_Allocations], ['Rebalancing Method:', Rebalancing_Method], ['Initial Invesment Value:', Initial_Invesment_Value], ['Method Of Wieghting:', Method_Of_Wieghting], ['Volatility Weight Strength:', Volatility_Weight_Strength], ['Volume Weight Strength:', Volume_Weight_Strength], ['Volatility Measuring Lookback Period:', Volatility_Measuring_Lookback_Period], ['Lookback Period:', Lookback_Period]]
        Input_parameters_record_Dataframe = pd.DataFrame(Input_parameters_record_list, columns=['Parameter', 'Value of Parameter'])

        Portfolio_Allocation_Output_Reformated_Array = []
        for allocation_Date in Portfolio_Allocation_Output_Array:
            Date_Allocation_Output_Reformated_Array = []
            Date_Allocation_Output_Reformated_Array.append(str(allocation_Date[0][0]))
            for allocation in allocation_Date[0][1]:
                Date_Allocation_Output_Reformated_Array.extend(allocation)
            Portfolio_Allocation_Output_Reformated_Array.append(Date_Allocation_Output_Reformated_Array)
        Strategy_Output_Allocations_Dataframe = pd.DataFrame(Portfolio_Allocation_Output_Reformated_Array)

        Portfolio_Value_Dataframe = pd.DataFrame(Portfolio_Value_Array, columns=['Rebalancing Date:', 'Portfolio Value:'])
        Portfolio_Returns_Dataframe = pd.DataFrame(Portfolio_Returns_Array, columns=['Rebalancing Date:', 'Portfolio returns:'])
        Portfolio_Running_High_Dataframe = pd.DataFrame(Portfolio_Running_High_Array, columns=['Rebalancing Date:', 'Portfolio Running High Value:'])
        Portfolio_Maximum_DrawDown_Dataframe = pd.DataFrame(Portfolio_Maximum_DrawDown_Array, columns=['Rebalancing Date:', 'Portfolio Maximum DrawDown:'])
        Strategy_Output_TimeSeries_Dataframe = Portfolio_Value_Dataframe
        Strategy_Output_TimeSeries_Dataframe = Strategy_Output_TimeSeries_Dataframe.join(Portfolio_Returns_Dataframe["Portfolio returns:"]) 
        Strategy_Output_TimeSeries_Dataframe = Strategy_Output_TimeSeries_Dataframe.join(Portfolio_Running_High_Dataframe["Portfolio Running High Value:"]) 
        Strategy_Output_TimeSeries_Dataframe = Strategy_Output_TimeSeries_Dataframe.join(Portfolio_Maximum_DrawDown_Dataframe["Portfolio Maximum DrawDown:"]) 

        if Input_JSON['Data_Source'] != "AlgoMart":
            Input_parameters_record_Dataframe.to_csv('tmp/Input_Parameters_Record.csv', index=False)
            Strategy_Output_TimeSeries_Dataframe.to_csv('tmp/Strategy_Output_TimeSeries.csv', index=False)
            Strategy_Output_Allocations_Dataframe.to_csv('tmp/Strategy_Output_Allocations.csv', index=False)

            Detailed_MetricsData_Array_CSV_Dataframe = pd.DataFrame(Detailed_MetricsData_Array_CSVFIle)
            Emergency_Metric_Data_Array_CSV_Dataframe = pd.DataFrame(Emergency_Metric_Data_Array_CSVFIle)

            if Backtest_Mode == False: 
                Detailed_MetricsData_Array_CSV_Dataframe.to_csv('tmp/Detailed Metrics Data.csv', index=False)
            if Backtest_Mode == True:
                Detailed_MetricsData_Array_CSV_Dataframe.to_csv('tmp/Detailed Metrics Data2.csv', index=False)
       
            if Backtest_Mode == False: 
                Emergency_Metric_Data_Array_CSV_Dataframe.to_csv('tmp/Detailed Metrics Emergency Data.csv', index=False)
            if Backtest_Mode == True:
                Emergency_Metric_Data_Array_CSV_Dataframe.to_csv('tmp/Detailed Metrics Emergency Data2.csv', index=False)



        plt.draw()
        plt.pause(0.1) 
        #plt.ioff() 
        plt.show()

    return Portfolio_Allocation_Output_Array, Emergnecy_Parameters





