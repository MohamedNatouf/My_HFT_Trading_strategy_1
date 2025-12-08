
import pandas as pd
import numpy as np

from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

import matplotlib.pyplot as plt
import Assets_Weightaging_Handler as Assets_Weightage
from dateutil.parser import parse
from datetime import datetime, date
from operator import itemgetter

import Functions as Functions

def Emergency_Detector(Rebalancing_Method, Emergency_Status, Emergency_Status_Input_Code, Equity_Lookback_Period, Bond_Lookback_Period, Normal_Lookback_Period, Asset_Prices_Array_Emergency, Current_Date, Date_Row, Date_Array, Start_Row, End_Row, Start_Date, End_Date, Previous_Price_Row, First_Date,  Selected_Assets_CountNumber_of_Portfolio_Allocations, Asset_Prices_Array, Open_prices_Array, High_prices_Array, Low_prices_Array, Close_prices_Array, Adj_Close_prices_Array, Volume_Array, Asset_Symbols, Rebalancing_Date_Switch, Current_Rebalancing_Date, Special_Asset_Filter_Returns_Reference,Total_Days_Pre_Allarm_ON_Equity, Total_Days_Pre_Allarm_ON_Bond, Total_Days_Allarm_ON, Equity_Signal_Wait_Period, Bond_Signal_Wait_Period, Equity_N_Bond_Hold_Period_Status, Equity_N_Bond_Hold_Period, Equity_Signal_Assets_List, Bond_Signal_Assets_List, BackUp_Emergency_Assets, Emergency_Assets, Reference_RiskFree_Asset, Emergency_Assets_Selected_Array, Emergency_Assets_Weights_Array, Total_Emergnecy_Weight, Input_JSON, Backtest_Mode):

    Equity_Alarm_Signal = "Signal OFF"
    Bonds_Alarm_Signal = "Signal OFF"
    
    Asset_Symbols_Emergency_AdjClose=pd.DataFrame(Asset_Prices_Array_Emergency).get("Adj Close").columns.tolist()

    Equity_Lookback_Date = Asset_Prices_Array_Emergency.index[Date_Row - Equity_Lookback_Period]
    Bond_Lookback_Date = Asset_Prices_Array_Emergency.index[Date_Row - Bond_Lookback_Period]

    Adj_Close_prices_Array = Asset_Prices_Array_Emergency.get("Adj Close").values.tolist()

    Emergency_Assets_Array = []
    Emergency_Metric_Data_Array = []

    Method_of_Emergncy_Signal  = "Equity_N_Bond_Emergency_Signal"

    Margin_RiskFree_Throshold = 0.01

    Emergency_Alocation_Asset_Selected = False
    Reached_Equity_Check = False 
    Reached_Bond_Check = False 
    previous_Emergency_Status = Emergency_Status

    # Wieghting method parameters:
    Method_of_Assets_Selection = "Price Momuntum Strategy"
    Type_of_Momentum_Method = "Cross_Sectional Momuntum"
    Selected_method_of_Momuntum_Metric = "Metrics Combind"
    Number_of_Emergency_Portfolio_Allocations = 3
    Wieghting_ON_Code = 4
    if  (Wieghting_ON_Code == 1 or Wieghting_ON_Code == 2 or Wieghting_ON_Code == 3 or Wieghting_ON_Code == 4) :
        if int(Wieghting_ON_Code) == 1:
            Method_Of_Wieghting = "Assets Inverse Volatility"
            Volatility_Weight_Strength =1
            Volume_Weight_Strength = 1- Volatility_Weight_Strength
        elif int(Wieghting_ON_Code) == 2:
            Method_Of_Wieghting = "Assets Volume"
            Volatility_Weight_Strength =0
            Volume_Weight_Strength = 1- Volatility_Weight_Strength
        elif int(Wieghting_ON_Code) == 3:
            Method_Of_Wieghting = "Volume & Inverse Volume"
            Volatility_Weight_Strength =0.6
            Volume_Weight_Strength = 1- Volatility_Weight_Strength
        elif int(Wieghting_ON_Code) == 4:
            Method_Of_Wieghting = "Efficient Frontier Optimization"
            Volatility_Weight_Strength =0
            Volume_Weight_Strength = 0
        else:
            Method_Of_Wieghting = "Assets Inverse Volatility"
            Volatility_Weight_Strength =1
            Volume_Weight_Strength = 1- Volatility_Weight_Strength
    else:
        Method_Of_Wieghting = "Assets Inverse Volatility"
        Volatility_Weight_Strength =1
        Volume_Weight_Strength = 1- Volatility_Weight_Strength

    # Lookback Periods parameters:
    Days_Per_Month = 22
    Volatility_Measuring_Lookback_Period = 12
    Lookback_Period_Code = 3
    Lookback_Period = Days_Per_Month * Lookback_Period_Code 


    if Emergency_Status_Input_Code == 1:
        
        if Method_of_Emergncy_Signal == "Equity_N_Bond_Emergency_Signal":

            Min_Equity_Asset_Total_Return = 10000
            Emergency_Equity_Asset_Total_Return = -10000

            Min_Bond_Asset_Total_Return = 10000
            Emergency_Bond_Asset_Total_Return = -10000

            Min_Equity_Asset = ""
            Min_Bond_Asset = ""

            Emergency_Asset_Array = []

            #print("Current Date = " + str(Asset_Prices_Array_Emergency.index[Date_Row]))
            #print("Lockback Equity Date = " + str(Asset_Prices_Array_Emergency.index[Date_Row - Equity_Lookback_Period]))
            #print("Lockback Bond Date = " + str(Asset_Prices_Array_Emergency.index[Date_Row - Bond_Lookback_Period]))

            #print("Current Emergency assets = " + Emergency_Assets_List)
            #print("Current BackUp Emergency assets = " + BackUp_Emergency_Assets_List)

            if Reference_RiskFree_Asset[0] in Asset_Symbols_Emergency_AdjClose: 

                # Current Risk Free Measured Asset's Yield :
                Currnet_Date_Risk_Free_Yield =  Asset_Prices_Array_Emergency.loc[Current_Date, ("Adj Close", Reference_RiskFree_Asset[0])] 

                if np.isnan(Currnet_Date_Risk_Free_Yield) == True:
                    OneDateBefore_Date = datetime.strptime(str(Date_Array[Date_Row-1]), '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d')
                    OneDateBefore_Date = datetime.strptime(OneDateBefore_Date, '%Y-%m-%d')
                    Currnet_Date_Risk_Free_Yield =  Asset_Prices_Array_Emergency.loc[OneDateBefore_Date, ("Adj Close", Reference_RiskFree_Asset[0])] 


                # Previous lookback Emergency Measured Asset's Price :
                LookBack_Date_Risk_Free_Yield = Asset_Prices_Array_Emergency.loc[Bond_Lookback_Date, ("Adj Close", Reference_RiskFree_Asset[0])]  

                if np.isnan(Currnet_Date_Risk_Free_Yield) == False and np.isnan(LookBack_Date_Risk_Free_Yield) == False:
                    Risk_Free_Returns=  (Currnet_Date_Risk_Free_Yield - LookBack_Date_Risk_Free_Yield)/100
                else:
                    Risk_Free_Returns=0

                #print(Currnet_Date_Risk_Free_Yield)
                #print(LookBack_Date_Risk_Free_Yield)
                #print(Risk_Free_Returns)

            for Emergency_Asset_Col in range(len(Asset_Symbols_Emergency_AdjClose)):

                if Asset_Symbols_Emergency_AdjClose[Emergency_Asset_Col] in Equity_Signal_Assets_List:

                    #print(Asset_Symbols_Emergency_AdjClose[Emergency_Asset_Col])
                   
                    Current_Date_Emergency_Asset_Price = Asset_Prices_Array_Emergency.loc[Current_Date, ("Adj Close", Asset_Symbols_Emergency_AdjClose[Emergency_Asset_Col])] 
                    LookBack_Date_Emergency_Asset_Price = Asset_Prices_Array_Emergency.loc[Equity_Lookback_Date, ("Adj Close", Asset_Symbols_Emergency_AdjClose[Emergency_Asset_Col])] 

                    #print("Current Equity = " + Current_Date_Emergency_Asset_Price)
                    #print("Lookbook Equity = " + LookBack_Date_Emergency_Asset_Price)

                    if np.isnan(Current_Date_Emergency_Asset_Price) == False and np.isnan(LookBack_Date_Emergency_Asset_Price) == False:   

                        #print(f'Asset Symbol: {Asset_Symbols_Emergency[Emergency_Asset_Col]} ---- Emergency asset : {Emergency_Assets_List}')

                        if LookBack_Date_Emergency_Asset_Price != 0:
                            Emergency_Equity_Asset_Total_Return = ((Current_Date_Emergency_Asset_Price/LookBack_Date_Emergency_Asset_Price) - 1)

                            Reached_Equity_Check = True 

                            if Emergency_Equity_Asset_Total_Return != -10000 and Emergency_Equity_Asset_Total_Return < Min_Equity_Asset_Total_Return and Emergency_Equity_Asset_Total_Return > -1:
                                Min_Equity_Asset_Total_Return = Emergency_Equity_Asset_Total_Return
                                Min_Equity_Asset = Asset_Symbols_Emergency_AdjClose[Emergency_Asset_Col]

                elif Asset_Symbols_Emergency_AdjClose[Emergency_Asset_Col] in Bond_Signal_Assets_List:
        
                        #print(Asset_Symbols_Emergency_AdjClose[Emergency_Asset_Col])
                  
                        Current_Date_Emergency_Asset_Price = Asset_Prices_Array_Emergency.loc[Current_Date, ("Adj Close", Asset_Symbols_Emergency_AdjClose[Emergency_Asset_Col])] 

                        LookBack_Date_Emergency_Asset_Price = Asset_Prices_Array_Emergency.loc[Bond_Lookback_Date, ("Adj Close", Asset_Symbols_Emergency_AdjClose[Emergency_Asset_Col])]  

                        #print("Current  Bond = " + Current_Date_Emergency_Asset_Price)
                        #print("Lookbook  Bond = " + LookBack_Date_Emergency_Asset_Price)
                    
                        if np.isnan(Current_Date_Emergency_Asset_Price) == False and np.isnan(LookBack_Date_Emergency_Asset_Price) == False:
                           
                           if LookBack_Date_Emergency_Asset_Price != 0:
                                Emergency_Bond_Asset_Total_Return =  ((Current_Date_Emergency_Asset_Price/LookBack_Date_Emergency_Asset_Price) - 1)
                                
                                Reached_Bond_Check = True 

                                if Emergency_Bond_Asset_Total_Return != -10000 and Emergency_Bond_Asset_Total_Return < Min_Bond_Asset_Total_Return and Emergency_Bond_Asset_Total_Return > -1:
                                    Min_Bond_Asset_Total_Return = Emergency_Bond_Asset_Total_Return
                                    Min_Bond_Asset = Asset_Symbols_Emergency_AdjClose[Emergency_Asset_Col]
             

            if   Reached_Equity_Check == True or Reached_Bond_Check == True: 

                if len(Bond_Signal_Assets_List) == 0:
                    Min_Bond_Asset_Total_Return = Min_Equity_Asset_Total_Return
                if len(Equity_Signal_Assets_List) == 0:
                    Min_Equity_Asset_Total_Return = Min_Bond_Asset_Total_Return  

                if Min_Equity_Asset_Total_Return < Risk_Free_Returns - (Margin_RiskFree_Throshold*Risk_Free_Returns):
                    Total_Days_Pre_Allarm_ON_Equity = Total_Days_Pre_Allarm_ON_Equity + 1

                    print("Total_Days_Pre_Allarm_ON_Equity = "+ str(Total_Days_Pre_Allarm_ON_Equity))

                else:
                    Total_Days_Pre_Allarm_ON_Equity = 0

                if Min_Bond_Asset_Total_Return < Risk_Free_Returns - (Margin_RiskFree_Throshold*Risk_Free_Returns):
                    Total_Days_Pre_Allarm_ON_Bond = Total_Days_Pre_Allarm_ON_Bond + 1

                    print("Total_Days_Pre_Allarm_ON_Bond = "+ str(Total_Days_Pre_Allarm_ON_Bond))

                else:
                    Total_Days_Pre_Allarm_ON_Bond = 0

                if Total_Days_Pre_Allarm_ON_Equity >= Equity_Signal_Wait_Period:
                    Equity_Alarm_Signal = "Signal ON"
                else:
                    Equity_Alarm_Signal = "Signal OFF"
                           
                if Total_Days_Pre_Allarm_ON_Bond >= Bond_Signal_Wait_Period:
                    Bonds_Alarm_Signal = "Signal ON"
                else:
                    Bonds_Alarm_Signal = "Signal OFF"


                if (Emergency_Status == False and Equity_Alarm_Signal == "Signal ON" and Bonds_Alarm_Signal == "Signal ON") or (Emergency_Status == True and Equity_N_Bond_Hold_Period_Status == True and  Total_Days_Allarm_ON <= Equity_N_Bond_Hold_Period):
                    Emergency_Status = True
                    Total_Days_Allarm_ON = Total_Days_Allarm_ON + 1

                    print("Total_Days_Allarm_ON = "+ str(Total_Days_Allarm_ON))

                elif Emergency_Status == True and (Equity_Alarm_Signal == "Signal OFF" or Bonds_Alarm_Signal == "Signal OFF"):
                    Emergency_Status = False
                    Total_Days_Allarm_ON = 0

                if Emergency_Status == True and previous_Emergency_Status != Emergency_Status :
               
                    if Emergency_Alocation_Asset_Selected == False and len(Emergency_Assets) != 0:

                        for Emergency_Assets_List_item in Emergency_Assets:
                            if Emergency_Assets_List_item in Asset_Symbols_Emergency_AdjClose:

                                if np.isnan(Asset_Prices_Array_Emergency.loc[Current_Date, ("Adj Close", Emergency_Assets_List_item)]) == False:
                                    #print(Emergency_Assets_List_item)
                                    Emergency_Allocation_Asset_Col_Ref = Asset_Symbols_Emergency_AdjClose.index(Emergency_Assets_List_item)
                                    #Combined_Metric_Values = [Min_Equity_Asset_Total_Return, Min_Bond_Asset_Total_Return, Risk_Free_Returns]

                                    Adj_Close_Lookback_Period_Array = pd.Series(list(map(itemgetter(Emergency_Allocation_Asset_Col_Ref), Adj_Close_prices_Array))[:])

                                    Momentum_MultiFactor_Maximizer, MetricsData_Array = Functions.Momentum_MultiFactor_Maximizer(Lookback_Period, Date_Row, Adj_Close_Lookback_Period_Array)
                                    Combined_Metric_Values = Momentum_MultiFactor_Maximizer

                                    Asset_Volatility = Functions.find_volatility_for_asset(Asset_Prices_Array_Emergency, Emergency_Assets_List_item)
                                    Emeregency_Asset_Volume = Asset_Prices_Array_Emergency.loc[Current_Date, ("Volume", Emergency_Assets_List_item)]  
                                
                                    Emergency_Asset_Array = [Emergency_Assets_List_item, Combined_Metric_Values, Emeregency_Asset_Volume, Asset_Volatility, Emergency_Allocation_Asset_Col_Ref, Current_Date]

                                    Emergency_Assets_Array.append(Emergency_Asset_Array)
                                    
                                    Emergency_Alocation_Asset_Selected = True


                    if Emergency_Alocation_Asset_Selected == False and len(BackUp_Emergency_Assets) != 0:

                        for BackUp_Emergency_Assets_List_item in BackUp_Emergency_Assets:

                            if BackUp_Emergency_Assets_List_item in Asset_Symbols_Emergency_AdjClose:
                                if np.isnan(Asset_Prices_Array_Emergency.loc[Current_Date, ("Adj Close", BackUp_Emergency_Assets_List_item)]) == False:
                                    #print(BackUp_Emergency_Assets_List_item)
                                    Emergency_Allocation_Asset_Col_Ref = Asset_Symbols_Emergency_AdjClose.index(BackUp_Emergency_Assets_List_item)
                                    #Combined_Metric_Values = [Min_Equity_Asset_Total_Return, Min_Bond_Asset_Total_Return, Risk_Free_Returns]

                                    Adj_Close_Lookback_Period_Array = pd.Series(list(map(itemgetter(Emergency_Allocation_Asset_Col_Ref), Adj_Close_prices_Array))[:])

                                    Momentum_MultiFactor_Maximizer, MetricsData_Array = Functions.Momentum_MultiFactor_Maximizer(Lookback_Period, Date_Row, Adj_Close_Lookback_Period_Array)
                                    Combined_Metric_Values = Momentum_MultiFactor_Maximizer

                                    Asset_Volatility = Functions.find_volatility_for_asset(Asset_Prices_Array_Emergency, BackUp_Emergency_Assets_List_item)
                                    Emeregency_Asset_Volume = Asset_Prices_Array_Emergency.loc[Current_Date, ("Volume", BackUp_Emergency_Assets_List_item)]  
                                
                                    Emergency_Asset_Array = [BackUp_Emergency_Assets_List_item, Combined_Metric_Values, Emeregency_Asset_Volume, Asset_Volatility, Emergency_Allocation_Asset_Col_Ref, Current_Date]

                                    Emergency_Assets_Array.append(Emergency_Asset_Array)
                               
                                    Emergency_Alocation_Asset_Selected = True


                    if  Emergency_Alocation_Asset_Selected == True:
                         if len(Emergency_Assets_Array)>=Number_of_Emergency_Portfolio_Allocations:
                             Selected_Assets_Count =Number_of_Emergency_Portfolio_Allocations
                         else:
                             Selected_Assets_Count =len(Emergency_Assets_Array)


                         Markets_Filter_1_FilterType = "Sub_Classes"
                         Emergency_Assets_Array = Functions.Special_Asset_Filter(Emergency_Assets_Array, Markets_Filter_1_FilterType)
                         Emergency_Assets_Selected_Array =  Functions.Get_Top_reformers_ByMetric_Value(Emergency_Assets_Array, Selected_Assets_Count)
                         Emergency_Assets_Weights_Array, Total_Emergnecy_Weight =  Assets_Weightage.Assets_Weightage_Handler(Date_Row, Previous_Price_Row, Current_Rebalancing_Date, Type_of_Momentum_Method, Selected_method_of_Momuntum_Metric, Method_of_Assets_Selection, Lookback_Period, int(Number_of_Emergency_Portfolio_Allocations), Method_Of_Wieghting, Volatility_Weight_Strength, Volume_Weight_Strength, Volatility_Measuring_Lookback_Period, Emergency_Assets_Selected_Array, Asset_Prices_Array, Open_prices_Array, High_prices_Array, Low_prices_Array, Close_prices_Array, Adj_Close_prices_Array, Volume_Array, Asset_Symbols, Date_Array)


            #print(Emergency_Assets_Selected_Array)
            #print(Emergency_Assets_Weights_Array)


            print(f'Checking Date for Emergency: {Current_Date}---- Emergency Status : {Emergency_Status} ---- Equity Signal Returns : {Min_Equity_Asset_Total_Return} ---- Bond Signal Returns : {Min_Bond_Asset_Total_Return}  ---- Risk Free Signal Reference : {Risk_Free_Returns}', sep='\n')

            if Backtest_Mode == True:
                Emergency_Metric_Data_Array = [Current_Date, Emergency_Status, Min_Equity_Asset, Min_Equity_Asset_Total_Return, Min_Bond_Asset, Min_Bond_Asset_Total_Return, Risk_Free_Returns]

            return Rebalancing_Date_Switch, Current_Rebalancing_Date, Emergency_Status, Emergency_Assets_Selected_Array, Total_Days_Pre_Allarm_ON_Equity, Total_Days_Pre_Allarm_ON_Bond, Total_Days_Allarm_ON, Emergency_Assets_Weights_Array, Total_Emergnecy_Weight, Emergency_Metric_Data_Array
    
    elif Emergency_Status_Input_Code == 2:

        Emergency_Assets_Selected_Array = []
        Emergency_Assets_Weights_Array = []
        Emergency_Metric_Data_Array = []
        Total_Emergnecy_Weight = 0
        Emergency_Status = False
            
        return Rebalancing_Date_Switch, Current_Rebalancing_Date, Emergency_Status, Emergency_Assets_Selected_Array, Total_Days_Pre_Allarm_ON_Equity, Total_Days_Pre_Allarm_ON_Bond, Total_Days_Allarm_ON, Emergency_Assets_Weights_Array, Total_Emergnecy_Weight, Emergency_Metric_Data_Array