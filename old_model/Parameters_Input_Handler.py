
import pandas as pd
import numpy as np

from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

from dateutil.parser import parse
from datetime import datetime, date
from operator import itemgetter

import Functions as Functions

separator = ", "

US_BUSINESS_DAY = CustomBusinessDay(calendar=USFederalHolidayCalendar())
Month_Days = 22
# Parameters:
#-------------

def Parameters_Input_Handler(Emergency_Status_Input_Code, Incoming_Input):

    Initial_Invesment_Value = 100
    Days_Per_Month = 22

    if Incoming_Input['Manual_Parameters_Entry'] == True:

        Method_of_Assets_Selection_Code = input("""Input A: Select the Method of Assets Selection: \n 
                                    1) Price Momuntum Strategy: --- Enter '1' for this option. \n
                                    2) Momuntum Method with AI  : --- Enter '2' for this option. \n
                                    ** Selection: """ )

        Rebalancing_Method = input("""Input 1: Strategy Rebalancing Method: \n 
                            1) Weekly Rebalance (periodic): --- Enter '1' for this option. \n
                            2) Monthly Rebalance (periodic): --- Enter '2' for this option. \n
                            3) Active Rebalance : --- Enter '3' for this option. \n
                            ** Selection: """ )

        if Method_of_Assets_Selection_Code == '1' and int(Rebalancing_Method)>=1 and int(Rebalancing_Method)<=3:

            AI_Model = False
            Method_of_Assets_Selection = "Price Momuntum Strategy"

        elif int(Method_of_Assets_Selection_Code) == 2 and int(Rebalancing_Method)>=1 and int(Rebalancing_Method)<=3:

            AI_Model = True
            Method_of_Assets_Selection = "Momuntum Method with AI"

        Type_of_Momentum_Method = "Cross_Sectional Momuntum"

        Number_of_Portfolio_Allocations = input("""Input A: Select the number of portfolio allocations: \n 
                                        ** Selection: """ )
        if AI_Model == False:

            Selected_method_of_Momuntum_Metric_Code = int(input("""Input A: Select the method of momuntum metric: \n 
                                                        1) Metrics Count: --- Enter '1' for this option. \n
                                                        2)  Metrics Combind: --- Enter '2' for this option. \n
                                                        ** Selection: """ ))
                   
            if (Selected_method_of_Momuntum_Metric_Code != 1 and Selected_method_of_Momuntum_Metric_Code != 2):
                Selected_method_of_Momuntum_Metric = "Metrics Combind"
            else:
                if Selected_method_of_Momuntum_Metric_Code == 1:
                    Selected_method_of_Momuntum_Metric = "Metrics Count"
                elif Selected_method_of_Momuntum_Metric_Code ==2:
                    Selected_method_of_Momuntum_Metric = "Metrics Combind"

        elif AI_Model == True:
            Selected_method_of_Momuntum_Metric = "Metrics as Selected Features"



        Wieghting_ON_Code = int(input("""Input A: Select the assets wieghting method: \n 
                                                    1) Assets Inverse Volatility Wieghting: --- Enter '1' for this option. \n
                                                    2) Assets Volume Wieghting: --- Enter '2' for this option. \n
                                                    3) Assets Inverse Volatility & Volume mix Wieghting: --- Enter '3' for this option. \n
                                                    4) Efficient Frontier Optimized Weighting: --- Enter '4' for this option. \n
                                                    ** Selection: """ ))

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

        Lookback_Period_Unit = int(input("""Input A: Select the measuring lookback Period unit: \n 
                                1) Days: --- Enter '1' for this option. \n
                                2) Weeks: --- Enter '2' for this option. \n
                                3) Months: --- Enter '3' for this option. \n
                                    ** Selection: """ ))                  
                   
        Volatility_Measuring_Lookback_Period = int(input("""Input A: Select the asssets volatility measuring lookback period (For calculations) to selected multipule of selected measuring unit: \n 
                                                ** Selection: """ ))

        if  (Lookback_Period_Unit == 1 or Lookback_Period_Unit == 2 or Lookback_Period_Unit == 3):
            if Lookback_Period_Unit ==  1:
                Volatility_Measuring_Lookback_Period =  Volatility_Measuring_Lookback_Period

            elif Lookback_Period_Unit == 2:
                Volatility_Measuring_Lookback_Period =  Volatility_Measuring_Lookback_Period * 5

            elif Lookback_Period_Unit == 3:
                Volatility_Measuring_Lookback_Period =  Volatility_Measuring_Lookback_Period * Days_Per_Month
        else:
            Volatility_Measuring_Lookback_Period =  12 * Days_Per_Month


        Lookback_Period_Code = int(input("""Input A: Select the measuring lookback period to selected multipule of selected measuring unit: \n 
                                    ** Selection: """ ))

        if (Lookback_Period_Unit == 1 or Lookback_Period_Unit == 2 or Lookback_Period_Unit == 3):
            if Lookback_Period_Unit ==  1:
                Lookback_Period =  int(Lookback_Period_Code)

            elif Lookback_Period_Unit == 2:
                Lookback_Period =  int(Lookback_Period_Code) * 5

            elif Lookback_Period_Unit == 3:
                Lookback_Period =  int(Lookback_Period_Code) * Days_Per_Month

        else:
            Lookback_Period =  3 * Days_Per_Month
                

        # Emergency System parameters:

        Emergency_Status_Input_Code = int(input("""Input A: Select Emergency System tracking Status : \n 
                                1) ON: --- Enter '1' for this option. \n
                                2) OFF: --- Enter '2' for this option. \n
                                    ** Selection: """ ))    
        Equity_Lookback_Period = 55
        Bond_Lookback_Period = 110

        Equity_Signal_Wait_Period = 3
        Bond_Signal_Wait_Period = 1
        Equity_N_Bond_Hold_Period_Status = True
        Equity_N_Bond_Hold_Period = 3


        #Assets Filter:
        Special_Filter_Status_Code = int(input("""Input A: Select Special Assets Filter Status : \n 
                                1) ON: --- Enter '1' for this option. \n
                                2) OFF: --- Enter '2' for this option. \n
                                    ** Selection: """ ))    

        if Special_Filter_Status_Code == 1:
            
            Special_Filter_Status = True 

            Special_Filter_Type_Code = int(input("""Input A: Select Special Assets Filter Type : \n 
                                    1) Sub_Classes: --- Enter '1' for this option. \n
                                    2) Markets: --- Enter '2' for this option. \n
                                    3) Sectors: --- Enter '3' for this option. \n
                                        ** Selection: """ ))    

            if Special_Filter_Type_Code == 1:
                Special_Filter_Type = "Sub_Classes"
            elif Special_Filter_Type_Code == 2:
                Special_Filter_Type = "Markets"
            elif Special_Filter_Type_Code == 3:
                Special_Filter_Type = "Sectors"
            else:
                Special_Filter_Type = "Sub_Classes"

        elif Special_Filter_Status_Code == 2:
            Special_Filter_Status = False 
            Special_Filter_Type = "Sub_Classes"
        else:
            Special_Filter_Status = False 
            Special_Filter_Type = "Sub_Classes"


        Output_Parameters_List = [Method_of_Assets_Selection, Type_of_Momentum_Method, Selected_method_of_Momuntum_Metric, Number_of_Portfolio_Allocations, Initial_Invesment_Value, Method_Of_Wieghting, Volatility_Weight_Strength, Volume_Weight_Strength, Volatility_Measuring_Lookback_Period, Equity_Lookback_Period, Bond_Lookback_Period, Lookback_Period, Rebalancing_Method, Emergency_Status_Input_Code, Special_Filter_Status, Equity_Signal_Wait_Period, Bond_Signal_Wait_Period, Equity_N_Bond_Hold_Period_Status, Equity_N_Bond_Hold_Period, Special_Filter_Type]
        return Output_Parameters_List
       

    elif Incoming_Input['Manual_Parameters_Entry'] == False:


        print("Step 2: Retireve Simulation parameters and setup:")

        # General parameters:

        #Rebalancing_Method = '1'
        Rebalancing_Method = '2'
        Method_of_Assets_Selection = "Price Momuntum Strategy"
        Type_of_Momentum_Method = "Cross_Sectional Momuntum"
        Selected_method_of_Momuntum_Metric = "Metrics Combind" #"Metrics Count"
        Number_of_Portfolio_Allocations = 3
           
        # Wieghting method parameters:

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
        Volatility_Measuring_Lookback_Period = 12
       
        Lookback_Period_Unit = 3
        
        Lookback_Period_Code = 3
        Lookback_Period = Days_Per_Month * Lookback_Period_Code

        # Emergency System parameters:

        Emergency_Status_Input_Code = 1

        Equity_Lookback_Period = 55
        Bond_Lookback_Period = 110

        #---------------------------------------------------------------------------------------
        ###Moderate 01 : (Bond)   &&&& Conservative 01. &&&&   Moderately Conservative 01

        #Equity_Signal_Wait_Period = 1 # 3
        #Bond_Signal_Wait_Period = 1 #  3
        #Equity_N_Bond_Hold_Period_Status = True
        #Equity_N_Bond_Hold_Period = 3

        #---------------------------------------------------------------------------------------
        ##Aggressive 01 :  (Equity)
        Equity_Signal_Wait_Period = 3
        Bond_Signal_Wait_Period = 1
        Equity_N_Bond_Hold_Period_Status = True
        Equity_N_Bond_Hold_Period = 3#4

        #Assets Filter:
        Special_Filter_Status = True 
        Special_Filter_Type = "Sub_Classes"


        Output_Parameters_List = [Method_of_Assets_Selection, Type_of_Momentum_Method, Selected_method_of_Momuntum_Metric, Number_of_Portfolio_Allocations, Initial_Invesment_Value, Method_Of_Wieghting, Volatility_Weight_Strength, Volume_Weight_Strength, Volatility_Measuring_Lookback_Period, Equity_Lookback_Period, Bond_Lookback_Period, Lookback_Period, Rebalancing_Method, Emergency_Status_Input_Code, Special_Filter_Status, Equity_Signal_Wait_Period, Bond_Signal_Wait_Period, Equity_N_Bond_Hold_Period_Status, Equity_N_Bond_Hold_Period, Special_Filter_Type]

        print("Step 2 - Completed")
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")  
            
        return Output_Parameters_List
