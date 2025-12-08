
import pandas as pd
import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from dateutil.relativedelta import relativedelta

from dateutil.parser import parse
from datetime import datetime, date
from operator import itemgetter

import Functions as Functions
import json

#Data_Record_Start_Date = ""
separator = ", "
Asset_Input = []
US_BUSINESS_DAY = CustomBusinessDay(calendar=USFederalHolidayCalendar())
Month_Days = 22

# Input:
#-------

def input_Entry_Handler(Incoming_Input):


    print("processing Data from AlgoMArt API ... ")
    print("_______________________________________ ")

    indicator_data_list= []
    asset_data_list = []

    if Incoming_Input['Data_Source'] == "AlgoMart":

        for asset in Incoming_Input["assets"]:
            symbol = asset["symbol"]

            body = json.loads(asset['data']['body'])
            for daily_bar in body:
                daily_bar["symbol"] = symbol
                asset_data_list.append(daily_bar)

        #print(asset_data_list)

        Asset_Prices_Array_Model = pd.DataFrame(asset_data_list)
        Asset_Prices_Array_Model.rename(columns={'adjustedClose': 'Adj Close', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
        Asset_Prices_Array_Model['date'] = pd.to_datetime(Asset_Prices_Array_Model['date'])
        pivot_Asset_Prices_Array_Model = Asset_Prices_Array_Model.pivot_table(index='date', columns='symbol', values=['Adj Close', 'Open', 'High', 'Low', 'Close', 'Volume'], aggfunc='first')
        pivot_Asset_Prices_Array_Model = pivot_Asset_Prices_Array_Model.sort_index(axis=1, level=0)

    elif Incoming_Input['Data_Source'] == "AlphaVanatage" or Incoming_Input['Data_Source'] == "Yahoo_Finanace":
        pivot_Asset_Prices_Array_Model = Incoming_Input["assets"]
        pivot_Asset_Prices_Array_Model.index = pd.to_datetime(pivot_Asset_Prices_Array_Model.index)

    #print(pivot_Asset_Prices_Array_Model)

    if Incoming_Input['Data_Source'] == "AlphaVanatage" or Incoming_Input['Data_Source'] == "AlgoMart" or Incoming_Input['Data_Source'] == "Yahoo_Finanace":

        for indicator in Incoming_Input["indicators"]:
            symbol = indicator["symbol"]
            if symbol == "US_TREASURY_YIELD_3M":
                body = json.loads(indicator['data']['body'])

                for daily_bar in body:
                    daily_bar["date"]= daily_bar["date"]
                    daily_bar["symbol"]= "^IRX"
                    daily_bar["adjustedClose"]= daily_bar.pop("value") if "value" in daily_bar else np.nan
                    daily_bar["open"]=  np.nan
                    daily_bar["high"]=  np.nan
                    daily_bar["low"]=  np.nan
                    daily_bar["close"]=  np.nan
                    daily_bar["volume"]=  np.nan

                    indicator_data_list.append(daily_bar)

        Indicator_Array_Model = pd.DataFrame(indicator_data_list)
        Indicator_Array_Model.rename(columns={'adjustedClose': 'Adj Close', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
        Indicator_Array_Model['date'] = pd.to_datetime(Indicator_Array_Model['date'])
        Indicator_Array_Model.set_index(['date', 'symbol'], inplace=True)
        Indicator_Array_Model = Indicator_Array_Model.unstack(level=-1)
        Indicator_Array_Model = Indicator_Array_Model.sort_index(axis=1)

        Asset_Prices_Array_Model = pd.concat([pivot_Asset_Prices_Array_Model, Indicator_Array_Model], axis=1)
        Asset_Prices_Array_Model = Asset_Prices_Array_Model.sort_index(axis=1)
        
    elif Incoming_Input['Data_Source'] == "StoredData":

        print("Retrieving AlgoMArt API Data from Local CSV File : Model_Assets_Prices_Database.CSV ... ")
        print("_________________________________________________________________________________________ ")

        Asset_Prices_Array_Model = Functions.read_csv_file_2headers("tmp/Model_Assets_Prices_Database.csv")


    Emergency_Assets_List = Incoming_Input["Emergency_Assets_All"]
    Models_Assets_List = Incoming_Input["Model_Assets"]
    Equity_Signal_Assets_List = Incoming_Input["Equity_Signal_Assets_List"]
    Bond_Signal_Assets_List = Incoming_Input["Bond_Signal_Assets_List"]
    BackUp_Emergency_Assets = Incoming_Input["BackUp_Emergency_Assets"]
    Emergency_Assets = Incoming_Input["Emergency_Assets"]
    Reference_RiskFree_Asset = Incoming_Input["Reference_RiskFree_Asset"]        

    all_assets = (
        Emergency_Assets_List +
        Models_Assets_List +
        Equity_Signal_Assets_List +
        Bond_Signal_Assets_List +
        BackUp_Emergency_Assets +
        Emergency_Assets +
        Reference_RiskFree_Asset
    )

    All_Asets_Combined = list(set(all_assets))
    All_Asets_Combined.sort()

    for Asset_Col in range(len(All_Asets_Combined)):
        Asset_Not_Found = True 
        for price_type, symbol in Asset_Prices_Array_Model.columns:
            if All_Asets_Combined[Asset_Col]==symbol :
                Asset_Not_Found = False

        if Asset_Not_Found == True:
            print(f"The asset '{All_Asets_Combined[Asset_Col]}' has not been downloaded from API.")

    for price_type, symbol in Asset_Prices_Array_Model.columns:
        if Asset_Prices_Array_Model[(price_type, symbol)].isna().all():
            print(f"The asset '{symbol}' with price type '{price_type}' has all NaN values.")

       
    if len(Asset_Prices_Array_Model)>0:

        Date_Array = Asset_Prices_Array_Model.index.tolist()

        Emergency_Status_Input_Code = 1

        Data_Record_Start_Date = Incoming_Input["sinceDate"]
        Data_Record_End_Date = Incoming_Input["untilDate"]

        try:
            Data_Record_Start_Date = datetime.strptime(str(Data_Record_Start_Date), '%Y-%m-%d')
            Data_Record_End_Date = datetime.strptime(str(Data_Record_End_Date), '%Y-%m-%d')

            Start_Row = None
            End_Row = None

            if Data_Record_Start_Date != Data_Record_End_Date:

                for Date_Row in range(len(Date_Array)):
                    Checked_Date = Date_Array[Date_Row].date()  
        
                    if Data_Record_Start_Date == Checked_Date:
                        Start_Row = Date_Row

                    if Data_Record_End_Date == Checked_Date:
                        End_Row = Date_Row
                        if Start_Row == End_Row:
                            End_Row = End_Row + 1


                if Start_Row is None:
                    closest_start_date = min(
                        (date for date in Date_Array if date >= Data_Record_Start_Date),
                        key=lambda x: abs(x - Data_Record_Start_Date)
                    )
                    Start_Row = Date_Array.index(closest_start_date)

                if End_Row is None:
                    closest_end_date = min(
                        (date for date in Date_Array if date <= Data_Record_End_Date),
                        key=lambda x: abs(x - Data_Record_End_Date)
                    )
                    End_Row = Date_Array.index(closest_end_date)
                    
                if Start_Row == End_Row:
                    End_Row = min(End_Row + 1, len(Date_Array) - 1)

            elif Data_Record_Start_Date == Data_Record_End_Date:

                closest_end_date = min(
                    (date for date in Date_Array if date <= Data_Record_End_Date),
                    key=lambda x: abs(x - Data_Record_End_Date)
                )

                Start_Row = Date_Array.index(closest_end_date)
                End_Row = Date_Array.index(closest_end_date)

                if Start_Row == End_Row:
                    End_Row = min(End_Row + 1, len(Date_Array) - 1)

            # Handle cases where no valid rows are found
            if Start_Row is None or End_Row is None:
                Start_Row = -1
                End_Row = -1
                Start_Simulation = False
            else:
                Start_Simulation = True

        except ValueError:
            Start_Row = -1
            End_Row = -1
            Start_Simulation = False

        if Incoming_Input['Data_Source'] == "AlphaVanatage"  or Incoming_Input['Data_Source'] == "Yahoo_Finanace" or Incoming_Input['Data_Source'] == "AlgoMart":

            print("Cleaning Data from AlgoMArt API ... ")
            print("_______________________________________ ")

            print(f"The Asset_Prices_Array_Model size before delete '{len(Asset_Prices_Array_Model)}'.")
            Asset_Prices_Array_Model, Start_Row, End_Row = Functions.clean_data(Asset_Prices_Array_Model, 80, 15, start_row=Start_Row, end_row=End_Row)

            print(f"The Asset_Prices_Array_Model size After delete '{len(Asset_Prices_Array_Model)}'.")

        if Incoming_Input['Data_Source'] == "AlphaVanatage"  or  Incoming_Input['Data_Source'] == "Yahoo_Finanace":

            print("Saving Data from AlgoMArt API to Local CSV File : Model_Assets_Prices_Database.CSV ... ")
            print("_________________________________________________________________________________________ ")

            Asset_Prices_Array_Model.to_csv('tmp/Model_Assets_Prices_Database.csv', index=True)


        print("Splitting Prices Data from AlgoMArt API ... ")
        print("_______________________________________ ")

        Asset_Prices_Array_Emergency = Asset_Prices_Array_Model.loc[:, Asset_Prices_Array_Model.columns.get_level_values(1).isin(Emergency_Assets_List)]
        Asset_Prices_Array_Model = Asset_Prices_Array_Model.loc[:, Asset_Prices_Array_Model.columns.get_level_values(1).isin(Models_Assets_List)]

        if Asset_Prices_Array_Emergency.empty:
            Asset_Prices_Array_Emergency = pd.DataFrame()
        else:
            Asset_Prices_Array_Emergency = Asset_Prices_Array_Emergency  

        if not Asset_Prices_Array_Model.empty:
                
            Open_prices_Array = Asset_Prices_Array_Model.get("Open").values.tolist()
            High_prices_Array = Asset_Prices_Array_Model.get("High").values.tolist()
            Low_prices_Array = Asset_Prices_Array_Model.get("Low").values.tolist()
            Close_prices_Array = Asset_Prices_Array_Model.get("Close").values.tolist()
            Adj_Close_prices_Array = Asset_Prices_Array_Model.get("Adj Close").values.tolist()
            Volume_Array = Asset_Prices_Array_Model.get("Volume").values.tolist()
            Asset_Symbols=Asset_Prices_Array_Model.get("Adj Close").columns.tolist()
            Data_Array= list(zip(Date_Array,Adj_Close_prices_Array))
            Date_Array = Asset_Prices_Array_Model.index.tolist()

        #-----------------------------------------------------------------------------------------------------------------------------

        # Output the result
        if Start_Row == None or End_Row == None:
            Start_Row = -1
            End_Row = -1
            Start_Simulation = False

        if Start_Row != -1 or End_Row != -1:
            print(f"Start_Date: {Date_Array[Start_Row]} , End_Row: {Date_Array[End_Row]}, Start_Simulation: {Start_Simulation}")

        input_parameters_List = [
            Start_Simulation,
            Data_Record_Start_Date,
            Data_Record_End_Date,
            Start_Row,
            End_Row,
            Asset_Prices_Array_Model,
            Open_prices_Array,
            High_prices_Array,
            Low_prices_Array,
            Close_prices_Array,
            Adj_Close_prices_Array,
            Volume_Array,
            Asset_Symbols,
            Date_Array,
            Data_Array,
            Emergency_Status_Input_Code,
            Asset_Prices_Array_Emergency,
            Equity_Signal_Assets_List,
            Bond_Signal_Assets_List,
            BackUp_Emergency_Assets,
            Emergency_Assets,
            Reference_RiskFree_Asset,
        ]

        print("Step 1 - Completed.")
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")  
                
        return input_parameters_List

    if len(Asset_Prices_Array_Model) == 0:
 
        input_parameters_List = []
                
        return input_parameters_List










    
        #import yfinance
        #import yahoo_fin.stock_info as si
        #import Active_Trades_Handler as Active_Trades_Handler
        #from alpha_vantage.timeseries import TimeSeries

        #Save_Total_ETFs_List_file = 'Total ETFs List.txt'
        #Save_Total_ETFs_performance_List_file = 'Total ETFs Performance List.txt' 

        #input_parameters_List = ()

        #Asset_performanc_Data_List = []
        #Total_ETFs_List = []

        #Update_Assets_input = input("""Input 1 : Do you want to update whole list of ETF's information and stats before ranking (from Online Soruce) ?: \n 
        #                                1) Yes   --- Enter '1' for this option. \n
        #                                2) No --- Enter '2' for this option. \n
        #                                ** Selection: """ )
        #Data_Collected = False

        #if Update_Assets_input == '1':

        #    # A) ETF's list :
        #    #----------------

        #    print(f'System is collecting ETFs list ... Please Wait .. ', sep='\n')

        #    Count_ETF_list = 0
        #    List_of_Tickers_1 = pd.DataFrame(si.tickers_nasdaq(True)).values.tolist()
        #    for ticker in range(len(List_of_Tickers_1)):
        #        if List_of_Tickers_1[ticker][6] ==  'Y':
        #            Count_ETF_list = Count_ETF_list + 1
        #            Total_ETFs_List.append([List_of_Tickers_1[ticker][0],List_of_Tickers_1[ticker][1]])
        #            #print(f'Asset Symbol = {List_of_Tickers_1[ticker][0]} ---- Asset Name= {List_of_Tickers_1[ticker][1]}', sep='\n')

        #    List_of_Tickers_2 = pd.DataFrame(si.tickers_other(True)).values.tolist()
        #    for ticker in range(len(List_of_Tickers_2)):
        #        ETF_Check_in_List = False
        #        ETF_Check_in_List = any(List_of_Tickers_2[ticker][0] in sublist for sublist in Total_ETFs_List)  
        #        if List_of_Tickers_2[ticker][4] ==  'Y' and ETF_Check_in_List == False:
        #            Count_ETF_list = Count_ETF_list + 1
        #            Total_ETFs_List.append([List_of_Tickers_2[ticker][0],List_of_Tickers_2[ticker][1]])
        #            #print(f'Asset Symbol = {List_of_Tickers_2[ticker][0]} ---- Asset Name= {List_of_Tickers_2[ticker][1]}', sep='\n')
        
        #    # save ETF's list in file:
        #    with open(Save_Total_ETFs_List_file, 'w') as Total_ETFs_List_Open:
        #        Total_ETFs_List_Open.truncate(0)
        #        Total_ETFs_List_Open.write(str(Total_ETFs_List))

        #    # Load ETF's list in file:
        #    if os.path.exists(Save_Total_ETFs_List_file):
        #        with open(Save_Total_ETFs_List_file) as Total_ETFs_List_Open:
        #                Total_ETFs_List = eval(Total_ETFs_List_Open.read(),{'nan':'nan'})        

        #    # B) ETF's Performanc list :
        #    #--------------------------

        #    print(f'System is collecting ETFs performance Data ... Please Wait .. ', sep='\n')

        #    for Symbol_Row in range(len(Total_ETFs_List)):

        #        try:
        #            Asset_performanc_Data_Dict = list(si.get_analysts_info(Total_ETFs_List[Symbol_Row][0]).values())
        #            Asset_performanc_Data_List.append([Total_ETFs_List[Symbol_Row][0], Total_ETFs_List[Symbol_Row][1], Functions.Text_to_num((Asset_performanc_Data_Dict[0][1][0]), math.nan) , float(Asset_performanc_Data_Dict[0][1][1]), float(Asset_performanc_Data_Dict[0][1][2]), float(str(Asset_performanc_Data_Dict[0][1][3]).strip('%')) / 100.0, float(str(Asset_performanc_Data_Dict[0][1][4]).strip('%')) / 100.0, float(Asset_performanc_Data_Dict[0][1][5]), float(str(Asset_performanc_Data_Dict[0][1][6]).strip('%')) / 100.0 , (Asset_performanc_Data_Dict[0][1][7] if (isinstance(pd.to_datetime(Asset_performanc_Data_Dict[0][1][7]), date) == True) else math.nan)])
        #            print(f'percent complete = {round((Symbol_Row/len(Total_ETFs_List))*100,1)}%', end='\r')
        #            sys.stdout.flush()
        #        except Exception:
        #            pass

        #        # save ETF's Performanc list in file:
        #    with open(Save_Total_ETFs_performance_List_file, 'w') as Total_ETFs_performance_List_Open: 
        #        Total_ETFs_performance_List_Open.truncate(0)
        #        Total_ETFs_performance_List_Open.write(str(Asset_performanc_Data_List))

        #    # load ETF's Performanc list in file:
        #    if os.path.exists(Save_Total_ETFs_performance_List_file):
        #        with open(Save_Total_ETFs_performance_List_file) as Total_ETFs_performance_List_Open:
        #                Asset_performanc_Data_List = eval(Total_ETFs_performance_List_Open.read(),{'nan':'nan'}) 
            
        #    if len(Asset_performanc_Data_List)> 0 and len(Total_ETFs_List)> 0:
        #        print(" \n ETFs List and performance Data are updated and loaded successfully.")
        #        Data_Collected = True
        #    else:
        #        print(" \n Models assets data and prices not loaded successfully. Please review your model's assets and backtesting date/periods when running the software again. this run is terminated.")
        #        input_parameters_List = []
        #        return input_parameters_List

        #elif Update_Assets_input == '2':

        #    if os.path.exists(Save_Total_ETFs_List_file) and os.path.exists(Save_Total_ETFs_performance_List_file) :

        #        # A) lETF's list in file:
        #        if os.path.exists(Save_Total_ETFs_List_file):
        #            with open(Save_Total_ETFs_List_file) as Total_ETFs_List_Open:
        #                Total_ETFs_List = eval(Total_ETFs_List_Open.read(),{'nan':'nan'}) 

        #        # B) load ETF's Performanc list in file:
        #        if os.path.exists(Save_Total_ETFs_performance_List_file):
        #            with open(Save_Total_ETFs_performance_List_file) as Total_ETFs_performance_List_Open:
        #                Asset_performanc_Data_List = eval(Total_ETFs_performance_List_Open.read(),{'nan':'nan'})
                
        #    if len(Asset_performanc_Data_List)> 0 and len(Total_ETFs_List)> 0:
        #        print(" \n ETFs List and performance Data are loaded successfully.")
        #        Data_Collected = True
        #    else:
        #        print(" \n Models assets data and prices not loaded successfully. Please review your model's assets and backtesting date/periods when running the software again. this run is terminated.")
        #        input_parameters_List = []
        #        return input_parameters_List
        #else:
        #    print(" \n Please select the correct option when running the software again. this run is terminated.")
        #    input_parameters_List = []
        #    return input_parameters_List

        #if Data_Collected == True:

        #    Ranking_Criteria = int(input("""Input 2: Please select the ranking method for the assets list : \n 
        #                                    1) Net Assets.   --- Enter '1' for this option. \n
        #                                    2) NAV (Net Assets Value). --- Enter '2' for this option. \n
        #                                    3) PE Ratio (TTM).  --- Enter '3' for this option. \n
        #                                    4) Yield. --- Enter '4' for this option. \n
        #                                    5) YTD Daily Total Return.  --- Enter '5' for this option. \n
        #                                    6) Beta (5Y Monthly).  --- Enter '6' for this option. \n
        #                                    7) Expense Ratio (net). --- Enter '7' for this option. \n
        #                                    8) Inception Date --- Enter '8' for this option. \n
        #                                    ** Selection: """ ))

        #                                    #2              Net Assets      36.98M
        #                                    #3                     NAV       48.43
        #                                    #4          PE Ratio (TTM)         NaN
        #                                    #5                   Yield       3.06%
        #                                    #6  YTD Daily Total Return     -22.76%
        #                                    #7       Beta (5Y Monthly)        1.14
        #                                    #8     Expense Ratio (net)       1.00%
        #                                    #9          Inception Date  2010-07-20}

        #    if Ranking_Criteria != "":
        #        Rank_Parameter = (Ranking_Criteria) +1

        #        if Rank_Parameter == 2:
        #            Is_Descending = True
        #        elif Rank_Parameter == 3:
        #            Is_Descending = True
        #        elif Rank_Parameter == 4:
        #            Is_Descending = True
        #        elif Rank_Parameter == 5:
        #            Is_Descending = True
        #        elif Rank_Parameter == 6:
        #            Is_Descending = True
        #        elif Rank_Parameter == 7:
        #            Is_Descending = True
        #        elif Rank_Parameter == 8:
        #            Is_Descending = False
        #        elif Rank_Parameter == 9:
        #            Is_Descending = False

        #        Number_of_Top_ETFs_Selected = int(input("""Input 3: Select model's number of assets (selected at the top of the list after ranking) : \n 
        #                                            ** Selection: """ ))
        #        Asset_Input = ""
        #        Count_Assets_Added = 0

        #        if len(Asset_performanc_Data_List) != 0 and Number_of_Top_ETFs_Selected > 0:
        #            Asset_performanc_Data_List.sort(key=lambda a: (a[Rank_Parameter] == 'nan', a[Rank_Parameter]), reverse=Is_Descending)

        #            for Symbol_Row in range(len(Asset_performanc_Data_List)):
        #                if  Count_Assets_Added < int(Number_of_Top_ETFs_Selected):
        #                    if Asset_performanc_Data_List[Symbol_Row][Rank_Parameter] != 'nan': # and Asset_performanc_Data_List[Symbol_Row][Rank_Parameter] ==1and Asset_performanc_Data_List[Symbol_Row][Rank_Parameter] <= 1.3:
        #                        if Asset_Input == "":
        #                            Asset_Input =  Asset_performanc_Data_List[Symbol_Row][0]
        #                            Count_Assets_Added = Count_Assets_Added +1
        #                        else:
        #                            Asset_Input = Asset_Input + " " + Asset_performanc_Data_List[Symbol_Row][0]
        #                            Count_Assets_Added = Count_Assets_Added +1
        #        else:
        #            print(" \n Models assets data and prices not loaded successfully. Please review your model's assets and backtesting date/periods when running the software again. this run is terminated.")
        #            input_parameters_List = []
        #            return input_parameters_List

        #    else:
        #        print(" \n Please select the correct option when running the software again. this run is terminated.")
        #        input_parameters_List = []
        #        return input_parameters_List
        #else:
        #    print(" \n Please select the correct option when running the software again. this run is terminated.")
        #    input_parameters_List = []
        #    return input_parameters_List




