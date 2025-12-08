
import pandas as pd
import numpy as np
import holidays

from datetime import datetime


import os
from scipy.stats import percentileofscore
import json
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

from scipy.optimize import minimize
import datetime
import math

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics
import requests
from scipy.stats import rankdata
import yfinance
import yahoo_fin.stock_info as si
import Active_Trades_Handler as Active_Trades_Handler
from alpha_vantage.timeseries import TimeSeries

Month_Days = 22

from decimal import Decimal, getcontext, Context


getcontext().prec = 28

def PercentRank_Inc(data, value):

    data = np.array(data)
    rank = percentileofscore(data, value, kind='rank')
    return rank


# Set up the decimal context
getcontext().prec = 28  # Increase precision to handle more complex numbers
getcontext().rounding = "ROUND_HALF_UP"  # Excel-like rounding

def manual_rank(data, value):
    try:
        # Filter out NaN values from data
        filtered_data = [d for d in data if not np.isnan(d)]
        
        # Convert the data to Decimal, ignoring NaNs
        extended_data = [Decimal(str(d)) for d in filtered_data] + [Decimal(str(value))]
        
        # Sort data while preserving original indexes to handle ties
        indexed_data = sorted((val, idx) for idx, val in enumerate(extended_data))
        ranks = [0] * len(extended_data)
        
        # Handle ranking with ties by averaging ranks
        sum_ranks = 0
        same_count = 1
        prev_val = None

        for i, (val, idx) in enumerate(indexed_data):
            if val == prev_val:
                same_count += 1
                sum_ranks += i + 1
            else:
                if same_count > 1:
                    avg_rank = sum_ranks / same_count
                    for j in range(i - same_count, i):
                        ranks[indexed_data[j][1]] = avg_rank
                    sum_ranks = 0
                sum_ranks = i + 1
                same_count = 1
            prev_val = val

        # Handle the last set of ties
        if same_count > 1:
            avg_rank = sum_ranks / same_count
            for j in range(len(extended_data) - same_count, len(extended_data)):
                ranks[indexed_data[j][1]] = avg_rank
        else:
            ranks[indexed_data[-1][1]] = len(extended_data)

        return ranks[-1], len(extended_data)  # Return the rank of the value and total count

    except Exception as e:
        print(f"An error occurred: {e}")
        # Optionally, print out data to debug what might be wrong
        print([str(d) for d in extended_data])
        raise

def excel_percent_rank_inc(data, value):
    if np.isnan(value):
        return None  # Return None if the value to rank is NaN

    value_rank, n = manual_rank(data, value)
    if n > 0: 

        percentile_rank = (Decimal(value_rank) - Decimal(1)) / (Decimal(n) - Decimal(1))
        return float(percentile_rank)  # Convert back to float if necessary
    else:
        return None


def calculate_average(numbers):
    if not numbers: 
        return None
    return sum(numbers) / len(numbers)


def calculate_daily_returns(prices):

    daily_returns = []

    for i in range(1, len(prices)):
        today_price = prices[i]
        yesterday_price = prices[i - 1]

        daily_return = (today_price / yesterday_price) - 1
        daily_returns.append(daily_return)

    return daily_returns


def calculate_standard_deviation(arr):
    if not arr:  # Checks if the array is empty
        return 0
    
    sum_values = 0
    sum_sq = 0
    n = 0

    # Adjust mean calculation to handle varying data structures
    numeric_values = []
    for sub_arr in arr:
        if isinstance(sub_arr, (int, float)):  # Check if the element is a number
            numeric_values.append(sub_arr)
        elif isinstance(sub_arr, list) and len(sub_arr) > 0 and isinstance(sub_arr[0], (int, float)):
            numeric_values.append(sub_arr[0])

    if not numeric_values:  # If no numeric values were found
        return 0

    avg = sum(numeric_values) / len(numeric_values)

    # Now compute the sum of squares
    for value in numeric_values:
        sum_sq += (value - avg) ** 2
        n += 1

    # Only calculate the standard deviation if there are enough data points
    if n > 1:
        return (sum_sq / n) ** 0.5
    else:
        return 0


def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)



def Text_to_num(text, bad_data_val = 0):
    d = {
        'K': 1000,
        'k': 1000,
        'M': 1000000,
        'm': 1000000,
        'B': 1000000000,
        'b': 1000000000
    }

    if not isinstance(text, str):
        return bad_data_val

    elif text[-1] in d:
        num, magnitude = text[:-1], text[-1]
        return int(float(num) * d[magnitude])
    else:
        return float(text)

def SortByMetric(Element, Metric_Ref):
    return Element[Metric_Ref]

def first_day_of_week(date):
  return date + datetime.timedelta(days = -date.weekday())

def portfolio_variance(weights, cov_matrix):
     return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    #pret = np.sum(Mean_returns * weights)
    #pstd = np.sqrt(pvol)
    #return [pret, pvol, pstd][2]

# Define the constraints
def constraint_weights(weights):
    return np.sum(weights) - 1

# Define the expected return function
def portfolio_return(weights, returns):
    return np.sum(weights * returns)

# Define the efficient frontier optimization function
def efficient_frontier(returns, cov_matrix, num_portfolios):
    num_assets = len(returns)
    results = np.zeros((num_portfolios, num_assets + 2))
    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        portfolio_ret = portfolio_return(weights, returns)
        portfolio_var = portfolio_variance(weights, cov_matrix)
        results[i, :num_assets] = weights
        results[i, num_assets] = portfolio_ret
        results[i, num_assets + 1] = portfolio_var
    return results

# Define the optimal portfolio function
def optimal_portfolio(Number_of_Portfolio_Allocations, Mean_returns, cov_matrix, returns):
    num_assets = Number_of_Portfolio_Allocations
    args = (cov_matrix,)
    constraints = ({'type': 'eq', 'fun': constraint_weights})

    bounds = tuple((0.1,1) for _ in range(num_assets))
    initial_weights = num_assets * [1. / num_assets,]
    optimal = minimize(fun=portfolio_variance, x0=initial_weights, method='SLSQP', args=args,constraints=constraints, bounds=bounds, options={'disp': True ,'eps': 1e-8}) 
    return optimal.x


def last_workday_of_month(date):
    business_month_end = pd.tseries.offsets.BMonthEnd()
    last_workday = business_month_end.rollforward(date)
    return last_workday


def find_date_value_multiindex(df, date, column_name, sub_column_name):

    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df.index = pd.to_datetime(df.index)

    target_date = pd.to_datetime(date)


    if target_date in df.index:
        return df.loc[target_date, (column_name, sub_column_name)]
    else:
        return None



def create_json_string(Emergency_Status, Total_Days_Pre_Allarm_ON_Equity, Total_Days_Pre_Allarm_ON_Bond, Total_Days_Allarm_ON):
    # Create a dictionary with the required variable names and values
    data = {
        "emergency status": Emergency_Status,
        "Total Days Pre Allarm Equity": Total_Days_Pre_Allarm_ON_Equity,
        "Total Days Pre Allarm Bond": Total_Days_Pre_Allarm_ON_Bond,
        "Total Days Allarm ON": Total_Days_Allarm_ON
    }
    
    # Convert the dictionary to a JSON string
    json_string = json.dumps(data, indent=4)
    
    return json_string


def clean_data(df, Row_Empty_Threshold, surroundings_Empty_Threshold, start_row=None, end_row=None):
    indices_to_drop = []

    # Store the original start_row and end_row as relative positions within the sliced DataFrame

    original_start_index_label = df.index[start_row]
    original_end_index_label  = df.index[end_row]

    # Adjust start_row and end_row based on provided values
    if start_row is None or start_row - 255 < 0:
        start_row = 0
    else:
        start_row = start_row - 255

    if end_row is None or end_row + 255 >= len(df):
        end_row = len(df) - 1
    else:
        end_row = end_row + 255

    # Convert start_row and end_row from integer positions to actual index labels
    start_index_label = df.index[start_row]
    end_index_label = df.index[end_row]

    # Filter the DataFrame to only include rows within the start_row and end_row range
    df = df.loc[start_index_label:end_index_label]


    for idx in range(1, len(df) - 1):
        current_row = df.iloc[idx]
        previous_row = df.iloc[idx - 1]
        next_row = df.iloc[idx + 1]

        row_date = df.index[idx]

        # Check if the date is a weekend
        if row_date.weekday() >= 5:
            indices_to_drop.append(df.index[idx])
            print(df.index[idx])
            continue

        # Calculate the percentage of NaNs and zeros for the current row
        current_nan_zero_ratio = ((current_row.isna() | (current_row == 0)).sum() / len(current_row)) * 100

        if current_nan_zero_ratio >= Row_Empty_Threshold:
            previous_nan_zero_ratio = ((previous_row.isna() | (previous_row == 0)).sum() / len(previous_row)) * 100
            next_nan_zero_ratio = ((next_row.isna() | (next_row == 0)).sum() / len(next_row)) * 100

            if previous_nan_zero_ratio <= surroundings_Empty_Threshold and next_nan_zero_ratio <= surroundings_Empty_Threshold:
                indices_to_drop.append(df.index[idx])
                print(df.index[idx])

    # Drop the identified rows
    df.drop(indices_to_drop, inplace=True)

    # Assume original_start_index_label and original_end_index_label are the labels for the original start and end rows.

    # Update the original start_row and end_row after dropping rows
    new_start_row = df.index.get_loc(original_start_index_label) if original_start_index_label in df.index else None
    new_end_row = df.index.get_loc(original_end_index_label) if original_end_index_label in df.index else None

    # Print the new start and end row index positions
    print("new_start_row = " + str(new_start_row) + " --- new_end_row = " + str(new_end_row))

    # Fill NaN or zero values with the average of the surrounding values
    for idx in range(1, len(df) - 1):
        for column in df.columns:
            if pd.isna(df.at[df.index[idx], column]) or df.at[df.index[idx], column] == 0:
                previous_value = df.at[df.index[idx - 1], column]
                next_value = df.at[df.index[idx + 1], column]

                if not (pd.isna(previous_value) or previous_value == 0) and not (pd.isna(next_value) or next_value == 0):
                    df.at[df.index[idx], column] = (previous_value + next_value) / 2

                    print(f'Checking Date: {df.index[idx]}---- Price Type : {column[0]} ---- Asset : {column[1]} ---- New Value : {df.at[df.index[idx], column]} ---- previous_value : {previous_value}  ---- next_value : {next_value}', sep='\n')

    return df, new_start_row, new_end_row





def read_csv_file(filename):
    try:
        df = pd.read_csv(filename)
        return df 
    except Exception as e:
        print(f"Error reading Database CSV file: {e}")
        return pd.DataFrame()  


def read_csv_file_2headers(filename):
    try:
        df = pd.read_csv(filename, header=[0, 1], index_col=0, parse_dates=True)
        return df
    except Exception as e:
        print(f"Error reading Database CSV file: {e}")
        return pd.DataFrame()  

def last_workday_of_month_New(date, country='US'):
    # Create a holiday calendar for the specified country
    holiday_calendar = holidays.CountryHoliday(country)
    
    # Find the last business day of the month
    business_month_end = pd.tseries.offsets.BMonthEnd()
    last_workday = business_month_end.rollforward(date)
    
    # Check if the last workday is a holiday
    while last_workday in holiday_calendar:
        # If it's a holiday, move one day back
        last_workday -= pd.Timedelta(days=1)
    
    return last_workday


def Data_exponential_smooth(data, alpha):
    """
    Function that exponentially smooths dataset so values are less 'rigid'
    :param alpha: weight factor to weight recent values more
    """
    
    return data.ewm(alpha=alpha).mean()


def Get_Top_reformers_ByMetric_Count(List_of_Ranked_Assets, Combined_Metric_Array_Dim, Number_of_Portfolio_Allocations):
   
   if len(List_of_Ranked_Assets) >0:

        for Asset in range(len(List_of_Ranked_Assets)):
            List_of_Ranked_Assets[Asset][1] = 0

        for Metric_Number in range(1, Combined_Metric_Array_Dim):
            List_of_Ranked_Assets.sort(key=lambda a: a[6+Metric_Number], reverse=True)
            for Asset in range(len(List_of_Ranked_Assets)):
                if Asset < int(Number_of_Portfolio_Allocations) and np.isnan(List_of_Ranked_Assets[Asset][6+Metric_Number]) == False:
                    List_of_Ranked_Assets[Asset][1] = List_of_Ranked_Assets[Asset][1] + 1
                    #print(f'Asset Symbol: {List_of_Ranked_Assets[Asset][0]} ---- Local Metric Value = {List_of_Ranked_Assets[Asset][7+Metric_Number]}', sep='\n')

        List_of_Ranked_Assets.sort(key=lambda a: a[1], reverse=True)

        del List_of_Ranked_Assets[int(Number_of_Portfolio_Allocations):]
   
        return List_of_Ranked_Assets
   else:

        return []


def Get_Top_reformers_ByMetric_Value(List_of_Ranked_Assets, Number_of_Portfolio_Allocations):

    if len(List_of_Ranked_Assets)>0:

        List_of_Ranked_Assets.sort(key=lambda a: a[1], reverse=True)
        del List_of_Ranked_Assets[int(Number_of_Portfolio_Allocations):]

    return List_of_Ranked_Assets

#Machine Learning Models:

import pandas as pd
import numpy as np

def calculate_volatility(prices):
    """Calculate the volatility of a series of prices."""
    # Calculate daily returns
    returns = prices.pct_change().dropna()
    # Calculate the standard deviation of returns
    volatility = returns.std()
    return volatility

def find_volatility_for_asset(df, asset):

    try:

        adj_close_prices = df["Adj Close"][asset]

        # Find the first and last non-NaN values
        valid_prices = adj_close_prices.dropna()

        if not valid_prices.empty:
            first_valid_date = valid_prices.index[0]
            last_valid_date = valid_prices.index[-1]

            returns = valid_prices.pct_change().dropna()
            volatility = returns.std() * math.sqrt(252)

             #(Asset_Volatility_Array.std().values.tolist())[0] * math.sqrt(252)

            return volatility
        else:
            return np.nan
    except KeyError:

        return np.nan


def _train_random_forest(X_train, y_train, X_test, y_test):

    """
    Function that uses random forest classifier to train the model
    :return:
    """
    
    # Create a new random forest classifier
    rf = RandomForestClassifier()
    
    # Dictionary of all values we want to test for n_estimators
    params_rf = {'n_estimators': [110,130,140,150,160,180,200]}
    
    # Use gridsearch to test all values for n_estimators
    rf_gs = GridSearchCV(rf, params_rf) #, cv=2)
    
    # Fit model to training data
    rf_gs.fit(X_train, y_train)
    
    # Save best model
    rf_best = rf_gs.best_estimator_
    
    # Check best n_estimators value
    #print(rf_gs.best_params_)
    
    #prediction = rf_best.predict(X_test)

    
    return rf_best


def _train_KNN(X_train, y_train, X_test, y_test):

    knn = KNeighborsClassifier()
    # Create a dictionary of all values we want to test for n_neighbors
    params_knn = {'n_neighbors': np.arange(1, 25)}
    
    # Use gridsearch to test all values for n_neighbors
    knn_gs = GridSearchCV(knn, params_knn) #, cv=2)
    
    # Fit model to training data
    knn_gs.fit(X_train, y_train)
    
    # Save best model
    knn_best = knn_gs.best_estimator_
     
    # Check best n_neigbors value
    #print(knn_gs.best_params_)
    
    #prediction = knn_best.predict(X_test)

    #print(classification_report(y_test, prediction))
    #print(confusion_matrix(y_test, prediction))
    
    return knn_best
    
def _train_GBC(X_train, y_train, X_test, y_test):

    GBC = GradientBoostingClassifier()
    # Create a dictionary of all values we want to test for n_neighbors
    params_GBC = {'n_estimators': [110,130,140,150,160,180,200]}
    
    # Use grdsearch to test all values for n_neighbors
    GBC_gs = GridSearchCV(GBC, params_GBC) #, cv=2)
    
    # Fit model to training data
    GBC_gs.fit(X_train, y_train)
    
    # Save best model
    GBC_best = GBC_gs.best_estimator_
     
    # Check best n_neigbors value
    #print(knn_gs.best_params_)
    
    #prediction = knn_best.predict(X_test)

    #print(classification_report(y_test, prediction))
    #print(confusion_matrix(y_test, prediction))
    
    return GBC_best
    #knn_model = _train_KNN(X_train, y_train, X_test, y_test)


def _ensemble_model(rf_model, knn_model, gbt_model, X_train, y_train, X_test, y_test):
    
    # Create a dictionary of our models
    estimators=[('knn', knn_model), ('rf', rf_model), ('gbt', gbt_model)]
    
    # Create our voting classifier, inputting our models
    ensemble = VotingClassifier(estimators, voting='hard')
    
    #fit model to training data
    ensemble.fit(X_train, y_train)
    
    #test our model on the test data
    #print(ensemble.score(X_test, y_test))
    
    #prediction = ensemble.predict(X_test)

    #print(classification_report(y_test, prediction))
    #print(confusion_matrix(y_test, prediction))
    
    return ensemble

def write_to_txt_file(Content_Added, file_path, line_number):
    lines = []

    # Read all existing lines from the file
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
    
    # Ensure the file has enough lines
    while len(lines) < line_number:
        lines.append("\n")

    # Update the specified line
    lines[line_number - 1] = str(Content_Added) + "\n"

    # Write all lines back to the file
    with open(file_path, 'w') as file:
        file.writelines(lines)
        read_from_txt_file

def read_from_txt_file(file_path, line_number, return_type=str):
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return ""
    
    with open(file_path, 'r') as file:
        lines = file.readlines()

    if len(lines) < line_number:
        return ""

    content = lines[line_number - 1].strip()

    if return_type == bool:
        return content.lower() == 'true'
    elif return_type == int:
        return int(content) if content.isdigit() else 0
    elif return_type == float:
        try:
            return float(content)
        except ValueError:
            return 0.0
    else:
        return content




import requests
import pandas as pd
import sys
import http.client
import json
import time

def AlphaVantage_RapidAPI(symbol):
    api_key = 'd1c907c6efmsh002f422951e45bdp1a7256jsn89a3074ab0b4'
    conn = http.client.HTTPSConnection("alpha-vantage.p.rapidapi.com")

    headers = {
        'x-rapidapi-key': api_key,
        'x-rapidapi-host': "alpha-vantage.p.rapidapi.com"
    }

    # Add a 500-millisecond delay before each request
    time.sleep(0.7)  # 500 milliseconds = 0.5 seconds

    conn.request("GET", f"/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&outputsize=full&datatype=json", headers=headers)

    res = conn.getresponse()
    data = res.read()

    # Decode the binary data and parse it as JSON
    try:
        data_dict = json.loads(data.decode("utf-8"))

    #except json.JSONDecodeError:
    except:
        print(f"Error decoding JSON for symbol: {symbol}")
        return None  # Return None if JSON parsing fails

    return data_dict




def fetch_asset_data(Asset_Input_list, Data_Source):
    Asset_Col = 0
    Asset_Prices_Array_Model = pd.DataFrame()

    if Data_Source == "AlphaVanatage":

        for Asset in Asset_Input_list:
            data_dict = AlphaVantage_RapidAPI(symbol=Asset)

            if isinstance(data_dict, dict) and 'Time Series (Daily)' in data_dict:
                data = pd.DataFrame(data_dict['Time Series (Daily)']).transpose()
                data = data.rename(columns=rename_columns)
                #print(f"Data detected for Symbol: {Asset}")

                # Create MultiIndex columns with the asset symbol
                current_headers = data.columns.tolist()
                new_headers = [Asset] * len(current_headers)
                multi_index = pd.MultiIndex.from_tuples(list(zip(current_headers, new_headers)))
                data.columns = multi_index

                # Convert columns to numeric, handling non-numeric entries as NaN
                data = data.apply(lambda x: pd.to_numeric(x, errors='coerce'))

                # Concatenate data for each asset without sorting each time
                Asset_Prices_Array_Model = pd.concat([Asset_Prices_Array_Model, data], axis=1, ignore_index=False)

                # Update progress bar
                Asset_Col += 1
                percentage = (Asset_Col / len(Asset_Input_list)) * 100
                num_dashes = int((percentage / 100) * 50)
                progress_bar = '-' * num_dashes + ' ' * (50 - num_dashes)
                sys.stdout.write(f"\r Assets Data Download Progress - ALPHAVANTAGE: [{progress_bar}] {percentage:.2f}%")
                sys.stdout.flush()
            else:
                print(f"Data not detected for Symbol: {Asset}")


        # Check for and remove duplicate columns in MultiIndex
        if Asset_Prices_Array_Model.columns.duplicated().any():
            Asset_Prices_Array_Model = Asset_Prices_Array_Model.loc[:, ~Asset_Prices_Array_Model.columns.duplicated()]

        # Sort columns once after the loop completes, if no duplicates
        Asset_Prices_Array_Model = Asset_Prices_Array_Model.reindex(sorted(Asset_Prices_Array_Model.columns), axis=1)

    elif Data_Source == "Yahoo_Finanace":
        
        print("Assets Data Download - YAHOO FINANCE: ")

        Asset_Input_Final = ' '.join(Asset_Input_list)
        Asset_Prices_Array_Model =  yfinance.download(Asset_Input_Final, period="max", interval="1d")

    return Asset_Prices_Array_Model




def is_within_threshold(first_value, second_value, threshold):

    difference = first_value - second_value

    if (difference < 0 and difference >= -threshold) or (difference > 0 and difference <= threshold) :
        return True
    else:
        return False


def rename_columns(x):
    if '1. open' in x:
        return f"Open"
    elif '2. high' in x:
        return f"High"
    elif '3. low' in x:
        return f"Low"
    elif '4. close' in x:
        return f"Close"
    elif '5. adjusted close' in x:
        return f"Adj Close"
    elif '6. volume' in x:
        return f"Volume"
    elif '7. dividend amount' in x:
        return f"Dividend Amount"
    elif '8. split coefficient' in x:
        return f"Split Coefficient"

#ensemble_model = _ensemble_model(rf_model, knn_model, gbt_model, X_train, y_train, X_test, y_test)

#  1) SMA Instantaneous Slope:

def SMA_Instantaneous_Slope(Smoothing_Days_Period, Current_Date_Row, Adj_Close_Lookback_Period_Array):
    
    Adj_Close_Lookback_Period_Array = Adj_Close_Lookback_Period_Array.tolist()
    
    if Current_Date_Row - Smoothing_Days_Period  >= 0:
        if Adj_Close_Lookback_Period_Array[Current_Date_Row - Smoothing_Days_Period] != 0:
            Short_SMA = calculate_average(Adj_Close_Lookback_Period_Array[Current_Date_Row - Smoothing_Days_Period:Current_Date_Row])
            Long_SMA = calculate_average(Adj_Close_Lookback_Period_Array[Current_Date_Row- Smoothing_Days_Period:Current_Date_Row+1])
            
            #print("Short_SMA - Lockback Price = " + str(Adj_Close_Lookback_Period_Array[Current_Date_Row - Smoothing_Days_Period]))
            #print("Short_SMA - Current Price = " + str(Adj_Close_Lookback_Period_Array[Current_Date_Row]))
            ##print("Short_SMA = " + str(Short_SMA))

            #print("Long_SMA - Current Price = " + str(Adj_Close_Lookback_Period_Array[Current_Date_Row- Smoothing_Days_Period:Current_Date_Row+1][-1]))
            #print("Long_SMA = " + str(Long_SMA))
            
            SMA_Instantaneous_Slope_Signal = ((Short_SMA / Long_SMA) - 1)
            
            if SMA_Instantaneous_Slope_Signal == 0:
                 SMA_Instantaneous_Slope_Signal = 1

            return SMA_Instantaneous_Slope_Signal
        else:
            return 1
    else:
        return 1
 

#  2) High Low Differential Ratio:

def High_Low_Differential_Ratio(Smoothing_Days_Period, Start_Days_Period, End_Days_Period, Step_Between_Period, Current_Date_Row, Adj_Close_Lookback_Period_Array):
    
    Adj_Close_Lookback_Period_Returns_Daily_Array = []
    Adj_Close_Lookback_Sub_Period_Array = []
    Max_Metric_Signal = -1000
    
    High_Low_Differential_Ratio_Value = 0
    
    Adj_Close_Lookback_Period_Array = Adj_Close_Lookback_Period_Array.tolist()
    
    for Processed_Element_2 in range(Start_Days_Period, End_Days_Period+1, Step_Between_Period):  
        
        Current_Price = 0
        Maximum_Price = 0
        Minimum_Price = 0
        
        Low_Differential = 0
        High_Differential = 0
        High_Low_Differential_Ratio = 0
        
        Smoothing_Days_Period = Processed_Element_2 
        
        if Current_Date_Row - Smoothing_Days_Period >= 0:
            if Adj_Close_Lookback_Period_Array[Current_Date_Row - Smoothing_Days_Period] != 0:

                Adj_Close_Lookback_Sub_Period_Array = Adj_Close_Lookback_Period_Array[Current_Date_Row - Smoothing_Days_Period:Current_Date_Row+1]
                 
                Current_Price = Adj_Close_Lookback_Sub_Period_Array[-1]
                Maximum_Price = max(Adj_Close_Lookback_Sub_Period_Array)
                Minimum_Price = min(Adj_Close_Lookback_Sub_Period_Array)

                #print("Processed_Element = " + str(Processed_Element_2))

                Price_Standard_Deviation = calculate_standard_deviation(Adj_Close_Lookback_Sub_Period_Array)

                #print("High_Low_Differential_Ratio - Lookback Price = " + str(Adj_Close_Lookback_Sub_Period_Array[0]))
                #print("High_Low_Differential_Ratio - Current Price = " + str(Adj_Close_Lookback_Sub_Period_Array[-1]))

                #print("High Low Differential  - Maximum_Price = " + str(Maximum_Price))
                #print("High Low Differential  - Minimum_Price =  " + str(Minimum_Price))
                #print("High Low Differential  - Price_Standard_Deviation =  " + str(Price_Standard_Deviation))

                if Price_Standard_Deviation == 0:
                    Price_Standard_Deviation = 1
    
                if Current_Price != 0:
                    High_Differential = ((Maximum_Price / Current_Price) / Price_Standard_Deviation)
    
                if Minimum_Price != 0:
                    Low_Differential = ((Current_Price / Minimum_Price) / Price_Standard_Deviation)
    
                High_Low_Differential_Ratio_Value = ((Low_Differential - High_Differential) * 100)
                
                if High_Low_Differential_Ratio_Value > Max_Metric_Signal:
                    Max_Metric_Signal = High_Low_Differential_Ratio_Value
            else:
                High_Low_Differential_Ratio_Value =  1
        else:
            High_Low_Differential_Ratio_Value =  1
            
    if Max_Metric_Signal != -1000:
        High_Low_Differential_Ratio_Value = Max_Metric_Signal
        
    return High_Low_Differential_Ratio_Value

#  3) Linear Regression Forecast/Intercept Ratio:

def Linear_Regression_Ratio(Smoothing_Days_Period, Start_Days_Period, End_Days_Period, Step_Between_Period, Current_Date_Row, Adj_Close_Lookback_Period_Array):


    Adj_Close_Lookback_Sub_Period_Array = []
    
    Max_Metric_Signal = -1000
    
    Adj_Close_Lookback_Period_Array = Adj_Close_Lookback_Period_Array.tolist()
    
    for Processed_Element_2 in range(Start_Days_Period, End_Days_Period+1, Step_Between_Period):   

        Count_Sum = 0
        Prices_Sum = 0
        Count_Prices_Product_Sum = 0
        Count_Squared_Sum = 0
        B_Variable = 0
        A_Variable = 0
        Linear_Regression_Indicator = 0
        Linear_Regression_Ratio = 0

        Smoothing_Days_Period = Processed_Element_2 

        if Current_Date_Row - Smoothing_Days_Period >= 0:
            if Adj_Close_Lookback_Period_Array[Current_Date_Row - Smoothing_Days_Period] != 0:
                
                Adj_Close_Lookback_Sub_Period_Array = Adj_Close_Lookback_Period_Array[Current_Date_Row - Smoothing_Days_Period:Current_Date_Row+1]

                #print("Processed_Element = " + str(Processed_Element_2))

                for Processed_Element in range(0, len(Adj_Close_Lookback_Sub_Period_Array)):
                    Prices_Sum = Prices_Sum + Adj_Close_Lookback_Sub_Period_Array[Processed_Element]
                    Count_Sum = Count_Sum + (Processed_Element+1)
                    Count_Prices_Product_Sum = Count_Prices_Product_Sum + ( (Processed_Element+1) * Adj_Close_Lookback_Sub_Period_Array[Processed_Element])
                    Count_Squared_Sum = Count_Squared_Sum + (( (Processed_Element+1)) ** 2)

                if ((len(Adj_Close_Lookback_Sub_Period_Array) * Count_Squared_Sum) - (Count_Sum ** 2)) != 0 and len(Adj_Close_Lookback_Sub_Period_Array) != 0:
    
                    B_Variable = (((len(Adj_Close_Lookback_Sub_Period_Array) * Count_Prices_Product_Sum) - (Count_Sum * Prices_Sum)) / ((len(Adj_Close_Lookback_Sub_Period_Array) * Count_Squared_Sum) - (Count_Sum ** 2)))
    
                    A_Variable = ((Prices_Sum * Count_Squared_Sum) - (Count_Sum * Count_Prices_Product_Sum)) / ((len(Adj_Close_Lookback_Sub_Period_Array) * Count_Squared_Sum) - (Count_Sum ** 2))
    
                    Linear_Regression_Indicator = ((A_Variable) + (B_Variable * (len(Adj_Close_Lookback_Sub_Period_Array))))
                    
                    #print("Linear Regression Forecast/Intercept Ratio - Lookback Price = " + str(Adj_Close_Lookback_Sub_Period_Array[0]))
                    #print("Linear Regression Forecast/Intercept Ratio - Current Price = " + str(Adj_Close_Lookback_Sub_Period_Array[-1]))

                    #print("Linear Regression Forecast/Intercept Ratio  - B_Variable = " + str(B_Variable))
                    #print("Linear Regression Forecast/Intercept Ratio  - A_Variable =  " + str(A_Variable))
                    #print("Linear Regression Forecast/Intercept Ratio  - Linear_Regression_Indicator =  " + str(Linear_Regression_Indicator))

                    if A_Variable != 0:
    
                        Linear_Regression_Ratio = (Linear_Regression_Indicator / A_Variable) - 1
                          
                        if Linear_Regression_Ratio > Max_Metric_Signal:
                            Max_Metric_Signal = Linear_Regression_Ratio

                    else:
                        Linear_Regression_Ratio = 1
                else:
                    Linear_Regression_Ratio = 1
            else:
                Linear_Regression_Ratio = 1
        else:
            Linear_Regression_Ratio =  1

    if Max_Metric_Signal != -1000:
        Linear_Regression_Ratio = Max_Metric_Signal
        
    return Linear_Regression_Ratio


#  4) Stochastic Oscillator:

def Stochastic_Oscillator(Smoothing_Days_Period,Start_Days_Period, End_Days_Period, Step_Between_Period, Current_Date_Row, Adj_Close_Lookback_Period_Array):

    Max_Metric_Signal = -1000
    Adj_Close_Lookback_Sub_Period_Array = []
    Adj_Close_Lookback_Period_Array = Adj_Close_Lookback_Period_Array.tolist()
    
    for Processed_Element in range(Start_Days_Period, End_Days_Period+1, Step_Between_Period):   
                 
        Maximum_Price = 0
        Minimum_Price = 100000000
        Stochastic_Oscillator = 0

        Smoothing_Days_Period = Processed_Element     
    
        if Current_Date_Row - Smoothing_Days_Period>= 0:
            if Adj_Close_Lookback_Period_Array[Current_Date_Row - Smoothing_Days_Period] != 0:

                Adj_Close_Lookback_Sub_Period_Array = Adj_Close_Lookback_Period_Array[Current_Date_Row - Smoothing_Days_Period:Current_Date_Row+1]

                #print("Stochastic_Oscillator Price = " + str(Adj_Close_Lookback_Sub_Period_Array[0]))
                #print("Processed_Element = " + str(Processed_Element))

                for Processed_Element_1 in range(0, len(Adj_Close_Lookback_Sub_Period_Array)):
                      
                    if Adj_Close_Lookback_Sub_Period_Array[Processed_Element_1] > Maximum_Price:
                        Maximum_Price = Adj_Close_Lookback_Sub_Period_Array[Processed_Element_1]
                        
                    if Adj_Close_Lookback_Sub_Period_Array[Processed_Element_1] < Minimum_Price and Adj_Close_Lookback_Sub_Period_Array[Processed_Element_1] != 0:
                        Minimum_Price = Adj_Close_Lookback_Sub_Period_Array[Processed_Element_1]
    
                Current_Price = Adj_Close_Lookback_Sub_Period_Array[-1]
    
                if (Maximum_Price - Minimum_Price) != 0:
    
                    Stochastic_Oscillator = ((Current_Price - Minimum_Price) / (Maximum_Price - Minimum_Price)) * 100

                    #print("Stochastic Oscillator  - Lookback Price = " + str(Adj_Close_Lookback_Sub_Period_Array[0]))
                    #print("Stochastic Oscillator  - Current Price = " + str(Adj_Close_Lookback_Sub_Period_Array[-1]))

                    #print("Stochastic Oscillator  - Maximum_Price = " + str(Maximum_Price))
                    #print("Stochastic Oscillator  - Minimum_Price =  " + str(Minimum_Price))
                    #print("Stochastic Oscillator  - Stochastic_Oscillator =  " + str(Stochastic_Oscillator))

                    if Stochastic_Oscillator > Max_Metric_Signal:
                        Max_Metric_Signal = Stochastic_Oscillator

                    if Stochastic_Oscillator > 100 or Stochastic_Oscillator == 0:
                        Stochastic_Oscillator = 1

                else:
                    Stochastic_Oscillator = 1

            else:
                Stochastic_Oscillator = 1
        else:
            Stochastic_Oscillator = 1
        
    if Max_Metric_Signal != -1000:
        Stochastic_Oscillator = Max_Metric_Signal
        
    return Stochastic_Oscillator

# 5) Price Percent Rank:

def Price_Percent_Rank(Smoothing_Days_Period,Start_Days_Period, End_Days_Period, Step_Between_Period, Current_Date_Row, Adj_Close_Lookback_Period_Array):

    # Start_Days_Period = 4
    # End_Days_Period = 110
    # Step_Between_Period = 2

    Max_Metric_Signal = -1000
    
    Adj_Close_Lookback_Sub_Period_Array = []
    
    Adj_Close_Lookback_Period_Array = Adj_Close_Lookback_Period_Array.tolist()

    for Processed_Element in range(Start_Days_Period, End_Days_Period+1, Step_Between_Period):

        #print("Processed_Element = " + str(Processed_Element))         
        
        Smoothing_Days_Period = Processed_Element     

        if Current_Date_Row - Smoothing_Days_Period>= 0:
            if Adj_Close_Lookback_Period_Array[Current_Date_Row - Smoothing_Days_Period] != 0:

                Adj_Close_Lookback_Sub_Period_Array = Adj_Close_Lookback_Period_Array[Current_Date_Row - Smoothing_Days_Period: Current_Date_Row+1]
                #Adj_Close_Lookback_Sub_Period_Array = np.round(Adj_Close_Lookback_Sub_Period_Array, 3)
                Adj_Close_Lookback_Sub_Period_Array = [round(num, 3) for num in Adj_Close_Lookback_Sub_Period_Array]
                Current_Price = Adj_Close_Lookback_Sub_Period_Array[-1]

                #print("Price_Percent_Rank Price = " + str(Adj_Close_Lookback_Sub_Period_Array[0]))

                #Price_Percent_Rank = PercentRank_Inc(Adj_Close_Lookback_Sub_Period_Array, Current_Price) /100
                Price_Percent_Rank = excel_percent_rank_inc(Adj_Close_Lookback_Sub_Period_Array, Current_Price)
                
                if Price_Percent_Rank > Max_Metric_Signal:
                    Max_Metric_Signal = Price_Percent_Rank

                #print("Price Percent Rank   - Lookback Price = " + str(Adj_Close_Lookback_Sub_Period_Array[0]))
                #print("Price Percent Rank   - Current Price = " + str(Adj_Close_Lookback_Sub_Period_Array[-1]))

                #print("Price Percent Rank  - Price_Percent_Rank =  " + str(Price_Percent_Rank))

                if Price_Percent_Rank == 0:
                    Price_Percent_Rank = 1
            else:
                Price_Percent_Rank = 1
        else:
            Price_Percent_Rank = 1
        
    if Max_Metric_Signal != -1000:
        Price_Percent_Rank = Max_Metric_Signal
        
    return Price_Percent_Rank


#  6) RSI Index Indicator:

def RSI_Index_Indicator(Smoothing_Days_Period, Start_Days_Period, End_Days_Period, Step_Between_Period, Current_Date_Row, Adj_Close_Lookback_Period_Array):
    
    Max_Metric_Signal = -1000
    
    Adj_Close_Lookback_Sub_Period_Array = []
    
    Adj_Close_Lookback_Period_Array = Adj_Close_Lookback_Period_Array.tolist()
    
    for Processed_Element in range(Start_Days_Period, End_Days_Period+1, Step_Between_Period):
        
        Smoothing_Days_Period = Processed_Element   
        
        EMA_Positive_Array=[]
        EMA_Negative_Array=[]

        Previous_EMA = 0
        Current_EMA = 0
        EMA_Change = 0
        EMA_Upday_Closing_Gains = 0
        EMA_Downday_Closing_Losses = 0
        RS_Ratio = 0
        RSI_Index = 0
        #print("Processed_Element = " + str(Processed_Element))
        Exponential_Multiplier = (2 / (Smoothing_Days_Period + 1))
        
        if Current_Date_Row - Smoothing_Days_Period > 0:
            if Adj_Close_Lookback_Period_Array[Current_Date_Row - Smoothing_Days_Period] != 0:

                Adj_Close_Lookback_Sub_Period_Array = Adj_Close_Lookback_Period_Array[Current_Date_Row - Smoothing_Days_Period:Current_Date_Row+1]

                #print("RSI_Index_Indicator Price = " + str(Adj_Close_Lookback_Sub_Period_Array[0]))

                if len(Adj_Close_Lookback_Sub_Period_Array)> Smoothing_Days_Period - 2:
        
                    for Processed_Element_2  in range(len(Adj_Close_Lookback_Sub_Period_Array) - Smoothing_Days_Period + 1, len(Adj_Close_Lookback_Sub_Period_Array)):
        
                        Previous_EMA = Adj_Close_Lookback_Sub_Period_Array[Processed_Element_2-1]
                        Current_EMA = Adj_Close_Lookback_Sub_Period_Array[Processed_Element_2]
        
                        EMA_Change = Current_EMA - Previous_EMA
        
                        if EMA_Change > 0:
                            EMA_Positive_Array.append(EMA_Change)
                        else:
                            EMA_Positive_Array.append(0)
        
                        if EMA_Change < 0:
                            EMA_Negative_Array.append(abs(EMA_Change))
                        else:
                            EMA_Negative_Array.append(0)
        
                    EMA_Upday_Closing_Gains = calculate_average(EMA_Positive_Array)
                    EMA_Downday_Closing_Losses = calculate_average(EMA_Negative_Array)
        
                    if EMA_Downday_Closing_Losses != 0:
                        RS_Ratio = (EMA_Upday_Closing_Gains) / (EMA_Downday_Closing_Losses)
                        
                        if RS_Ratio != -1:
                            RSI_Index = ((100 - (100 / (1 + (RS_Ratio)))))
                            RSI_Index_Indicator_Value = RSI_Index
                        else:
                            RSI_Index = 0
                            RSI_Index_Indicator_Value = 1

                    else:
                        RSI_Index = 0
                        RSI_Index_Indicator_Value = 1

                    #print("RSI Index Indicator  - Lookback Price = " + str(Adj_Close_Lookback_Sub_Period_Array[0]))
                    #print("RSI Index Indicator  - Current Price = " + str(Adj_Close_Lookback_Sub_Period_Array[-1]))

                    #print("RSI Index Indicator - EMA_Upday_Closing_Gains = " + str(EMA_Upday_Closing_Gains))
                    #print("RSI Index Indicator - EMA_Downday_Closing_Losses =  " + str(EMA_Downday_Closing_Losses))
                    #print("RSI Index Indicator  - RS_Ratio =  " + str(RS_Ratio))
                    #print("RSI Index Indicator  - RSI_Index =  " + str(RSI_Index))

                else:
                   RSI_Index = 0
                   RSI_Index_Indicator_Value = 1
            else:
               RSI_Index = 0
               RSI_Index_Indicator_Value = 1
  
        if RSI_Index > Max_Metric_Signal:
            Max_Metric_Signal = RSI_Index
            
        if RSI_Index > 100 or RSI_Index <= 0:
            Max_Metric_Signal = 1
            
    if Max_Metric_Signal != -1000 and Max_Metric_Signal != 1:
        RSI_Index_Indicator_Value = Max_Metric_Signal / 100

    return RSI_Index_Indicator_Value
       
def Momentum_MultiFactor_Maximizer(Lookback_Period, Current_Date_Row, Adj_Close_Lookback_Period_Array):

    High_Low_Differential_Ratio_Value = High_Low_Differential_Ratio(Lookback_Period, 22, 132, 22, Current_Date_Row, Adj_Close_Lookback_Period_Array)
    Linear_Regression_Ratio_Value = Linear_Regression_Ratio(Lookback_Period, 11, 132, 11, Current_Date_Row, Adj_Close_Lookback_Period_Array )
    Price_Percent_Rank_Value= Price_Percent_Rank(Lookback_Period, 4, 110, 2, Current_Date_Row, Adj_Close_Lookback_Period_Array)
    RSI_Index_Indicator_Value = RSI_Index_Indicator(Lookback_Period, 11, 132, 11, Current_Date_Row, Adj_Close_Lookback_Period_Array)
    SMA_Instantaneous_Slope_Signal_Value = SMA_Instantaneous_Slope(Lookback_Period, Current_Date_Row, Adj_Close_Lookback_Period_Array)
    Stochastic_Oscillator_Value  =Stochastic_Oscillator(Lookback_Period, 22, 132, 22, Current_Date_Row, Adj_Close_Lookback_Period_Array)
    
    Momentum_MultiFactor_Maximizer_Value = High_Low_Differential_Ratio_Value * abs(Linear_Regression_Ratio_Value) * Price_Percent_Rank_Value  * RSI_Index_Indicator_Value  * abs(SMA_Instantaneous_Slope_Signal_Value) * abs(Stochastic_Oscillator_Value)
    
    #print("High_Low_Differential_Ratio_Value = " + str(High_Low_Differential_Ratio_Value))  
    #print("Linear_Regression_Ratio_Value = " + str(Linear_Regression_Ratio_Value))     
    #print("Price_Percent_Rank_Value = " + str(Price_Percent_Rank_Value))     
    #print("RSI_Index_Indicator_Value = " + str(RSI_Index_Indicator_Value))     
    #print("SMA_Instantaneous_Slope_Signal_Value = " + str(SMA_Instantaneous_Slope_Signal_Value))    
    #print("Stochastic_Oscillator_Value = " + str(Stochastic_Oscillator_Value))   

    MetricsData_Array  = [High_Low_Differential_Ratio_Value, Linear_Regression_Ratio_Value, Price_Percent_Rank_Value, RSI_Index_Indicator_Value, SMA_Instantaneous_Slope_Signal_Value, Stochastic_Oscillator_Value]

    return Momentum_MultiFactor_Maximizer_Value, MetricsData_Array


def Special_Asset_Filter(original_momentum_array, concentration):

    # Load assets classifications
    Assets_Details_DB = pd.read_csv('tmp/Model_Assets_classifications.csv')
    
    # Default output value for non-selected assets
    default_out_value = -1000
    total_number_of_assets = len(original_momentum_array)
    
    if concentration == "Sub_Classes":
        for asset_index in range(total_number_of_assets):
            asset_data = original_momentum_array[asset_index]
            symbol = asset_data[0]
            value = asset_data[1]
            
            # Get asset details from the database
            asset_details = Assets_Details_DB[Assets_Details_DB['Asset Symbol'] == symbol]
            if asset_details.empty:
                continue
            
            asset_class = asset_details['Asset Class'].values[0]
            subclass = asset_details['Asset Sub-Class'].values[0]
            sector = asset_details['Asset Sector'].values[0]
            country = asset_details['Asset Country'].values[0]
            
            # Filter assets based on the specified criteria
            matched_assets = Assets_Details_DB[
                (Assets_Details_DB['Asset Class'] == asset_class) &
                (
                    ((Assets_Details_DB['Asset Sub-Class'] == subclass))
                )
            ]

            #(Assets_Details_DB['Asset Country'] == country) & 
            # Find the best-performing asset in this matched group
            best_asset_value = default_out_value
            best_asset_symbol = None
            
            for _, matched_asset in matched_assets.iterrows():
                matched_symbol = matched_asset['Asset Symbol']
                
                # Find the matched asset in the original momentum array
                for data in original_momentum_array:
                    if data[0] == matched_symbol:
                        matched_value = data[1]
                        if matched_value > best_asset_value:
                            best_asset_value = matched_value
                            best_asset_symbol = matched_symbol
                        break  # Exit inner loop if matched symbol is found
            
            # Set default_out_value for all non-best assets in the matched group
            for data in original_momentum_array:
                if data[0] in matched_assets['Asset Symbol'].values and data[0] != best_asset_symbol:
                    data[1] = default_out_value
                    print(f"Kicked out symbol: {data[0]}, Out value: {default_out_value}, Group: {asset_class}, {subclass}, {sector}, {country}")

            # Output debug information for the best asset in the group
            if best_asset_symbol:
                print(f"Best symbol: {best_asset_symbol}, Best value: {best_asset_value}, Group: {asset_class}, {subclass}, {sector}, {country}")


    # Markets logic

    elif concentration == "Markets":

        Assets_Details_DB_List = Assets_Details_DB['Asset Sub-Class'].unique()
        sub_class_markets = {
            "US Markets": {"filter": "US -", "exclude": ["Mid", "Bonds", "Bond"]},
            "Global Markets": {"filter": "Global", "exclude": ["Bonds"]},
            "Developed Markets": {"filter": "Global - Developed Markets Equities", "exclude": []},
            "Emerging Markets": {"filter": "Global - Emerging Markets Equities", "exclude": []},
            "Emerging Markets - Asia": {"filter": "Global - Emerging Markets Equities - Asia", "exclude": []},
            "Emerging Markets - Others": {"filter": "Global - Emerging Markets Equities - Others", "exclude": []},
        }

        # Arrays to store filtered assets
        filtered_assets = {market: [] for market in sub_class_markets}

        # Filter assets by sub-class and store the values
        for asset_index in range(total_number_of_assets):
            asset_data = original_momentum_array[asset_index]
            symbol = asset_data[0]  # Assuming symbol is the first value
            value = asset_data[1]   # Assuming value is the second value

            # Get asset details from the database
            asset_data_db = Assets_Details_DB[Assets_Details_DB['Asset Symbol'] == symbol]

            if not asset_data_db.empty:
                current_subclass = asset_data_db['Asset Sub-Class'].values[0]

                # Filter assets into sub-markets
                for market, conditions in sub_class_markets.items():
                    if conditions["filter"] in current_subclass and all(excl not in current_subclass for excl in conditions["exclude"]):
                        filtered_assets[market].append((value, symbol))

        # Calculate the returns for each sub-class
        for market, assets in filtered_assets.items():
            if len(assets) > 0:
                market_return = sum(float(asset[0]) for asset in assets)
                Markets_Returns_Array_Metrics.append([market_return, market])

        # Find maximum and lowest market returns
        if Markets_Returns_Array_Metrics:
            max_market_return, max_market_name = max(Markets_Returns_Array_Metrics, key=lambda x: x[0])
        else:
            max_market_name = "US Markets"  # Default to US Markets

        # Mark non-relevant markets to the default_out_value
        for asset_index in range(total_number_of_assets):
            symbol = original_momentum_array[asset_index][0]
            value = original_momentum_array[asset_index][1]
            if max_market_name == "US Markets" and symbol in [asset[1] for asset in filtered_assets["Global Markets"]]:
                original_momentum_array[asset_index][1] = default_out_value

        return original_momentum_array, max_market_return


    elif concentration == "Sectors":

        Assets_Details_DB_List = Assets_Details_DB['Asset Sector'].unique()

        for sector in Assets_Details_DB_List:
            best_asset_value = default_out_value
            best_asset_symbol = None

            # Find the best-performing asset in the sector
            for asset_index in range(total_number_of_assets):
                asset_data = original_momentum_array[asset_index]
                symbol = asset_data[0]  # Assuming symbol is the first value
                value = asset_data[1]   # Assuming value is the second value
                asset_data_db = Assets_Details_DB[Assets_Details_DB['Asset Symbol'] == symbol]

                print("sector = " + str(sector))

                if not asset_data_db.empty and asset_data_db['Asset Sector'].values[0] == sector:
                   
                    print(asset_data_db['Asset Sector'].values[0])

                    if value > best_asset_value:
                        best_asset_value = value
                        best_asset_symbol = symbol

            # Mark all other assets in the sector with default_out_value
            for asset_index in range(total_number_of_assets):
                symbol = original_momentum_array[asset_index][0]
                value = original_momentum_array[asset_index][1]
                asset_data_db = Assets_Details_DB[Assets_Details_DB['Asset Symbol'] == symbol]
                if not asset_data_db.empty and asset_data_db['Asset Sector'].values[0] == sector and symbol != best_asset_symbol:
                    original_momentum_array[asset_index][1] = default_out_value
                elif not asset_data_db.empty and asset_data_db['Asset Sector'].values[0] == sector and symbol == best_asset_symbol:
                    print("Best symbol = " + str(best_asset_value))
                    print("Best value = " + str(best_asset_value))
                    print("Best symbol - sector = " + sector)

    return original_momentum_array


















#|
#                    (Assets_Details_DB['Asset Sub-Class'] == subclass)


#def specific_markets_strategy_1(original_momentum_array):

#    Assets_Details_DB = read_csv_file("Model_Assets_classifications.csv")

#    if not Assets_Details_DB.empty:
#        Assets_Details_DB_List = Assets_Details_DB['Asset Sub-Class'].unique()
#    else:
#        Assets_Details_DB_List = []
   
#    Markets_Returns_Array = ["US Markets", "Global Markets", "Global Markets - General", "Developed Markets - General", "Emerging Markets - General", "Emerging Markets - Asia", "Emerging Markets - Others"]
   
#    #Markets_Returns_Array = ["US Markets", "Global Markets", "Global Markets - General", "Developed Markets - General", "Developed Markets - Canada & Europe", "Developed Markets - Asia & Pacific", "Emerging Markets - General", "Emerging Markets - Asia", "Emerging Markets - Others"]

#    #Markets_Returns_Array = ["US Markets", "Emerging Markets - General", "Emerging Markets - Asia"]

#    Markets_Returns_Array_Metrics = []

#    #asset_found_in_specific_markets = None

#    default_out_value = -1000
    
#    total_number_of_participating_asset = len(original_momentum_array)

#    # Initialize asset arrays
#    us_sub_class_assets = np.array([], dtype=float)
#    global_sub_class_assets = np.array([], dtype=float)
#    general_global_sub_class_assets = np.array([], dtype=float)
#    developed_sub_class_assets = np.array([], dtype=float)
#    developed_canadaneurope_sub_class_assets = np.array([], dtype=float)
#    developed_asia_pacific_sub_class_assets = np.array([], dtype=float)
#    emerging_sub_class_assets = np.array([], dtype=float)
#    emerging_asia_sub_class_assets = np.array([], dtype=float)
#    emerging_others_sub_class_assets = np.array([], dtype=float)

#    US_Markets_ON = False
#    Global_Markets_ON = False
#    General_Global_Markets_ON = False
#    Developed_Markets_ON = False 
#    Developed_CanadaNEurope_Markets_ON = False 
#    Developed_Asia_Pacific_Markets_ON = False 
#    Emerging_Markets_ON = False 
#    Emerging_Asia_Markets_ON = False 
#    Emerging_Others_Markets_ON = False

#    us_sub_class_market_returns = default_out_value
#    global_sub_class_market_returns = default_out_value
#    general_global_sub_class_market_returns = default_out_value
#    developed_sub_class_market_returns = default_out_value
#    developed_canadaneurope_sub_class_market_returns = default_out_value
#    developed_asia_pacific_sub_class_returns = default_out_value
#    emerging_sub_class_returns = default_out_value
#    emerging_asia_sub_class_returns = default_out_value
#    emerging_others_sub_class_returns = default_out_value


#    for subclass in Assets_Details_DB_List:
#        for Asset in range(total_number_of_participating_asset):
#            processed_symbol = original_momentum_array[Asset][0]
#            find_processed_symbol_assets_sheet = Assets_Details_DB[Assets_Details_DB['Asset Symbol'] == processed_symbol]

#            if not find_processed_symbol_assets_sheet.empty:
#                category_match = find_processed_symbol_assets_sheet['Asset Sub-Class'].values[0] == subclass

#                if category_match and "US -" in subclass and all(x not in subclass for x in ["Mid", "Bonds", "Bond"]) and original_momentum_array[Asset][1] != -1000:
#                    us_sub_class_assets = np.append(us_sub_class_assets, [original_momentum_array[Asset][1], processed_symbol])
#                if category_match and "Global" in subclass and all(x not in subclass for x in ["Bonds"]) and original_momentum_array[Asset][1] != -1000:
#                    global_sub_class_assets = np.append(global_sub_class_assets, [original_momentum_array[Asset][1], processed_symbol])
#                if category_match and any(x in subclass for x in ["Global - Global Equities", "Global - Real Estate"]) and original_momentum_array[Asset][1] != -1000:
#                    general_global_sub_class_assets = np.append(general_global_sub_class_assets, [original_momentum_array[Asset][1], processed_symbol])
#                if category_match and any(x in subclass for x in ["Global - Developed Markets Equities"]) and original_momentum_array[Asset][1] != -1000:
#                    developed_sub_class_assets = np.append(developed_sub_class_assets, [original_momentum_array[Asset][1], processed_symbol])
#                if category_match and any(x in subclass for x in ["Global - Developed Markets Equities - Canada & Europe"]) and original_momentum_array[Asset][1] != -1000:
#                    developed_canadaneurope_sub_class_assets = np.append(developed_canadaneurope_sub_class_assets, [original_momentum_array[Asset][1], processed_symbol])
#                if category_match and any(x in subclass for x in ["Global - Developed Markets Equities - Asia"]) and original_momentum_array[Asset][1] != -1000:
#                    developed_asia_pacific_sub_class_assets = np.append(developed_asia_pacific_sub_class_assets, [original_momentum_array[Asset][1], processed_symbol])
#                if category_match and any(x in subclass for x in ["Global - Emerging Markets Equities"]) and original_momentum_array[Asset][1] != -1000:
#                    emerging_sub_class_assets = np.append(emerging_sub_class_assets, [original_momentum_array[Asset][1], processed_symbol])
#                if category_match and any(x in subclass for x in ["Global - Emerging Markets Equities - Asia"]) and original_momentum_array[Asset][1] != -1000:
#                    emerging_asia_sub_class_assets = np.append(emerging_asia_sub_class_assets, [original_momentum_array[Asset][1], processed_symbol])
#                if category_match and any(x in subclass for x in ["Global - Emerging Markets Equities - Others"]) and original_momentum_array[Asset][1] != -1000:
#                    emerging_others_sub_class_assets = np.append(emerging_others_sub_class_assets, [original_momentum_array[Asset][1], processed_symbol])


#    if len(us_sub_class_assets) > 0:
#        us_sub_class_assets = np.array(us_sub_class_assets).reshape(-1, 2)
#        us_sub_class_market_returns = us_sub_class_assets[:, 0].astype(float).sum()

#    if len(global_sub_class_assets) > 0:
#        global_sub_class_assets = np.array(global_sub_class_assets).reshape(-1, 2)
#        global_sub_class_market_returns = global_sub_class_assets[:, 0].astype(float).sum()

#    if len(general_global_sub_class_assets) > 0:
#        general_global_sub_class_assets = np.array(general_global_sub_class_assets).reshape(-1, 2)
#        general_global_sub_class_market_returns = general_global_sub_class_assets[:, 0].astype(float).sum()

#    if len(developed_sub_class_assets) > 0:
#        developed_sub_class_assets = np.array(developed_sub_class_assets).reshape(-1, 2)
#        developed_sub_class_market_returns = developed_sub_class_assets[:, 0].astype(float).sum()

#    if len(developed_canadaneurope_sub_class_assets) > 0:
#        developed_canadaneurope_sub_class_assets = np.array(developed_canadaneurope_sub_class_assets).reshape(-1, 2)
#        developed_canadaneurope_sub_class_market_returns = developed_canadaneurope_sub_class_assets[:, 0].astype(float).sum()

#    if len(developed_asia_pacific_sub_class_assets) > 0:
#        developed_asia_pacific_sub_class_assets = np.array(developed_asia_pacific_sub_class_assets).reshape(-1, 2)
#        developed_asia_pacific_sub_class_returns = developed_asia_pacific_sub_class_assets[:, 0].astype(float).sum()

#    if len(emerging_sub_class_assets) > 0:
#        emerging_sub_class_assets = np.array(emerging_sub_class_assets).reshape(-1, 2)
#        emerging_sub_class_returns = emerging_sub_class_assets[:, 0].astype(float).sum()

#    if len(emerging_asia_sub_class_assets) > 0:
#        emerging_asia_sub_class_assets = np.array(emerging_asia_sub_class_assets).reshape(-1, 2)
#        emerging_asia_sub_class_returns = emerging_asia_sub_class_assets[:, 0].astype(float).sum()

#    if len(emerging_others_sub_class_assets) > 0:
#        emerging_others_sub_class_assets = np.array(emerging_others_sub_class_assets).reshape(-1, 2)
#        emerging_others_sub_class_returns = emerging_others_sub_class_assets[:, 0].astype(float).sum()



#    for Market in Markets_Returns_Array:
#        if Market == "US Markets":
#            Markets_Returns_Array_Metrics.append([us_sub_class_market_returns, Market])
#        elif Market == "Global Markets":
#            Markets_Returns_Array_Metrics.append([global_sub_class_market_returns, Market])
#        elif Market == "Global Markets - General":
#            Markets_Returns_Array_Metrics.append([general_global_sub_class_market_returns, Market])
#        elif Market == "Developed Markets - General":
#            Markets_Returns_Array_Metrics.append([developed_sub_class_market_returns, Market])
#        elif Market == "Developed Markets - Canada & Europe":
#            Markets_Returns_Array_Metrics.append([developed_canadaneurope_sub_class_market_returns, Market])
#        elif Market == "Developed Markets - Asia & Pacific":
#            Markets_Returns_Array_Metrics.append([developed_asia_pacific_sub_class_returns, Market])
#        elif Market == "Emerging Markets - General":
#            Markets_Returns_Array_Metrics.append([emerging_sub_class_returns, Market])
#        elif Market == "Emerging Markets - Asia":
#            Markets_Returns_Array_Metrics.append([emerging_asia_sub_class_returns, Market])
#        elif Market == "Emerging Markets - Others":
#            Markets_Returns_Array_Metrics.append([emerging_others_sub_class_returns, Market])

       
#    Maximum_Market_Return = max([item[0] for item in Markets_Returns_Array_Metrics])
#    Lowest_Return_for_Markets = min([item[0] for item in Markets_Returns_Array_Metrics]) 
        
#    try:

#        Maximum_Market_Reference = Markets_Returns_Array_Metrics[:][0].index(Maximum_Market_Return)

#        if Markets_Returns_Array_Metrics[Maximum_Market_Reference][1] == "US Markets":
#            US_Markets_ON = True 
#            Global_Markets_ON = False
#            General_Global_Markets_ON = False
#            Developed_Markets_ON = False
#            Developed_CanadaNEurope_Markets_ON = False
#            Developed_Asia_Pacific_Markets_ON = False
#            Emerging_Markets_ON = False
#            Emerging_Asia_Markets_ON = False
#            Emerging_Others_Markets_ON = False

#        elif Markets_Returns_Array_Metrics[Maximum_Market_Reference][1] == "Global Markets":
#            US_Markets_ON = False 
#            Global_Markets_ON = True
#            General_Global_Markets_ON = False
#            Developed_Markets_ON = False
#            Developed_CanadaNEurope_Markets_ON = False
#            Developed_Asia_Pacific_Markets_ON = False
#            Emerging_Markets_ON = False
#            Emerging_Asia_Markets_ON = False
#            Emerging_Others_Markets_ON = False

#        elif Markets_Returns_Array_Metrics[Maximum_Market_Reference][1] == "Global Markets - General":
#            US_Markets_ON = False 
#            Global_Markets_ON = False
#            General_Global_Markets_ON = True
#            Developed_Markets_ON == False
#            Developed_CanadaNEurope_Markets_ON = False
#            Developed_Asia_Pacific_Markets_ON = False
#            Emerging_Markets_ON = False
#            Emerging_Asia_Markets_ON = False
#            Emerging_Others_Markets_ON = False

#        elif Markets_Returns_Array_Metrics[Maximum_Market_Reference][1] == "Developed Markets - General":
#            US_Markets_ON = False 
#            Global_Markets_ON = False
#            General_Global_Markets_ON = False
#            Developed_Markets_ON = True
#            Developed_CanadaNEurope_Markets_ON = False
#            Developed_Asia_Pacific_Markets_ON = False
#            Emerging_Markets_ON = False
#            Emerging_Asia_Markets_ON = False
#            Emerging_Others_Markets_ON = False

#        elif Markets_Returns_Array_Metrics[Maximum_Market_Reference][1] == "Developed Markets - Canada & Europe":
#            US_Markets_ON = False 
#            Global_Markets_ON = False
#            General_Global_Markets_ON = False
#            Developed_Markets_ON = False
#            Developed_CanadaNEurope_Markets_ON = True
#            Developed_Asia_Pacific_Markets_ON = False
#            Emerging_Markets_ON = False
#            Emerging_Asia_Markets_ON = False
#            Emerging_Others_Markets_ON = False

#        elif Markets_Returns_Array_Metrics[Maximum_Market_Reference][1] == "Developed Markets - Asia & Pacific":
#            US_Markets_ON = False 
#            Global_Markets_ON = False
#            General_Global_Markets_ON = False
#            Developed_Markets_ON = False
#            Developed_CanadaNEurope_Markets_ON = False
#            Developed_Asia_Pacific_Markets_ON = True
#            Emerging_Markets_ON = False
#            Emerging_Asia_Markets_ON = False
#            Emerging_Others_Markets_ON = False

#        elif Markets_Returns_Array_Metrics[Maximum_Market_Reference][1] == "Emerging Markets - General":
#            US_Markets_ON = False 
#            Global_Markets_ON = False
#            General_Global_Markets_ON = False
#            Developed_Markets_ON = False
#            Developed_CanadaNEurope_Markets_ON = False
#            Developed_Asia_Pacific_Markets_ON = False
#            Emerging_Markets_ON = True
#            Emerging_Asia_Markets_ON = False
#            Emerging_Others_Markets_ON = False

#        elif Markets_Returns_Array_Metrics[Maximum_Market_Reference][1] == "Emerging Markets - Asia":
#            US_Markets_ON = False 
#            Global_Markets_ON = False
#            General_Global_Markets_ON = False
#            Developed_Markets_ON = False
#            Developed_CanadaNEurope_Markets_ON = False
#            Developed_Asia_Pacific_Markets_ON = False
#            Emerging_Markets_ON = False
#            Emerging_Asia_Markets_ON = True
#            Emerging_Others_Markets_ON = False

#        elif Markets_Returns_Array_Metrics[Maximum_Market_Reference][1] == "Emerging Markets - Others":
#            US_Markets_ON = False 
#            Global_Markets_ON = False
#            General_Global_Markets_ON = False
#            Developed_Markets_ON = False
#            Developed_CanadaNEurope_Markets_ON = False
#            Developed_Asia_Pacific_Markets_ON = False
#            Emerging_Markets_ON = False
#            Emerging_Asia_Markets_ON = False
#            Emerging_Others_Markets_ON = True


#    except ValueError:
#        US_Markets_ON = True 
#        Global_Markets_ON = False
#        General_Global_Markets_ON = False
#        Developed_Markets_ON = False
#        Developed_CanadaNEurope_Markets_ON = False
#        Developed_Asia_Pacific_Markets_ON = False
#        Emerging_Markets_ON = False
#        Emerging_Asia_Markets_ON = False
#        Emerging_Others_Markets_ON = False
            
#    for Asset in range(total_number_of_participating_asset):

#        processed_symbol = original_momentum_array[Asset][0]

#        if US_Markets_ON == True:
#            for Specific_Markets_Asset_Column in global_sub_class_assets[1]:
#                if processed_symbol == Specific_Markets_Asset_Column:
#                    original_momentum_array[Asset][1] = default_out_value

#        elif General_Global_Markets_ON == True:
#            for Specific_Markets_Asset_Column in us_sub_class_assets[1]:
#                if processed_symbol == Specific_Markets_Asset_Column:
#                    original_momentum_array[Asset][1] = default_out_value
       
#        elif General_Global_Markets_ON == True:
#            for Specific_Markets_Asset_Column in us_sub_class_assets[1]:
#                if processed_symbol == Specific_Markets_Asset_Column:
#                    original_momentum_array[Asset][1] = default_out_value
            
#            for Specific_Markets_Asset_Column in global_sub_class_assets[1]:
#                if processed_symbol == Specific_Markets_Asset_Column:
#                    Asset_Found_In_Specific_Markets = np.where(general_global_sub_class_assets[:, 1] == processed_symbol)[0]
#                    if Asset_Found_In_Specific_Markets.size == 0:
#                        original_momentum_array[Asset][1] = default_out_value

#        elif Developed_Markets_ON == True:
#            for Specific_Markets_Asset_Column in us_sub_class_assets[1]:
#                if processed_symbol == Specific_Markets_Asset_Column:
#                    original_momentum_array[Asset][1] = default_out_value
            
#            for Specific_Markets_Asset_Column in global_sub_class_assets[1]:
#                if processed_symbol == Specific_Markets_Asset_Column:
#                    Asset_Found_In_Specific_Markets = np.where(developed_sub_class_assets[:, 1] == processed_symbol)[0]
#                    if Asset_Found_In_Specific_Markets.size == 0:
#                        original_momentum_array[Asset][1] = default_out_value
    
#        elif Developed_CanadaNEurope_Markets_ON == True:
#            for Specific_Markets_Asset_Column in us_sub_class_assets[1]:
#                if processed_symbol == Specific_Markets_Asset_Column:
#                    original_momentum_array[Asset][1] = default_out_value
            
#            for Specific_Markets_Asset_Column in global_sub_class_assets[1]:
#                if processed_symbol == Specific_Markets_Asset_Column:
#                    Asset_Found_In_Specific_Markets = np.where(developed_canadaneurope_sub_class_assets[:, 1] == processed_symbol)[0]
#                    if Asset_Found_In_Specific_Markets.size == 0:
#                        original_momentum_array[Asset][1] = default_out_value
        
#        elif Developed_Asia_Pacific_Markets_ON == True:
#            for Specific_Markets_Asset_Column in us_sub_class_assets[1]:
#                if processed_symbol == Specific_Markets_Asset_Column:
#                    original_momentum_array[Asset][1] = default_out_value
            
#            for Specific_Markets_Asset_Column in global_sub_class_assets[1]:
#                if processed_symbol == Specific_Markets_Asset_Column:
#                    Asset_Found_In_Specific_Markets = np.where(developed_asia_pacific_sub_class_assets[:, 1] == processed_symbol)[0]
#                    if Asset_Found_In_Specific_Markets.size == 0:
#                        original_momentum_array[Asset][1] = default_out_value

#        elif Emerging_Markets_ON == True:
#            for Specific_Markets_Asset_Column in us_sub_class_assets[1]:
#                if processed_symbol == Specific_Markets_Asset_Column:
#                    original_momentum_array[Asset][1] = default_out_value
            
#            for Specific_Markets_Asset_Column in global_sub_class_assets[1]:
#                if processed_symbol == Specific_Markets_Asset_Column:
#                    Asset_Found_In_Specific_Markets = np.where(emerging_sub_class_assets[:, 1] == processed_symbol)[0]
#                    if Asset_Found_In_Specific_Markets.size == 0:
#                        original_momentum_array[Asset][1] = default_out_value

#        elif Emerging_Markets_ON == True:
#            for Specific_Markets_Asset_Column in us_sub_class_assets[1]:
#                if processed_symbol == Specific_Markets_Asset_Column:
#                    original_momentum_array[Asset][1] = default_out_value
            
#            for Specific_Markets_Asset_Column in global_sub_class_assets[1]:
#                if processed_symbol == Specific_Markets_Asset_Column:
#                    Asset_Found_In_Specific_Markets = np.where(emerging_sub_class_assets[:, 1] == processed_symbol)[0]
#                    if Asset_Found_In_Specific_Markets.size == 0:
#                        original_momentum_array[Asset][1] = default_out_value
  
#        elif Emerging_Asia_Markets_ON == True:
#            for Specific_Markets_Asset_Column in us_sub_class_assets[1]:
#                if processed_symbol == Specific_Markets_Asset_Column:
#                    original_momentum_array[Asset][1] = default_out_value
            
#            for Specific_Markets_Asset_Column in global_sub_class_assets[1]:
#                if processed_symbol == Specific_Markets_Asset_Column:
#                    Asset_Found_In_Specific_Markets = np.where(emerging_asia_sub_class_assets[:, 1] == processed_symbol)[0]
#                    if Asset_Found_In_Specific_Markets.size == 0:
#                        original_momentum_array[Asset][1] = default_out_value

#        elif Emerging_Others_Markets_ON == True:
#            for Specific_Markets_Asset_Column in us_sub_class_assets[1]:
#                if processed_symbol == Specific_Markets_Asset_Column:
#                    original_momentum_array[Asset][1] = default_out_value
            
#            for Specific_Markets_Asset_Column in global_sub_class_assets[1]:
#                if processed_symbol == Specific_Markets_Asset_Column:
#                    Asset_Found_In_Specific_Markets = np.where(emerging_others_sub_class_assets[:, 1] == processed_symbol)[0]
#                    if Asset_Found_In_Specific_Markets.size == 0:
#                        original_momentum_array[Asset][1] = default_out_value

#        Specific_Markets_Strategy_1_Returns_Reference = Lowest_Return_for_Markets
                          
#    return original_momentum_array, Specific_Markets_Strategy_1_Returns_Reference




            #Functions.write_to_txt_file(Emergency_Status, 'emergency_previous_Parameters.txt', 1)
            #Functions.write_to_txt_file(Total_Days_Pre_Allarm_ON_Equity, 'emergency_previous_Parameters.txt', 2)
            #Functions.write_to_txt_file(Total_Days_Pre_Allarm_ON_Bond, 'emergency_previous_Parameters.txt', 3)
            #Functions.write_to_txt_file(Total_Days_Allarm_ON, 'emergency_previous_Parameters.txt', 4)

            #try:
            #    if Functions.read_from_txt_file('emergency_previous_Parameters.txt', 1, bool)!="":
            #        Emergency_Status = Functions.read_from_txt_file('emergency_previous_Parameters.txt', 1, bool)
            #        print(f"Source Read Emergency_Status: {Emergency_Status}")
            #    else:
            #        Emergency_Status = False
            #        print(f"Emergency_Status empty ''. Assigned to False")
            #except ValueError as e:
            #    Emergency_Status = False
            #    print(f"Emergency_Status not Found. Assigned to False")

            #try:
            #    if Functions.read_from_txt_file('emergency_previous_Parameters.txt', 2, int)!= "" :
            #        Total_Days_Pre_Allarm_ON_Equity = Functions.read_from_txt_file('emergency_previous_Parameters.txt', 2, int)
            #        print(f"Source Read Total_Days_Pre_Allarm_ON_Equity: {Total_Days_Pre_Allarm_ON_Equity}")
            #    else:
            #        Total_Days_Pre_Allarm_ON_Equity = 0 
            #        print(f"Total_Days_Pre_Allarm_ON_Equity empty ''. Assigned to 0")
            #except ValueError as e:
            #    Total_Days_Pre_Allarm_ON_Equity = 0
            #    print(f"Total_Days_Pre_Allarm_ON_Equity not Found. Assigned to 0")

            #try:
            #    if Functions.read_from_txt_file('emergency_previous_Parameters.txt', 3, int)!= "":
            #        Total_Days_Pre_Allarm_ON_Bond = Functions.read_from_txt_file('emergency_previous_Parameters.txt', 3, int)
            #        print(f"Source Read Total_Days_Pre_Allarm_ON_Bond: {Total_Days_Pre_Allarm_ON_Bond}")
            #    else:
            #        Total_Days_Pre_Allarm_ON_Bond =0 
            #        print(f"Total_Days_Pre_Allarm_ON_Bond empty ''. Assigned to 0")
            #except ValueError as e:
            #    Total_Days_Pre_Allarm_ON_Bond = 0
            #    print(f"Total_Days_Pre_Allarm_ON_Bond not Found. Assigned to 0")


            #try:
            #    if Functions.read_from_txt_file('emergency_previous_Parameters.txt', 4, int) != "":
            #        Total_Days_Allarm_ON = Functions.read_from_txt_file('emergency_previous_Parameters.txt', 4, int)
            #        print(f"Source Read Total_Days_Allarm_ON: {Total_Days_Allarm_ON}")
            #    else:
            #        Total_Days_Allarm_ON =0 
            #        print(f"Total_Days_Allarm_ON empty ''. Assigned to 0")
            #except ValueError as e:
            #    Total_Days_Allarm_ON = 0
            #    print(f"Total_Days_Allarm_ON not Found. Assigned to 0")





    #    import numpy as np
    #import pandas as pd
    #import matplotlib.pyplot as plt
    #from sklearn.preprocessing import MinMaxScaler
    #from tensorflow.keras.models import Sequential
    #from tensorflow.keras.layers import LSTM, Dense, Dropout
    #from tensorflow.keras.callbacks import EarlyStopping
    #import yfinance as yf
    #import datetime

    #def load_stock_data(stock_symbol, start_date, end_date):
    #    data = yf.download(stock_symbol, start=start_date, end=end_date)
    #    data = data[['Close']]
    #    return data

    ## Parameters
    #stock_symbol = 'AAPL'
    #start_date = '2018-01-01'
    #end_date = '2023-01-01'
    #predict_days = 30

    ## Load and preprocess data
    #data = load_stock_data(stock_symbol, start_date, end_date)
    #scaler = MinMaxScaler(feature_range=(0, 1))
    #scaled_data = scaler.fit_transform(data)

    #X_train, y_train = [], []
    #lookback_period = 60

    #for i in range(lookback_period, len(scaled_data) - predict_days):
    #    X_train.append(scaled_data[i - lookback_period:i, 0])
    #    y_train.append(scaled_data[i:i + predict_days, 0])

    #X_train, y_train = np.array(X_train), np.array(y_train)
    #X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    ## Build LSTM model
    #model = Sequential()
    #model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    #model.add(Dropout(0.2))
    #model.add(LSTM(units=50, return_sequences=True))
    #model.add(Dropout(0.2))
    #model.add(LSTM(units=50))
    #model.add(Dropout(0.2))
    #model.add(Dense(units=predict_days))

    #model.compile(optimizer='adam', loss='mean_squared_error')
    #early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

    #history = model.fit(X_train, y_train, epochs=50, batch_size=32, callbacks=[early_stop])

    ## Test data preparation
    #test_data = scaled_data[-(lookback_period + predict_days):]
    #X_test = []
    #for i in range(lookback_period, len(test_data) - predict_days + 1):
    #    X_test.append(test_data[i - lookback_period:i, 0])

    #X_test = np.array(X_test)
    #X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    ## Predict and inverse scale
    #predictions = model.predict(X_test)
    #predictions = scaler.inverse_transform(predictions)

    ## Plot results
    #dates = pd.date_range(end=end_date, periods=len(data) + predict_days).to_pydatetime().tolist()
    #original_data_extended = np.append(data['Close'].values, [np.nan]*predict_days)

    #plt.figure(figsize=(14, 7))
    #plt.plot(dates, original_data_extended, color='blue', label=f"{stock_symbol} Actual Closing Price")
    #plt.plot(dates[-predict_days:], predictions[0], color='red', label="Predicted Prices")
    #plt.title(f'{stock_symbol} Closing Price Prediction')
    #plt.xlabel('Date')
    #plt.ylabel('Price')
    #plt.legend()
    #plt.show()





                ## Lists to store the results from each model
                #Random_Forest_Accuracy_Results = []
                #KNN_Accuracy_Results = []
                #Ensemble_Accuracy_Results = []

    
                ##print(rf_accuracy, knn_accuracy, ensemble_accuracy)
                #Random_Forest_Accuracy_Results.append(Random_Forest_Accuracy)
                #KNN_Accuracy_Results.append(Knn_Accuracy)
                #Ensemble_Accuracy_Results.append(Ensemble_Accuracy)

        #Training_set_step = 10 # Increment of how many starting points (len(data) / num_train  =  number of train-test sets)
        #length_Train_Test_Set = 40 # Length of each train-test set    
        #Set_number = 0
        #while True:
        
        # Partition the data into chunks of size len_train every num_train days
        #Train_Test_Set = Asset_parameters_Array.iloc[Set_number * Training_set_step : (Set_number * Training_set_step) + length_Train_Test_Set]
        #Set_number += 1
        #print(i * Training_set_step, (i * Training_set_step) + length_Train_Test_Set)
        
        #if len(Train_Test_Set) < 40:
        #    break

        #import yfinance as yf
        #import datetime
        #import pandas as pd
        #import numpy as np
        #from finta import TA

        #import matplotlib.pyplot as plt

        #from sklearn import svm
        #from sklearn.ensemble import RandomForestClassifier
        #from sklearn.neighbors import KNeighborsClassifier
        #from sklearn.ensemble import AdaBoostClassifier
        #from sklearn.ensemble import GradientBoostingClassifier
        #from sklearn.ensemble import VotingClassifier
        #from sklearn.model_selection import train_test_split, GridSearchCV
        #from sklearn.metrics import confusion_matrix, classification_report
        #from sklearn import metrics


        #Defining some constants for data mining
        #"""

        #NUM_DAYS = 10000     # The number of days of historical data to retrieve
        #INTERVAL = '1d'     # Sample rate of historical data
        #symbol = 'SPY'      # Symbol of the desired stock

        ## List of symbols for technical indicators
        #INDICATORS = ['RSI', 'MACD', 'STOCH','ADL', 'ATR', 'MOM', 'MFI', 'ROC', 'OBV', 'CCI', 'EMV', 'VORTEX']

        #"""
        #Next we pull the historical data using yfinance
        #Rename the column names because finta uses the lowercase names
        #"""

        #start = (datetime.date.today() - datetime.timedelta( NUM_DAYS ))
        #end = datetime.datetime.today()

        #data = yf.download(symbol, start=start, end=end, interval=INTERVAL)
        #data.rename(columns={"Close": 'close', "High": 'high', "Low": 'low', 'Volume': 'volume', 'Open': 'open'}, inplace=True)
        #print(data.head())

        #tmp = data.iloc[-60:]
        #tmp['close'].plot()

        #"""
        #Next we clean our data and perform feature engineering to create new technical indicator features that our
        #model can learn from
        #"""

        #def _exponential_smooth(data, alpha):
        #    """
        #    Function that exponentially smooths dataset so values are less 'rigid'
        #    :param alpha: weight factor to weight recent values more
        #    """
    
        #    return data.ewm(alpha=alpha).mean()

        #data = Functions.Data_exponential_smooth(data, 0.65)

        #tmp1 = data.iloc[-60:]
        #tmp1['close'].plot()

        #def _get_indicator_data(data):
        #    """
        #    Function that uses the finta API to calculate technical indicators used as the features
        #    :return:
        #    """

        #    for indicator in INDICATORS:
        #        ind_data = eval('TA.' + indicator + '(data)')
        #        if not isinstance(ind_data, pd.DataFrame):
        #            ind_data = ind_data.to_frame()
        #        data = data.merge(ind_data, left_index=True, right_index=True)
        #    data.rename(columns={"14 period EMV.": '14 period EMV'}, inplace=True)

        #    # Also calculate moving averages for features
        #    data['ema50'] = data['close'] / data['close'].ewm(50).mean()
        #    data['ema21'] = data['close'] / data['close'].ewm(21).mean()
        #    data['ema15'] = data['close'] / data['close'].ewm(14).mean()
        #    data['ema5'] = data['close'] / data['close'].ewm(5).mean()

        #    # Instead of using the actual volume value (which changes over time), we normalize it with a moving volume average
        #    data['normVol'] = data['volume'] / data['volume'].ewm(5).mean()

        #    # Remove columns that won't be used as features
        #    del (data['open'])
        #    del (data['high'])
        #    del (data['low'])
        #    del (data['volume'])
        #    del (data['Adj Close'])
    
        #    return data

        #data = _get_indicator_data(data)
        #print(data.columns)


        #def _produce_prediction(data, window):
        #    """
        #    Function that produces the 'truth' values
        #    At a given row, it looks 'window' rows ahead to see if the price increased (1) or decreased (0)
        #    :param window: number of days, or rows to look ahead to see what the price did
        #    """
    
        #    prediction = (data.shift(-window)['close'] >= data['close'])
        #    prediction = prediction.iloc[:-window]
        #    data['pred'] = prediction.astype(int)
    
        #    return data

        #data = _produce_prediction(data, window=15)
        #del (data['close'])
        #data = data.dropna() # Some indicators produce NaN values for the first few rows, we just remove them here
        #data.tail()


        #def _train_random_forest(X_train, y_train, X_test, y_test):

        #    """
        #    Function that uses random forest classifier to train the model
        #    :return:
        #    """
    
        #    # Create a new random forest classifier
        #    rf = RandomForestClassifier()
    
        #    # Dictionary of all values we want to test for n_estimators
        #    params_rf = {'n_estimators': [110,130,140,150,160,180,200]}
    
        #    # Use gridsearch to test all values for n_estimators
        #    rf_gs = GridSearchCV(rf, params_rf, cv=5)
    
        #    # Fit model to training data
        #    rf_gs.fit(X_train, y_train)
    
        #    # Save best model
        #    rf_best = rf_gs.best_estimator_
    
        #    # Check best n_estimators value
        #    print(rf_gs.best_params_)
    
        #    prediction = rf_best.predict(X_test)

        #    print(classification_report(y_test, prediction))
        #    print(confusion_matrix(y_test, prediction))
    
        #    return rf_best
    
        ##rf_model = _train_random_forest(X_train, y_train, X_test, y_test)



        #def _train_KNN(X_train, y_train, X_test, y_test):

        #    knn = KNeighborsClassifier()
        #    # Create a dictionary of all values we want to test for n_neighbors
        #    params_knn = {'n_neighbors': np.arange(1, 25)}
    
        #    # Use gridsearch to test all values for n_neighbors
        #    knn_gs = GridSearchCV(knn, params_knn, cv=5)
    
        #    # Fit model to training data
        #    knn_gs.fit(X_train, y_train)
    
        #    # Save best model
        #    knn_best = knn_gs.best_estimator_
     
        #    # Check best n_neigbors value
        #    print(knn_gs.best_params_)
    
        #    prediction = knn_best.predict(X_test)

        #    print(classification_report(y_test, prediction))
        #    print(confusion_matrix(y_test, prediction))
    
        #    return knn_best
    
    
        ##knn_model = _train_KNN(X_train, y_train, X_test, y_test)


        #def _ensemble_model(rf_model, knn_model, gbt_model, X_train, y_train, X_test, y_test):
    
        #    # Create a dictionary of our models
        #    estimators=[('knn', knn_model), ('rf', rf_model), ('gbt', gbt_model)]
    
        #    # Create our voting classifier, inputting our models
        #    ensemble = VotingClassifier(estimators, voting='hard')
    
        #    #fit model to training data
        #    ensemble.fit(X_train, y_train)
    
        #    #test our model on the test data
        #    print(ensemble.score(X_test, y_test))
    
        #    prediction = ensemble.predict(X_test)

        #    print(classification_report(y_test, prediction))
        #    print(confusion_matrix(y_test, prediction))
    
        #    return ensemble
    
        ##ensemble_model = _ensemble_model(rf_model, knn_model, gbt_model, X_train, y_train, X_test, y_test)


        #def cross_Validation(data):

        #    # Split data into equal partitions of size len_train
    
        #    num_train = 10 # Increment of how many starting points (len(data) / num_train  =  number of train-test sets)
        #    len_train = 40 # Length of each train-test set
    
        #    # Lists to store the results from each model
        #    rf_RESULTS = []
        #    knn_RESULTS = []
        #    ensemble_RESULTS = []
    
        #    i = 0
        #    while True:
        
        #        # Partition the data into chunks of size len_train every num_train days
        #        df = data.iloc[i * num_train : (i * num_train) + len_train]
        #        i += 1
        #        print(i * num_train, (i * num_train) + len_train)
        
        #        if len(df) < 40:
        #            break
        
        #        y = df['pred']
        #        features = [x for x in df.columns if x not in ['pred']]
        #        X = df[features]

        #        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size= 7 * len(X) // 10,shuffle=False)
        
        #        rf_model = _train_random_forest(X_train, y_train, X_test, y_test)
        #        knn_model = _train_KNN(X_train, y_train, X_test, y_test)
        #        ensemble_model = _ensemble_model(rf_model, knn_model, X_train, y_train, X_test, y_test)
        
        #        rf_prediction = rf_model.predict(X_test)
        #        knn_prediction = knn_model.predict(X_test)
        #        ensemble_prediction = ensemble_model.predict(X_test)
        
        #        print('rf prediction is ', rf_prediction)
        #        print('knn prediction is ', knn_prediction)
        #        print('ensemble prediction is ', ensemble_prediction)
        #        print('truth values are ', y_test.values)
        
        #        rf_accuracy = accuracy_score(y_test.values, rf_prediction)
        #        knn_accuracy = accuracy_score(y_test.values, knn_prediction)
        #        ensemble_accuracy = accuracy_score(y_test.values, ensemble_prediction)
        
        #        print(rf_accuracy, knn_accuracy, ensemble_accuracy)
        #        rf_RESULTS.append(rf_accuracy)
        #        knn_RESULTS.append(knn_accuracy)
        #        ensemble_RESULTS.append(ensemble_accuracy)
        
        
        #    print('RF Accuracy = ' + str( sum(rf_RESULTS) / len(rf_RESULTS)))
        #    print('KNN Accuracy = ' + str( sum(knn_RESULTS) / len(knn_RESULTS)))
        #    print('Ensemble Accuracy = ' + str( sum(ensemble_RESULTS) / len(ensemble_RESULTS)))
    
    
        #cross_Validation(data)











        # Wavelet Transformation




        #import yfinance as yf
        #import pywt
        #import numpy as np
        #import matplotlib.pyplot as plt
        #import copy
        #import pandas as pd
        ##define the ticker symbol
        ##tickerSymbol = 'MSFT'
        ###get data on this ticker
        ##tickerData = yf.Ticker(tickerSymbol)
        ###get the historical prices for this ticker
        ##tickerDf = tickerData.history(period='1d', start='2010-1-1', end='2020-1-25')
        ##composite_signal =  tickerDf['Close'].values
        ##composite_signal

        ##def filter_bank(index_list, wavefunc='db4', lv=4, m=1, n=4, plot=False):
        ##    '''
        #    WT: Wavelet Transformation Function
        #    index_list: Input Sequence;
   
        #    lv: Decomposing Level；
 
        #    wavefunc: Function of Wavelet, 'db4' default；
    
        #    m, n: Level of Threshold Processing
   
        ##    '''
   
        #    # Decomposing :
        #    coeff = pywt.wavedec(index_list,wavefunc,mode='sym',level=lv)   #  Decomposing by levels，cD is the details coefficient
        #    sgn = lambda x: 1 if x > 0 else -1 if x < 0 else 0 # sgn function
        #    # Denoising:
        #    # Soft Threshold Processing Method
        #    for i in range(m,n+1):   #  Select m~n Levels of the wavelet coefficients，and no need to dispose the cA coefficients(approximation coefficients)
        #        cD = coeff[i]
        #        Tr = np.sqrt(2*np.log2(len(cD)))  # Compute Threshold
        #        for j in range(len(cD)):
        #            if cD[j] >= Tr:
        #                coeff[i][j] = sgn(cD[j]) * (np.abs(cD[j]) -  Tr)  # Shrink to zero
        #            else:
        #                coeff[i][j] = 0   # Set to zero if smaller than threshold
        ## Reconstructing:
        #    coeffs = {}
        #    for i in range(len(coeff)):
        #        coeffs[i] = copy.deepcopy(coeff)
        #        for j in range(len(coeff)):
        #            if j != i:
        #                coeffs[i][j] = np.zeros_like(coeff[j])
    
        #    for i in range(len(coeff)):
        #        coeff[i] = pywt.waverec(coeffs[i], wavefunc)
        #        if len(coeff[i]) > len(index_list):
        #            coeff[i] = coeff[i][:-1]
        
        #    if plot:     
        #        denoised_index = np.sum(coeff, axis=0)   
        #        data = pd.DataFrame({'CLOSE': index_list, 'denoised': denoised_index})
        #        data.plot(figsize=(10,10),subplots=(2,1))
        #        data.plot(figsize=(10,5))
   
        #    return coeff

        #coeff=filter_bank(composite_signal,plot=True)
        #fig, ax =  plt.subplots(len(coeff), 1, figsize=(10, 20))

        #for i in range(len(coeff)):
        #    if i == 0:
        #        ax[i].plot(coeff[i], label = 'cA[%.0f]'%(len(coeff)-i-1))
        #        ax[i].legend(loc = 'best')
        #    else:
        #        ax[i].plot(coeff[i], label = 'cD[%.0f]'%(len(coeff)-i))
        #        ax[i].legend(loc = 'best')




















        #import pandas as pd
        #import numpy as np
        #import pywt

        ## Load the data
        #prices = pd.read_csv('prices.csv', index_col='Date')

        ## Calculate the wavelet transform
        #coeffs = pywt.swt(prices, wavelet='sym12', level=1, trim_approx=True, norm=True)

        ## Get the long-run component
        #long_run = coeffs[0]

        ## Calculate the cumulative returns
        #cumret = (1 + long_run).cumprod()

        ## Define the trading rules
        #def buy(cumret):
        #  return cumret > 2 * cumret.std()

        #def sell(cumret):
        #  return cumret < -2 * cumret.std()

        ## Backtest the strategy
        #positions = []
        #for i in range(len(cumret) - 1):
        #  if buy(cumret[i]):
        #    positions.append(1)
        #  elif sell(cumret[i]):
        #    positions.append(-1)
        #  else:
        #    positions.append(0)

        ## Calculate the returns
        #returns = np.cumprod(positions + 1)

        ## Print the results
        #print('Cumulative returns:', returns[-1])
        #print('Sharpe ratio:', np.mean(returns) / np.std(returns))



































    

                #Currnet_date_price =  Adj_Close_prices_Array[Date_Row][Asset_Col]
                #LookBack_date_Price = Adj_Close_prices_Array[Date_Row - Lookback_Period][Asset_Col]

                #Asset_Inception_Date_Row = 0

                #if np.isnan(Currnet_date_price) ==  False: 
                #    Asset_Inception_Date_Row= Adj_Close_prices_Array.index(next(filter(lambda x: not np.isnan(x[:][Asset_Col]), Adj_Close_prices_Array)))
                #    if Date_Row - int(Asset_Inception_Date_Row)>10:
                
                #        Assets_Prices_Range=(list(map(itemgetter(Asset_Col), Adj_Close_prices_Array))[Asset_Inception_Date_Row:])
                #        Assets_Prices_Dataframe = pd.DataFrame(Assets_Prices_Range)
                #        Asset_Volatility_Array = (Assets_Prices_Dataframe.pct_change())
                #        Asset_Volatility = (Asset_Volatility_Array.std().values.tolist())[0] * math.sqrt(252)

                #    else:
                #        Asset_Volatility = 0
                #else:
                #    Asset_Volatility = 0
            
                #if np.isnan(LookBack_date_Price) == False and np.isnan(Currnet_date_price) == False:

                    #if First_Month == False:
                    #    Previous_date_price = Adj_Close_prices_Array[Previous_price_Row][Asset_Col]
                    #else:
                    #    Previous_date_price = Adj_Close_prices_Array[Date_Row][Asset_Col]

                    #High_Lookback_Period_Array =pd.Series(list(map(itemgetter(Asset_Col), High_prices_Array))[Date_Row- Lookback_Period:Date_Row])
                    #Low_Lookback_Period_Array = pd.Series(list(map(itemgetter(Asset_Col), Low_prices_Array))[Date_Row- Lookback_Period:Date_Row])
                    #Open_Lookback_Period_Array = pd.Series(list(map(itemgetter(Asset_Col), Open_prices_Array))[Date_Row- Lookback_Period:Date_Row]) 
                    #Close_Lookback_Period_Array = pd.Series(list(map(itemgetter(Asset_Col), Close_prices_Array))[Date_Row- Lookback_Period:Date_Row])
                    #Adj_Close_Lookback_Period_Array = pd.Series(list(map(itemgetter(Asset_Col), Adj_Close_prices_Array))[Date_Row- Lookback_Period:Date_Row])
                    #Volume_Lookback_Period_Array = pd.Series(list(map(itemgetter(Asset_Col), Volume_Array))[Date_Row- Lookback_Period:Date_Row])
                    #Adj_Close_Lookback_Period_Previous_Array = pd.Series(list(map(itemgetter(Asset_Col), Adj_Close_prices_Array))[Date_Row- Lookback_Period:Date_Row-1])
                
                    #Asset_Total_Return_RS=  ((Currnet_date_price/LookBack_date_Price)-1)*100
                    #MACD_Metric = (((ta_trend.MACD(Adj_Close_Lookback_Period_Array,26,12,9,False)).macd().tolist())[-1] / ((ta_trend.MACD(Adj_Close_Lookback_Period_Array,26,12,9,False)).macd_signal().tolist())[-1]) if (((ta_trend.MACD(Adj_Close_Lookback_Period_Array,26,12,9,False)).macd_signal().tolist())[-1] != 0) else 0
                    #Commodity_Channel_Index_CCI_Metric =  ((ta_trend.CCIIndicator(High_Lookback_Period_Array, Low_Lookback_Period_Array, Close_Lookback_Period_Array, 20, 0.015,False)).cci().tolist())[-1]
                    #RSI_Metric = (((ta_mum.RSIIndicator(Adj_Close_Lookback_Period_Array,14,False))).rsi().tolist())[-1]
                    #Stochastic_Oscillator_Metric = ((ta_mum.StochasticOscillator(High_Lookback_Period_Array, Low_Lookback_Period_Array, Close_Lookback_Period_Array, 14, 3, False)).stoch().tolist())[-1]
                    #Percentage_Price_Oscillator_Metric = ((ta_mum.PercentagePriceOscillator(Adj_Close_Lookback_Period_Array, 26, 12, 9, False)).ppo().tolist())[-1]
                    #ROC_Metric = (((ta_mum.ROCIndicator(Adj_Close_Lookback_Period_Array,14,False))).roc().tolist())[-1]
                    #AwesomeOscillator_Metric = (((ta_mum.AwesomeOscillatorIndicator(High_Lookback_Period_Array,Low_Lookback_Period_Array,5,34,False))).awesome_oscillator().tolist())[-1]
                    #Percentage_Volume_Oscillator_Metric = (((ta_mum.PercentageVolumeOscillator(Volume_Lookback_Period_Array,26,12,9,False))).pvo().tolist())[-1] 
                    #Trend_ADX_Metric = (((ta_trend.ADXIndicator(High_Lookback_Period_Array,Low_Lookback_Period_Array,Close_Lookback_Period_Array,14,False))).adx().tolist())[-1]
                    #Aroon_Metric = (((ta_trend.AroonIndicator(Adj_Close_Lookback_Period_Array,25,False))).aroon_indicator().tolist())[-1]
                    #Stoch_RSI_Metric = (((ta_mum.StochRSIIndicator(Adj_Close_Lookback_Period_Array,14,1,1,False))).stochrsi_d().tolist())[-1]
                    #TSI_Metric = (((ta_mum.TSIIndicator(Adj_Close_Lookback_Period_Array,25,13,False))).tsi().tolist())[-1]
                    #Kaufman_Adaptive_Moving_Average_Metric = ((ta_mum.kama(Adj_Close_Lookback_Period_Array, 10, 2, 30, False)).tolist())[-1]
                    #Vortex_Metric = ((ta_trend.VortexIndicator(High_Lookback_Period_Array, Low_Lookback_Period_Array, Close_Lookback_Period_Array, 14, False)).vortex_indicator_diff().tolist())[-1]
                    #SMA_Instantaneous_Metric = (((((ta_trend.sma_indicator(Adj_Close_Lookback_Period_Array, 14, False)).tolist())[-1])/(((ta_trend.sma_indicator(Adj_Close_Lookback_Period_Previous_Array, 14, False)).tolist())[-1]))-1)

                    #AverageTrueRange_Metric = ((ta_Volatility.AverageTrueRange(High_Lookback_Period_Array,Low_Lookback_Period_Array,Close_Lookback_Period_Array,14,False)).average_true_range().tolist())[-1]
                    #ChaikinMoneyFlow_Metric = (((ta_Vol.ChaikinMoneyFlowIndicator(High_Lookback_Period_Array,Low_Lookback_Period_Array,Close_Lookback_Period_Array,Volume_Lookback_Period_Array,14,False))).chaikin_money_flow().tolist())[-1]
                    #KST_Metric = ((ta_trend.KSTIndicator(Adj_Close_Lookback_Period_Array,10, 15, 20, 30, 10, 10, 10, 15, 9, False)).kst().tolist())[-1]
                    #BollingerBands_Metric = (Adj_Close_Lookback_Period_Array.tolist())[-1]/(((ta_Volatility.BollingerBands(Adj_Close_Lookback_Period_Array,20,2,False)).bollinger_hband().tolist())[-1])
                    #Donchian_channel_Metric = (Adj_Close_Lookback_Period_Array.tolist())[-1]/(((ta_Volatility.donchian_channel_hband(High_Lookback_Period_Array, Low_Lookback_Period_Array, Close_Lookback_Period_Array,20,0,False)).tolist())[-1])
                    #keltner_channel_Metric = (Adj_Close_Lookback_Period_Array.tolist())[-1]/(((ta_Volatility.KeltnerChannel(High_Lookback_Period_Array, Low_Lookback_Period_Array, Close_Lookback_Period_Array, 14, 10, False, True, 2)).keltner_channel_hband().tolist())[-1])
                            
                    ##ta_ut.math 
                            
                    #if Selected_method_of_Momuntum_Metric == "Metrics Combind": 
                    #    Metric_Value =   Asset_Total_Return_RS * abs(Percentage_Price_Oscillator_Metric)  * abs(Percentage_Volume_Oscillator_Metric) * abs(Vortex_Metric) * abs(TSI_Metric) * abs(SMA_Instantaneous_Metric) * abs(KST_Metric) * abs(keltner_channel_Metric) * abs(Donchian_channel_Metric) #           * abs(BollingerBands_Metric) 
                    #    #Metric_Value = Asset_Total_Return_RS* abs(MACD_Metric)* abs(Commodity_Channel_Index_CCI_Metric)* abs(RSI_Metric)* abs(Stochastic_Oscillator_Metric)* abs(Percentage_Price_Oscillator_Metric)* abs(Vortex_Metric)
                    #    #Metric_Value =  (KST_Metric) #abs(Stoch_RSI_Metric)* Asset_Total_Return_RS * abs(ROC_Metric)* abs(AwesomeOscillator_Metric) * abs(Percentage_Volume_Oscillator_Metric) * abs(TSI_Metric)  * abs(ChaikinMoneyFlow_Metric) * abs(Trend_ADX_Metric) * abs(Aroon_Metric) * abs(MACD_Metric)* abs(AverageTrueRange_Metric)
                    #    Assets_parameters_Array = [Asset_Symbols[Asset_Col], Metric_Value, Volume_Array[Date_Row][Asset_Col], Asset_Volatility, Asset_Col, Previous_date_price, Current_Date]
                    #    List_of_Ranked_Assets.append(Assets_parameters_Array)

                    #    #print(f'Asset Symbol: {List_of_Ranked_Assets[-1][0]} ---- Metric Value = {List_of_Ranked_Assets[-1][1]}', sep='\n')

                    #elif Selected_method_of_Momuntum_Metric == "Metrics Count":
                    #    Combined_Metric_Array = [Asset_Total_Return_RS, Percentage_Price_Oscillator_Metric , Percentage_Volume_Oscillator_Metric, Vortex_Metric, TSI_Metric , SMA_Instantaneous_Metric, KST_Metric, keltner_channel_Metric , Donchian_channel_Metric] # , ChaikinMoneyFlow_Metric,  BollingerBands_Metric, Donchian_channel_Metric, keltner_channel_Metric, KST_Metric]
                    #    Assets_parameters_Array = [Asset_Symbols[Asset_Col], Metric_Value, Volume_Array[Date_Row][Asset_Col], Asset_Volatility, Asset_Col, Previous_date_price, Current_Date]
                    #    Assets_parameters_Array.extend(Combined_Metric_Array) 
                    #    List_of_Ranked_Assets.append(Assets_parameters_Array)


























        #from sklearn.preprocessing import MinMaxScaler
        #from keras.layers import LSTM, Dense, Dropout
        #from sklearn.model_selection import TimeSeriesSplit
        #from sklearn.metrics import mean_squared_error, r2_score
        #import matplotlib.dates as mandates
        #from sklearn.preprocessing import MinMaxScaler
        #from sklearn import linear_model
        #from keras.models import Sequential
        #from keras.layers import Dense
        #import keras.backend as K
        #from keras.callbacks import EarlyStopping
        #from keras.optimizers import Adam
        #from keras.models import load_model
        #from keras.layers import LSTM
        #from keras.utils.vis_utils import plot_model

        ##Get the Dataset
        #df=pd.read_csv(“MicrosoftStockData.csv”,na_values=[‘null’],index_col=’Date’,parse_dates=True,infer_datetime_format=True)
        #df.head()

        ##Print the shape of Dataframe  and Check for Null Values
        #print(“Dataframe Shape: “, df. shape)
        #print(“Null Value Present: “, df.IsNull().values.any())
        #Output:
        #>> Dataframe Shape: (7334, 6)
        ##>>Null Value Present: False

        ##Plot the True Adj Close Value
        #df[‘Adj Close’].plot()

        ##Set Target Variable
        #output_var = PD.DataFrame(df[‘Adj Close’])
        ##Selecting the Features
        #features = [‘Open’, ‘High’, ‘Low’, ‘Volume’]

        ##Scaling
        #scaler = MinMaxScaler()
        #feature_transform = scaler.fit_transform(df[features])
        #feature_transform= pd.DataFrame(columns=features, data=feature_transform, index=df.index)
        #feature_transform.head()

        ##Splitting to Training set and Test set
        #timesplit= TimeSeriesSplit(n_splits=10)
        #for train_index, test_index in timesplit.split(feature_transform):
        #        X_train, X_test = feature_transform[:len(train_index)], feature_transform[len(train_index): (len(train_index)+len(test_index))]
        #        y_train, y_test = output_var[:len(train_index)].values.ravel(), output_var[len(train_index): (len(train_index)+len(test_index))].values.ravel()

        ##Process the data for LSTM
        #trainX =np.array(X_train)
        #testX =np.array(X_test)

        #X_train = trainX.reshape(X_train.shape[0], 1, X_train.shape[1])
        #X_test = testX.reshape(X_test.shape[0], 1, X_test.shape[1])

        ##Building the LSTM Model
        #lstm = Sequential()
        #lstm.add(LSTM(32, input_shape=(1, trainX.shape[1]), activation=’relu’, return_sequences=False))
        #lstm.add(Dense(1))
        #lstm.compile(loss=’mean_squared_error’, optimizer=’adam’)
        #plot_model(lstm, show_shapes=True, show_layer_names=True)

        ##Model Training
        #history=lstm.fit(X_train, y_train, epochs=100, batch_size=8, verbose=1, shuffle=False)

        ##LSTM Prediction
        #y_pred= lstm.predict(X_test)

        ##Predicted vs True Adj Close Value – LSTM
        #plt.plot(y_test, label=’True Value’)
        #plt.plot(y_pred, label=’LSTM Value’)
        #plt.title(“Prediction by LSTM”)
        #plt.xlabel(‘Time Scale’)
        #plt.ylabel(‘Scaled USD’)
        #plt.legend()
        #plt.show()


































        #import seaborn as sb
 
        #from sklearn.model_selection import train_test_split
        #from sklearn.preprocessing import StandardScaler
        #from sklearn.linear_model import LogisticRegression
        #from sklearn.svm import SVC
        #from xgboost import XGBClassifier
        #from sklearn import metrics
 
        #import warnings
        #warnings.filterwarnings('ignore')


        #df = pd.read_csv('/content/Tesla.csv')
        #df.head()
        #df.shape
        #df.describe()
        #df.info()
        #plt.figure(figsize=(15,5))
        #plt.plot(df['Close'])
        #plt.title('Tesla Close price.', fontsize=15)
        #plt.ylabel('Price in dollars.')
        #plt.show()
        #df.isnull().sum()

        #features = ['Open', 'High', 'Low', 'Close', 'Volume']

        #plt.subplots(figsize=(20,10))

        #for i, col in enumerate(features):
        #    plt.subplot(2,3,i+1)
        #    sb.distplot(df[col])
        #    plt.show()

        #plt.subplots(figsize=(20,10))

        #for i, col in enumerate(features):
        #    plt.subplot(2,3,i+1)

        #sb.boxplot(df[col])
        #plt.show()


        #splitted = df['Date'].str.split('/', expand=True)

        #df['day'] = splitted[1].astype('int')
        #df['month'] = splitted[0].astype('int')
        #df['year'] = splitted[2].astype('int')

        #df.head()

        #df['is_quarter_end'] = np.where(df['month']%3==0,1,0)
        #df.head()

        #data_grouped = df.groupby('year').mean()
        #plt.subplots(figsize=(20,10))

        #for i, col in enumerate(['Open', 'High', 'Low', 'Close']):
        #    plt.subplot(2,2,i+1)
        #    data_grouped[col].plot.bar()
        #    plt.show()


        #df['open-close'] = df['Open'] - df['Close']
        #df['low-high'] = df['Low'] - df['High']
        #df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

        #plt.pie(df['target'].value_counts().values,labels=[0, 1], autopct='%1.1f%%')
        #plt.show()

        #plt.figure(figsize=(10, 10))

        ## As our concern is with the highly
        ## correlated features only so, we will visualize
        ## our heatmap as per that criteria only.

        #sb.heatmap(df.corr() > 0.9, annot=True, cbar=False)
        #plt.show()


        #features = df[['open-close', 'low-high', 'is_quarter_end']]
        #target = df['target']

        #scaler = StandardScaler()
        #features = scaler.fit_transform(features)

        #X_train, X_valid, Y_train, Y_valid = train_test_split(features, target, test_size=0.1, random_state=2022)
        #print(X_train.shape, X_valid.shape)



        #models = [LogisticRegression(), SVC(kernel='poly', probability=True), XGBClassifier()]

        #for i in range(3):
        #    models[i].fit(X_train, Y_train)

        #print(f'{models[i]} : ')
        #print('Training Accuracy : ', metrics.roc_auc_score(Y_train, models[i].predict_proba(X_train)[:,1]))
        #print('Validation Accuracy : ', metrics.roc_auc_score(Y_valid, models[i].predict_proba(X_valid)[:,1]))

        #print()


        #metrics.plot_confusion_matrix(models[0], X_valid, Y_valid)
        #plt.show()































        #import numpy as np
        #import pandas as pd
        #import matplotlib.pyplot as plt
        #from sklearn.preprocessing import MinMaxScaler
        #from keras.models import Sequential
        #from keras.layers import Dense, LSTM, Dropout

        ## Load dataset
        #data = pd.read_csv('stock_prices.csv')  # Replace with your own dataset
        #prices = data['Close'].values
        #prices = prices.reshape(-1, 1)

        ## Normalize data
        #scaler = MinMaxScaler(feature_range=(0, 1))
        #prices = scaler.fit_transform(prices)

        ## Prepare training and test data
        #train_size = int(len(prices) * 0.8)
        #train_data = prices[:train_size]
        #test_data = prices[train_size:]

        ## Function to create input-output sequence
        #def create_sequence(data, seq_length):
        #    X, y = [], []
        #    for i in range(len(data) - seq_length):
        #        X.append(data[i:(i + seq_length)])
        #        y.append(data[i + seq_length])
        #    return np.array(X), np.array(y)

        ## Create sequences
        #seq_length = 60
        #X_train, y_train = create_sequence(train_data, seq_length)
        #X_test, y_test = create_sequence(test_data, seq_length)

        ## Build LSTM model
        #model = Sequential()
        #model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        #model.add(Dropout(0.2))
        #model.add(LSTM(50, return_sequences=True))
        #model.add(Dropout(0.2))
        #model.add(LSTM(50))
        #model.add(Dropout(0.2))
        #model.add(Dense(1))

        #model.compile(optimizer='adam', loss='mean_squared_error')

        ## Train the model
        #model.fit(X_train, y_train, epochs=100, batch_size=32)

        ## Make predictions
        #predicted_prices = model.predict(X_test)
        #predicted_prices = scaler.inverse_transform(predicted_prices)

        ## Plot the results
        #plt.figure(figsize=(10, 5))
        #plt.plot(y_test, color='blue', label='Actual Stock Price')
        #plt.plot(predicted_prices, color='red', label='Predicted Stock Price')
        #plt.title('Stock Price Prediction')
        #plt.xlabel('Time')
        #plt.ylabel('Stock Price')
        #plt.legend()
        #plt.show()
