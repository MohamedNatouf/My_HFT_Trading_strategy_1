
import os
import tempfile
import requests
import concurrent.futures
from datetime import datetime
os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()
import strategy
import Functions as Functions

def run(event, context):
  #since_date = datetime.fromisoformat(event["sinceDate"])
  #until_date = datetime.fromisoformat(event["untilDate"] if "untilDate" in event else datetime.now().format("%Y-%m-%d"))

  since_date = event["sinceDate"]
  until_date = (event["untilDate"] if "untilDate" in event else datetime.now().format("%Y-%m-%d"))

  print(f"Calculating allocations from {since_date} until {until_date}")

  if event['Data_Source'] == "Yahoo_Finanace":
      assets =Functions.fetch_asset_data(event['assets'],event['Data_Source'])
      indicators = get_indicators(["US_TREASURY_YIELD_3M", "US_INTEREST_RATE", "US_CPI", "US_INFLATION"])
  if event['Data_Source'] == "AlphaVanatage":
      assets =Functions.fetch_asset_data(event['assets'], event['Data_Source'])
      indicators = get_indicators(["US_TREASURY_YIELD_3M", "US_INTEREST_RATE", "US_CPI", "US_INFLATION"])
  elif event['Data_Source'] == "AlgoMart":
      assets = get_assets(event['assets'])
      indicators = get_indicators(["US_TREASURY_YIELD_3M", "US_INTEREST_RATE", "US_CPI", "US_INFLATION"])
  elif event['Data_Source'] == "StoredData":
      assets = []
      indicators = []

  result = strategy.run({
    "state": event['state'],
    "sinceDate": since_date,
    "untilDate": until_date,
    "assets": assets,
    "indicators": indicators,
  })

  if result['state'] is not None and (not isinstance(result['state'], str) or len(result['state']) > 1000):
    raise Exception("State object is too large, must be less than 1000 characters")

  return result

def get_assets(symbols):
  assets = []

  with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = [executor.submit(fetch_asset_data, symbol) for symbol in symbols]

    for future in concurrent.futures.as_completed(futures):
      asset_data = future.result()
      if asset_data:
        assets.append(asset_data)

  return assets

def get_indicators(symbols):
  indicators = []

  with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = [executor.submit(fetch_indicator_data, symbol) for symbol in symbols]

    for future in concurrent.futures.as_completed(futures):
      indicator_data = future.result()
      if indicator_data:
        indicators.append(indicator_data)

  return indicators

def fetch_asset_data(symbol):
  try:
    print(f"Fetching asset data for {symbol}")
    response = requests.get(f"{os.environ['ASSET_API_URL']}/assets/{symbol}/data?apiKey={os.environ['ASSET_API_KEY']}")
    if response.status_code != 200:
      raise Exception(f"Server responded with {response.status_code} - {response.json()}")
    return { "symbol": symbol, "data": response.json() }
  except Exception as e:
    print(f"Failed to fetch asset data for {symbol}. Error: {e}")
    raise e

def fetch_indicator_data(symbol):
  try:
    print(f"Fetching indicator data for {symbol}")
    response = requests.get(f"{os.environ['ASSET_API_URL']}/indicators/{symbol}/data?apiKey={os.environ['ASSET_API_KEY']}")
    if response.status_code != 200:
      raise Exception(f"Server responded with {response.status_code} - {response.json()}")
    return { "symbol": symbol, "data": response.json() }
  except Exception as e:
    print(f"Failed to fetch indicator data for {symbol}. Error: {e}")
    raise e