import Strategy_Central
import Functions as Functions
from Test_Data_Event import event

def run(params):

    for key, value in event.items():
        if key not in params:
            params[key] = value
            print("Input params do not have this key. the key ( " + key + " ) is added to Input params")
    
    output, Emergnecy_Parameters = Strategy_Central.Portfolio_Simulator(params)

    if len(output) > 1:
        trading_days = [{
            "date": allocation_Date[0][0],
            "positions": [
                {
                    "symbol": allocation[0],
                    "weight": round(allocation[1], 3)
                } for allocation in allocation_Date[0][1]
            ]
        } for allocation_Date in output]

    elif len(output) == 1:
        trading_days = [{
            "date": allocation_Date[0],
            "positions": [
                {
                    "symbol": allocation[0],
                    "weight": round(allocation[1], 3)
                } for allocation in allocation_Date[1]
            ]
        } for allocation_Date in output]

    else:
        trading_days = []
        print("Output is empty.")

    result = {
        "tradingDays": trading_days,
        "state": Emergnecy_Parameters
    }

    return result


    #parmas format :
#{
#  “sinceDate”: “2023-04-01”,
#  “untilDate”: “2023-05-31”,
#  “assets”: [
#    {
#      “symbol”: “AAPL”,
#      “data”: [
#        {
#          "date": "1980-12-12",
#          "adjustedClose": 1.4953,
#          "close": 1.4953,
#          "high": 1.5399,
#          "low": 1.48,
#          "open": 1.5,
#          "volume": 11251
#        }
#      ]  
#    }
#  ]
#  “indicators”: [
#    {
#      “symbol”: “US_INFLATION”,
#      “data”: [
#        {
#          "date": "1980-12-12",
#          "value": 11251
#        }
#      ]  
#    }
#  ],
#  “state”: “your-last-state-here”
#}



# output format:
#{
#  "tradingDays": [
#    {
#      "date": "2023-04-28",
#      "positions": [
#        { "symbol": "AAPL", "weight": 0.7 },
#        { "symbol": "MSFT", "weight": 0.2 },
#        { "symbol": "TSLA", "weight": 0.1 }
#      ]
#    },
#    {
#      "date": "2023-05-28",
#      "positions": [
#        { "symbol": "COKE", "weight": 0.5 },
#        { "symbol": "SH", "weight": 0.3 },
#        { "symbol": "TSLA", "weight": 0.2 }
#      ]
#    }
#  ],
#  "state": "your-new-state-here"
#}

