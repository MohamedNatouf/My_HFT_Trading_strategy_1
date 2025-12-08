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
import Model_Input_Handler as Model_Input
import Parameters_Input_Handler as Parameters_Input
import Dates_Handler as Dates_Handler
import Assets_Weightaging_Handler as Assets_Weightage
import Assets_Selection_Handler as Assets_Selection
import Output_Handler as Output_Generator



def Active_Trades_Dates_Generaters(Request_Type, Rebalancing_Method,  Date_Row, Date_Array, Start_Row, End_Row, Start_Date, End_Date):


    print("Yes")

    return Rebalancing_Method
