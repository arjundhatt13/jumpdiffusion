# Import modules
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime as dt
import datetime as dt1
import pandas_market_calendars as mcal
import astropy as ast
from astropy.modeling import models, fitting


# Uses yfinance to return relevant stock data
def getData(ticker, parameterStart, T, model):

    # Grabs parameter estimation data
    parameterEnd=str(dt.today()+dt1.timedelta(days=1))[0:10]
    parameterData = yf.download(ticker,parameterStart, parameterEnd)
    
    # Sets forecast from last day of parameterData to the specified amount of forecasted days
    forecastStart=(str(parameterData.index[-1]))[0:10]
    
    forecastEnd=str(dt.strptime(forecastStart,"%Y-%m-%d")+dt1.timedelta(days=T))[0:10]
    
    # Grabs forecast data (if applicable for backtesting)
    forecastData = yf.download(ticker,forecastStart,forecastEnd)
    
    return parameterData, forecastData
