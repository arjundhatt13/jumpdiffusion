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

# Uses parameter data to return historical drift and volatility coefficients for GBM
def calculateGBMParameters(parameterData):
    
    # Stores the daily log returns
    logReturns=(np.log(parameterData['Adj Close'].pct_change()+1))

    mu=logReturns.mean()
    sigma=logReturns.std()

    return mu, sigma

# Conducts the Monte-Carlo Simulation 
def gbmSimulation(mu, sigma, M, forecastData, T):
    
    # Sets initial stock adjusted price 
    S0=forecastData['Adj Close'].iloc[0]
    
    # Number of Time Steps (Assumed to be a daily time step)
    n = T 

    # Calculates each time step
    dt = T/n

    # GBM model with normally distributed Wiener Process. Simulation is done using Arrays, which is more efficient than using a loop. 
    St = np.exp((mu* dt) + sigma * np.random.normal(0, np.sqrt(dt), size=(M,n)).T)
    St = np.vstack([np.ones(M), St])


    # This will calculate the cumulative return at each time step, for each simulation. Multiplying through by S0 will change the St matrix from returns to stock prices.
    St = St.cumprod(axis=0)
    St = S0*St

    return n, T, M, S0, St

