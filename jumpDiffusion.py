# Import modules
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime as dt
import datetime as dt1
import pandas_market_calendars as mcal


# Uses parameter data to return historical drift and volatility coefficients for GBM
def calculateJumpParameters(parameterData, jumpThreshold):
    
    # Stores the daily log returns
    logReturns=(np.log(parameterData['Adj Close'].pct_change()+1))

    mu=logReturns.mean()
    sigma=logReturns.std()

    # A list of the values that are considered jumps as per the jumpLevel
    jumpList = [value for value in logReturns if np.abs(value) > np.log(jumpThreshold + 1)]

    # JD parameters
    mu_j = np.mean(jumpList)

    sigma_j = np.std(jumpList)
    
    lambda_r = len(jumpList) / len(parameterData)

    return mu, sigma, mu_j, sigma_j, lambda_r

# Conducts the Monte-Carlo Simulation 
def jumpSimulations(mu, sigma, M, forecastData, T, mu_j, sigma_j, lambda_r):
    
    # Sets initial stock adjusted price 
    S0=forecastData['Adj Close'].iloc[0]
    
    # Number of Time Steps (Assumed to be a daily time step)
    n = T

    # Calculates each time step
    dt = T/n

    St=np.zeros((n+1, M))
    St[0,:] = S0

    # GBM model with normally distributed Wiener Process. Simulation is done using Arrays, which is more efficient than using a loop. 
    jumpRV = np.multiply(np.random.poisson(lambda_r*dt, size=(n,M)), np.random.normal(mu_j, sigma_j, size=(n,M))).cumsum(axis=0)
    gbmRV = np.cumsum(((mu - lambda_r*(mu_j  + sigma_j**2*0.5))*dt + sigma*np.sqrt(dt) * np.random.normal(size=(n,M))), axis=0)
    St = S0 * np.exp (gbmRV + jumpRV)
    St = np.vstack([np.ones(M)*S0, St])

    return n, T, M, S0, St