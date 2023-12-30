from jumpDiffusion import *
from GBM import *
from plotting import *
from historicDataScraper import *
from riskMetrics import *

# User Inputs
ticker = "AAPL"                      # Stock Ticker
T = 252                              # Forecast Horizon (Trading Days)
M = 2                               # Number of Simulations

a = 0.95                             # VaR, ES Confidence Level

parameterStart='2022-12-05'          # Historic Data Scrape Start Date

model = "GBM"                        # GBM, Jump

jumpThreshold = 0.015                # Defines the minimum daily % change which is considered jump -used in parameter estimation (Recommend 3.5%)

# Scrapes Historic Data
parameterData, forecastData = getData(ticker, parameterStart, T, model)

# Simulation
if model == "GBM":
    # Calculate Model Parameters
    mu, sigma = calculateGBMParameters(parameterData)
    
    # Run Simulation
    n, T, M, S0, St = gbmSimulation(mu, sigma, M, forecastData, T)

    # Calculate Risk Metrics
    aIndex, VaR, ES, sortedLoss = riskMetrics(a, T, M, S0, St)

    # Prints Model Output
    modelOutput(model, ticker, T, M, St, sortedLoss, a, VaR, ES, mu, sigma, None, None, None, None)    # Model Output (Table)

elif model == "Jump":
    # Calculate Model Parameters
    mu, sigma, mu_j, sigma_j, lambda_r = calculateJumpParameters(parameterData, jumpThreshold)
    
    # Run Simulation
    n, T, M, S0, St = jumpSimulations(mu, sigma, M, forecastData, T, mu_j, sigma_j, lambda_r)

    # Calculate Risk Metrics
    aIndex, VaR, ES, sortedLoss = riskMetrics(a, T, M, S0, St)

    # Prints Model Output
    modelOutput(model, ticker, T, M, St, sortedLoss, a, VaR, ES, mu, sigma, lambda_r, mu_j, sigma_j, jumpThreshold)    # Model Output (Table)


# Plots Data
mcPathsPlot(n, T, St, aIndex, a, VaR, ES, ticker, model)                # Monte Carlo Simulation Paths
lossDistPlot(a, M, VaR, ES, sortedLoss, ticker)                         # Loss Distribution
plt.show()                                                              # Show Plots