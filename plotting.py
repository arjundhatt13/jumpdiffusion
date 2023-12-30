# Import modules
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime as dt
import datetime as dt1
import pandas_market_calendars as mcal
import seaborn as sns
import matplotlib.patches as mpatches

# Plot the simulation paths
def mcPathsPlot(n, T, St, aIndex, a, VaR, ES, ticker, model):

    tempList=St.transpose().tolist()
    tempList.sort(key=lambda ele: (ele[n]), reverse=True)

    pathsWithinLevel = np.asarray(tempList[:aIndex]).T
    np.mean(pathsWithinLevel, axis=1)
    pathsBreachedLevel = np.asarray(tempList[aIndex:]).T

    averagePaths = np.mean(np.asarray(tempList).T, axis=1)
    averagePathsBreachedLevel = np.mean(pathsBreachedLevel, axis=1)

    # Used for the x-axis of the plot
    time = np.linspace(0,T,n+1)

    # Plots
    plt.plot(time, pathsWithinLevel, color='green', label = "Non-Breaching Paths")
    plt.plot(time, pathsBreachedLevel, color='red', label = "VaR Breaching Paths")

    # Axis Labels
    plt.xlabel("Trading Days $(t)$")
    plt.ylabel("Stock Price $(S_t)$")
    plt.title(r"$\bf{" + ticker + "\ Stock\ Price\ Paths\ for\ " + model + "\ Simulation" + "}$" + "\n Mean Price = " + '\${:,.2f}'.format(St[-1,:].mean()) + ",  $\mathregular{VaR_{" + str(a) + "}}$ =" + "\${:,.2f}".format(VaR) + ",  $\mathregular{ES_{" + str(a) + "}}$ =" + "\${:,.2f}".format(ES))
    
    # Legend
    green_patch = mpatches.Patch(color = 'green', label = "Non-Breaching Paths")
    red_patch = mpatches.Patch(color = 'red', label = "VaR Breaching Paths")

    plt.legend(handles = [green_patch, red_patch])

    plt.show()

# Plot the loss distribution
def lossDistPlot(a, M, VaR, ES, sortedLoss, ticker):
    # Histogram Bins
    df=pd.DataFrame(sortedLoss)
    ax = sns.distplot(sortedLoss)
    
    data_x, data_y = ax.lines[0].get_data()

    # Plotting
    plt.fill_between(sortedLoss, np.interp(sortedLoss, data_x, data_y), where = sortedLoss >= VaR, color='red', alpha=0.8)
    
    #Labeling
    plt.xlabel("Loss ($)")
    plt.ylabel("Relative Frequency (%)")
    plt.title(r"$\bf{" + ticker + "\ Loss\ Distribution" + "}$" + "\n Mean Loss = " + '\${:,.2f}'.format(sortedLoss.mean()) + ",  $\mathregular{VaR_{" + str(a) + "}}$ =" + "\${:,.2f}".format(VaR) + ",  $\mathregular{ES_{" + str(a) + "}}$ =" + "\${:,.2f}".format(ES))
    plt.show()

# Model output in tabular form
def modelOutput(model, ticker, T, M, St, sortedLoss, a, VaR, ES, mu, sigma, lambda_r, mu_j, sigma_j, jumpThreshold):
    fig = plt.figure(figsize = (6,4)) 
    ax = fig.add_subplot(111)

    if model == "GBM":
        df = pd.DataFrame(["",
                        model, 
                        ticker, 
                        T,
                        M,
                        "",
                        "", 
                        '\${:,.2f}'.format(St[-1,:].mean()), 
                        '\${:,.2f}'.format(-sortedLoss.mean()),
                        "",
                        "",  
                        "\${:,.2f}".format(VaR), 
                        "\${:,.2f}".format(ES), 
                        "",
                        "",
                        "{:.2f}%".format(mu*252*100), 
                        "{:.2F}%".format(sigma*np.sqrt(252)*100)
                        ],

        index=["Model Information (User Inputted)", 
            "       Model Used", 
            "       Stock",
            "       Time Horizon (T) in Trading Days",
            "       Number of Simulations (M)",
            "", 
            "Projected Price Information",  
            "       Projected Price", 
            "       Projected P&L",
            "",
            "Risk Metrics", 
            "       $\mathregular{VaR_{" + str(a) + "}}$", 
            "       $\mathregular{ES_{" + str(a) + "}}$",
            "",
            "Model Parameters", 
            "       μ (Annualized)",
            "       σ (Annualized)" 
                    ])
    
    elif model == "Jump":
        df = pd.DataFrame(["",
                        model, 
                        ticker, 
                        T,
                        M,
                        '{:,.2f}%'.format(100*jumpThreshold),
                        "",
                        "", 
                        '\${:,.2f}'.format(St[-1,:].mean()), 
                        '\${:,.2f}'.format(-sortedLoss.mean()),
                        "",
                        "",  
                        "\${:,.2f}".format(VaR), 
                        "\${:,.2f}".format(ES), 
                        "",
                        "",
                        "{:.2f}%".format(mu*252*100), 
                        "{:.2f}%".format(sigma*np.sqrt(252)*100),
                        "{}".format(int(np.round(lambda_r * 252))), 
                        "{:.2e}".format(mu_j),
                        "{:.2e}".format(sigma_j),
                        ],

        index=["Model Information (User Inputted)", 
            "       Model Used", 
            "       Stock",
            "       Time Horizon (T) in Trading Days",
            "       Number of Simulations (M)", 
            "       Jump Threshold ($\mathregular{J_{T}}$)", 
            "", 
            "Projected Price Information",  
            "       Projected Price", 
            "       Projected P&L",
            "",
            "Risk Metrics", 
            "       $\mathregular{VaR_{" + str(a) + "}}$", 
            "       $\mathregular{ES_{" + str(a) + "}}$",
            "",
            "Model Parameters", 
            "       μ (Annualized)",
            "       σ (Annualized)",
            "       λ (Number of Jumps Annually)",
            "       $\mathregular{μ_{j}}$",
            "       $\mathregular{σ_{j}}$",

                    ])


    table = ax.table(cellText=df.values, rowLabels=df.index, loc = "upper center", colWidths=[0.2]*len(df.columns))
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    
    
    if model == "Jump":
        ax.set_title("Jump Diffusion Model Output", loc='left')
    else:
        ax.set_title("GBM Model Output", loc='left')
       
    ax.axis("off")
    fig.tight_layout()
    plt.show()


