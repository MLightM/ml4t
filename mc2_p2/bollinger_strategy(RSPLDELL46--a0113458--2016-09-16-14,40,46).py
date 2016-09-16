import pandas as pd
import numpy as np
import datetime as dt
import math
import os
import copy
import matplotlib.pyplot as plt
from util import get_data, plot_data



def get_bollinger_bands(rm, rstd):
    upper_band = rm + rstd * 2
    lower_band = rm - rstd * 2
    return upper_band, lower_band

def compute_cumulative_return(df):
    cumu_return = (df[-1] / df[0]) - 1
    return cumu_return

def compute_daily_return(df):
    daily_returns = df.copy()
    daily_returns[1:] = (df[1:] / df[:-1].values) - 1
    daily_returns = daily_returns[1:]
    # daily_returns = (df / df.shift(1)) - 1
    return daily_returns

def plot_data(df, rm_df, upper_band, lower_band):
    ax = df['IBM'].plot(title="IBM rolling mean", label='IBM')
    
    rm_df.plot(label='Rolling mean', ax=ax)
    upper_band.plot(label='upper_band', ax=ax)
    lower_band.plot(label='lower_band', ax=ax)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc='lower right')
    plt.show()

def bollinger_strategy():
    sv = 10000
    start_date = dt.datetime(2007,12,31)
    end_date = dt.datetime(2009,12,31)
    Dates_range = pd.date_range(start_date, end_date)

    df_IBM = get_data(['IBM'], Dates_range, addSPY=True)
    rm_IBM = pd.rolling_mean(df_IBM['IBM'], window=20)
    rstd_IBM = pd.rolling_std(df_IBM['IBM'], window=20)
    upper_band, lower_band = get_bollinger_bands(rm_IBM, rstd_IBM)
    # plot_data(df_IBM, rm_IBM, upper_band, lower_band)
    # print rm_IBM.head(30)

    #-------------strategy starts------------------
    # print df_IBM.index[19]
    print df_IBM.ix[19]['IBM']
    # for IBM in df_IBM[19:]['IBM']:
    # 	IBM_price = IBM.values


if __name__ == "__main__":
    bollinger_strategy()