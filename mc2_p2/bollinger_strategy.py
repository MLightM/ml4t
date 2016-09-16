import pandas as pd
import numpy as np
import datetime as dt
import math
import os
import copy
import matplotlib.pyplot as plt
from util import get_data, plot_data
from math import sqrt


def get_bollinger_bands(rm, rstd):
    upper_band = rm + rstd * 2
    lower_band = rm - rstd * 2
    return upper_band, lower_band

def compute_cumulative_return(df):
    cumu_return = (df.ix[-1] / df.ix[0]) - 1
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
    # plt.show()
    return ax

def judge_state(IBM_price, IBM_rm, upper, lower):
	if IBM_price[0] > upper[0] and IBM_price[1] < upper[1]:
		return 'short_entry'
	elif IBM_price[0] > IBM_rm[0] and IBM_price[1] < IBM_rm[1]:
		return 'short_exit'
	elif IBM_price[0] < lower[0] and IBM_price[1] > lower[1]:
		return 'long_entry'
	elif IBM_price[0] < IBM_rm[0] and IBM_price[1] > IBM_rm[1]:
		return 'long_exit'
	else:
		pass

def bollinger_strategy():
    sv = 10000
    Shares = 0 # initial Shares of IBM security
    Cash = sv
    portval = Cash
    start_date = dt.datetime(2007,12,31)
    end_date = dt.datetime(2009,12,31)
    Dates_range = pd.date_range(start_date, end_date)

    df_IBM = get_data(['IBM','$SPX'], Dates_range, addSPY=True)
    rm_IBM = pd.rolling_mean(df_IBM['IBM'], window=20)
    rstd_IBM = pd.rolling_std(df_IBM['IBM'], window=20)
    upper_band, lower_band = get_bollinger_bands(rm_IBM, rstd_IBM)
    ax = plot_data(df_IBM, rm_IBM, upper_band, lower_band)
    # print rm_IBM.head(30)

    #-------------strategy starts------------------
    # print df_IBM.index[19]
    # print df_IBM[19:]['IBM']
    # Dates_range2 = pd.date_range(df_IBM.index[19], end_date)
    daily_portvals = pd.DataFrame(index=df_IBM.index)
    daily_portvals['Daily_portval'] = sv
    yestoday = df_IBM.index[19]
    for date in df_IBM.index[20:]:
    	IBM_price = df_IBM['IBM'][yestoday:date]
    	IBM_rm = rm_IBM[yestoday:date]
    	upper = upper_band[yestoday:date]
    	lower = lower_band[yestoday:date]
    	# plt.show()
    	# print IBM_price, IBM_rm, upper, lower
    	flag = judge_state(IBM_price, IBM_rm, upper, lower)
    	# flag = 'short_entry'
    	if Shares == 0 and flag == 'short_entry':
    		ax.axvline(pd.to_datetime(date), color='r')
    		Shares -= 100
    		Cash += 100 * IBM_price[1]
    		portval = Cash + Shares*IBM_price[1]
    		# print date, 'IBM', 'SELL', 100
    	elif Shares == 0 and flag == 'long_entry':
    		ax.axvline(pd.to_datetime(date), color='g')
    		Shares += 100
    		Cash -= 100 * IBM_price[1]
    		portval = Cash + Shares*IBM_price[1]
    		# print date, 'IBM', 'BUY', 100
    	elif Shares == 100 and flag == 'long_exit':
    		ax.axvline(pd.to_datetime(date), color='k')
    		# print date, 'IBM', 'SELL', 100
    		Shares -= 100
    		Cash += 100 * IBM_price[1]
    		portval = Cash + Shares*IBM_price[1]
    	elif Shares == -100 and flag == 'short_exit':
    		ax.axvline(pd.to_datetime(date), color='k')
    		# print date, 'IBM', 'BUY', 100
    		Shares += 100
    		Cash -= 100 * IBM_price[1]
    		portval = Cash + Shares*IBM_price[1]
    	else:
    		portval = Cash + Shares*IBM_price[1]

    	yestoday = date
    	daily_portvals['Daily_portval'][date] = portval

    print daily_portvals
    plt.show()

    norm_portvals = daily_portvals/daily_portvals.ix[0]
    norm_portvals.plot()
    SPX_norm = df_IBM['$SPX']/df_IBM['$SPX'].ix[0]
    SPX_norm.plot(color='g')
    plt.show()

    #-------------Calculate marks---------------
    daily_portvals = daily_portvals['2008-02-28':'2009-12-29']
    IBM_daily_returns = compute_daily_return(daily_portvals)
    IBM_cumu = compute_cumulative_return(daily_portvals)
    IBM_std = IBM_daily_returns.std()
    IBM_aver = IBM_daily_returns.mean()
    IBM_SR = sqrt(252)*(IBM_aver-0)/IBM_std
    print "SR: ", IBM_SR
    print "Cumulative Return: ", IBM_cumu
    print "Standard Deviation: ", IBM_std
    print "Average Daily Return: ", IBM_aver
    print "Final Portfolio Vaule: ", daily_portvals.ix[-1]


if __name__ == "__main__":
    bollinger_strategy()