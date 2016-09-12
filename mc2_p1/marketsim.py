"""MC2-P1: Market simulator."""

import pandas as pd
import numpy as np
import datetime as dt
import math
import os
import copy
from util import get_data, plot_data

def compute_portvals(orders_file = "./orders/orders-leverage-3.csv", start_val = 1000000):
    # this is the function the autograder will call to test your code
    # TODO: Your code here
    longs = 0
    shorts = 0
    cash = start_val
    Symbols = {}
    ordersfromfile = pd.read_csv(orders_file, index_col='Date', parse_dates=True, na_values=['nan'])
    print ordersfromfile
    for date, row in ordersfromfile.iterrows():
    	old_Symbols = copy.deepcopy(Symbols)
        old_cash = cash
        symbol = row['Symbol']
        if symbol not in Symbols:
        	Symbols[symbol] = 0
        dates = pd.date_range(date, date)
    	df = get_data(Symbols.keys(), dates, addSPY=False)     # get that row's date, symbol, price
    	# print df
        security_price = df[symbol].values[0]
        trans_price = security_price*row['Shares']
    	if row['Order'] == 'BUY':
    		Symbols[symbol] += row['Shares']
    		cash -= trans_price
        else:
    		Symbols[symbol] -= row['Shares']
    		cash += trans_price
    	
        portfolio_stats = df.values[0] * np.array(Symbols.values())
        longs = portfolio_stats[portfolio_stats>=0]
        shorts = portfolio_stats[portfolio_stats<0]

        # print Symbols, "--------Portfoilo-------",'\n'
        # print date, row, "--------Current Trading-------",'\n'
        print "Portvals: {}, Cash: {}, Longs: {}, Shorts: {}".format(cash+longs.sum()+shorts.sum(), cash, longs.sum(), shorts.sum()), '\n'
        leverage = compute_leverage(longs, shorts, cash)
        if leverage > 2:
        	Symbols = old_Symbols
        	# longs = old_longs
        	# shorts = old_shorts
        	cash = old_cash

    	

    # In the template, instead of computing the value of the portfolio, we just
    # read in the value of IBM over 6 months
    #start_date = dt.datetime(2008,1,1)
    #end_date = dt.datetime(2008,6,1)
    #portvals = get_data(['IBM'], pd.date_range(start_date, end_date))
    #portvals = portvals[['IBM']]  # remove SPY

    #return portvals

def compute_leverage(longs, shorts, cash):
	return (sum(longs) - sum(abs(shorts))) / (sum(longs) + sum(abs(shorts)) + cash)

# def 

def test_code():
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    of = "./orders/orders2.csv"
    sv = 1000000

    # Process orders
    portvals = compute_portvals(orders_file = of, start_val = sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]] # just get the first column
    else:
        "warning, code did not return a DataFrame"
    
    # Get portfolio stats
    # Here we just fake the data. you should use your code from previous assignments.
    start_date = dt.datetime(2008,1,1)
    end_date = dt.datetime(2008,6,1)
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [0.2,0.01,0.02,1.5]
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [0.2,0.01,0.02,1.5]

    # Compare portfolio against $SPX
    print "Date Range: {} to {}".format(start_date, end_date)
    print
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    print "Sharpe Ratio of SPY : {}".format(sharpe_ratio_SPY)
    print
    print "Cumulative Return of Fund: {}".format(cum_ret)
    print "Cumulative Return of SPY : {}".format(cum_ret_SPY)
    print
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    print "Standard Deviation of SPY : {}".format(std_daily_ret_SPY)
    print
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    print "Average Daily Return of SPY : {}".format(avg_daily_ret_SPY)
    print
    print "Final Portfolio Value: {}".format(portvals[-1])

if __name__ == "__main__":
    test_code()
