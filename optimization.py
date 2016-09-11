import pandas as pd
import scipy.optimize as spo
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from math import sqrt

def get_data(sd, ed, syms):
	dates = pd.date_range(sd, ed)
	df_dates = pd.DataFrame(index=dates)
	df_total = df_dates.copy()
	if "SPY" not in syms:
		syms.append("SPY")

	for sym in syms:
		df = pd.read_csv("data/{}.csv".format(sym), index_col="Date", parse_dates=True, usecols=['Date','Adj Close'],na_values=['nan'])
		df = df.rename(columns={"Adj Close":sym})
		df_total = df_total.join(df)
		if sym == "SPY":
			df_total = df_total.dropna(subset=["SPY"])

	return df_total

def daily_port_val(df, allocs, sv):
	norm = df / df.ix[0,:]
	alloced = norm * allocs
	pos_vals = alloced * sv
	port_vals = pos_vals.sum(axis = 1)
	return port_vals

def compute_daily_return(df):
    daily_returns = df.copy()
    daily_returns[1:] = (df[1:] / df[:-1].values) - 1
    daily_returns = daily_returns[1:]
    # daily_returns = (df / df.shift(1)) - 1
    return daily_returns

def NegSharpRatio(allocs, df):
	rfr = 0
	sf = 252
	sv = 1000000
	port_vals = daily_port_val(df.ix[:,0:-1], allocs, sv)    # exclude SPY
	daily_returns = compute_daily_return(port_vals)
	dr_aver = daily_returns.mean()
	dr_std = daily_returns.std()
	SR = sqrt(sf)*(dr_aver-rfr)/dr_std
	return -SR

def fit_line(data, error_func):
    syms_num = data.shape[1]-1 # num of securites
    iniAlloc = np.ones((syms_num,), dtype=np.float)/syms_num # uniform allocs
    bnds = ((0,1),)*syms_num
    cons = ({'type':'eq','fun': lambda x: np.sum(x)-1},)
    result = spo.minimize(error_func, iniAlloc, args=(data,), method='SLSQP', options={'disp':False}, bounds=bnds, constraints=cons)
    return result.x, result.fun

def optimize_portfolio(sd, ed, syms):
    df_total = get_data(sd, ed, syms)
    allocs, SR = fit_line(df_total, NegSharpRatio)
    #print allocs, -SR
    print 'Start Date: ', sd
    print 'End Date: ', ed
    print 'Symbols: ', syms[0:-1]
    print 'Optimal allocations: ', allocs
    print 'Sharpe Ratio: ', -SR
    # print 'Volatility (stdev of daily returns): ',
    # print 'Average Daily Return: ',
    # print 'Cumulative Return: ',

	

if __name__ == "__main__":
	optimize_portfolio(dt.datetime(2004,12,1), dt.datetime(2006,05,31), ['YHOO', 'XOM', 'GLD', 'HNZ'])

