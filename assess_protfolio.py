import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

def assess_portfolio(sd,ed,syms,allocs,sv,rfr,sf,gen_plot):
    dates = pd.date_range(sd, ed)
    df_dates = pd.DataFrame(index = dates)
    df_total = df_dates.copy()
    if 'SPY' not in syms:
        syms.append('SPY')

    for symbol in syms:
        df = pd.read_csv("data/{}.csv".format(symbol), index_col="Date", parse_dates=True, usecols=['Date','Adj Close'],na_values=['nan'])
        df = df.rename(columns={'Adj Close':symbol})
        df_total = df_total.join(df)
        if symbol == 'SPY':
            df_total = df_total.dropna(subset=["SPY"])
    #print df_total.head(), df_total.tail()

    port_val = daily_port_val(df_total.ix[:,0:-1],allocs,sv)
    #print port_val.head()
	
    print 'Start Date: ', sd
    print 'End Date: ', ed
    print 'Symbols: ', syms[0:-1]
    print 'Allocations: ', allocs
    

    daily_returns = compute_daily_return(port_val)
    dr_std = daily_returns.std()
    dr_aver = daily_returns.mean()
    cumu_return = compute_cumulative_return(port_val)
    SR = sqrt(sf)*(dr_aver-rfr)/dr_std
    print 'Sharp Ratio: ', SR
    print 'Volatility (stdev of daily returns): ', dr_std
    print 'Average Daily Return: ', dr_aver
    print 'Cumulative Return: ', cumu_return

    if gen_plot:
        s1 = normalize_data(port_val)
        s1.name = "Portfolio"
        s2 = normalize_data(df_total['SPY'])
        final_data = pd.concat([s1, s2], axis=1)
        plot_data(final_data)
    
def normalize_data(df):
    return df/ df.ix[0,:]

def daily_port_val(df,allocs,sv):
    normed = normalize_data(df)
    alloced = normed*allocs
    pos_vals = alloced*sv
    port_val = pos_vals.sum(axis=1)
    return port_val

def plot_data(df, title="Stock prices"):
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    #rm_SPY = get_rolling_mean(df['SPY'], window=20)
    #rstd_SPY = get_rolling_std(df['SPY'], window=20)
    #upper_band, lower_band = get_bollinger_bands(rm_SPY,rstd_SPY)
    #upper_band.plot(label='upper band', ax=ax)
    #lower_band.plot(label='lower band', ax=ax)
    plt.grid()
    plt.show()
    
def compute_beta_alpha(df):
    daily_returns = compute_daily_return(df)
    daily_returns.plot(kind='scatter',x='SPY',y='GLD')
    beta_GLD,alpha_GLD=np.polyfit(daily_returns['SPY'],daily_returns['GLD'],1)
    print "beta_GLD ",beta_GLD
    print "alpha_GLD= ",alpha_GLD
    plt.plot(daily_returns['SPY'], beta_GLD*daily_returns['SPY']+alpha_GLD,'-',color='r')
    plt.show()

def get_rolling_mean(df, window):
    return pd.rolling_mean(df, window=window)

def get_rolling_std(df, window):
    return pd.rolling_std(df, window=window)

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


if __name__ == "__main__":
    assess_portfolio(sd='2010-01-01',ed='2010-12-31',syms=['GOOG','AAPL','GLD','XOM'], allocs=[0.2,0.3,0.4,0.1], sv=1000000, rfr=0.0,sf=252,gen_plot=True)
