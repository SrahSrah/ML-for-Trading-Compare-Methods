"""MC2-P1: Market simulator.

Copyright 2018, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 4646/7646

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as github and gitlab.  This copyright statement should not be removed
or edited.

We do grant permission to share solutions privately with non-students such
as potential employers. However, sharing with other current or future
students of CS 7646 is prohibited and subject to being investigated as a
GT honor code violation.

-----do not edit anything above this line---

Student Name: Sarah Hernandez
GT User ID: shernandez43
GT ID: 903458532
"""

import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data
#import TheoreticallyOptimalStrategy

def author():
    return "shernandez43"

def fillNA(prices):
    prices = prices.fillna(method = "ffill")
    prices = prices.fillna(method = "bfill")
    return prices

# takes in date pd index and returns datetime obj:
def get_date(pd_date):
    pd_date = str(pd_date)
    year, month, day = int(pd_date[:4]), int(pd_date[5:7]), int(pd_date[8:10])
    return dt.datetime(year, month, day)

def compute_portvals(prices, orders, start_val = 1000000, commission=9.95, impact=0.005):


    # Create orders df
    #orders = pd.read_csv(orders_file, index_col = "Date", parse_dates = True, na_values = np.nan)

    # TODO: change input orders df from Symbol Order  Shares to just the change in Shares
    orders = orders.sort_index()
    syms = orders.columns.values[0]

    # Set start and end dates
    start_date = get_date(orders.index.values[0])
    end_date = get_date(orders.index.values[-1])
    dates = pd.date_range(start_date, end_date)


    # Get adjusted prices:
    #prices = get_data([syms], dates)
    #prices = prices[syms].to_frame()
    prices = fillNA(prices)
    # Add CASH to prices df:
    prices["CASH"] = 1
    syms = prices.columns.values
    sym = syms[0]
    # Create trades df:
    valid_dates = prices.index.values
    all_dates = np.unique(orders.index.values)
    trades = np.zeros(shape = (len(valid_dates),len(syms)))
    trades = pd.DataFrame(trades, index = valid_dates, columns = syms)


    for date in all_dates:
        if date not in valid_dates:
            continue
        day_trades = orders.loc[[date]]
        if day_trades.empty:
            continue
        else:
            for index, trade in day_trades.iterrows():
                shares = trade.values[0]
                impact_cost = impact * prices.loc[date][sym] * np.abs(shares)

                trades.loc[date][sym] = trades.loc[date][sym] + shares
                cash_change = -shares * prices.loc[date][sym]
                trades.loc[date]["CASH"] = trades.loc[date]["CASH"] + cash_change


                trades.loc[date]["CASH"] = trades.loc[date]["CASH"] - commission - impact_cost



    # Create holdings df:
    holdings = np.zeros(shape = (len(valid_dates),len(syms)))
    holdings = pd.DataFrame(holdings, index = valid_dates, columns = syms)

    holdings = trades.cumsum()
    holdings["CASH"] = holdings["CASH"] + start_val

    # Create value df:
    values = np.zeros(shape = (len(valid_dates),len(syms)))
    values = pd.DataFrame(values, index = valid_dates, columns = syms)
    for i in range(len(holdings)):
        values.iloc[i] = prices.iloc[i] * holdings.iloc[i]

    portvals = pd.DataFrame(values.sum(axis = 1))

    return portvals



def get_port_stats(portvals, title = ""):
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters


    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]] # just get the first column
    else:
        "warning, code did not return a DataFrame"

    # Get portfolio stats
    # Here we just fake the data. you should use your code from previous assignments.
    start_date = get_date(portvals.index.values[0])
    end_date = get_date(portvals.index.values[-1])
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = get_stats(portvals)
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = get_SPY_stats(start_date, end_date)

    # Compare portfolio against $SPX
    print
    print "STATS OF: " + title
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
    print



def get_SPY_stats(sd, ed):

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(["$SPX"], dates)  # automatically adds SPY
    prices_all = fillNA(prices_all) # forward and backfills nan values as neccessary

    prices_SPY = prices_all['$SPX']  # only SPY, for comparison later


    # computes stats based on optimal allocation
    cr, adr, sddr, sr = get_stats(prices_SPY)



    return cr, adr, sddr, sr


def get_stats(port_vals):

    drs = (port_vals/port_vals.shift(1)) - 1
    drs = drs[1:]

    cr = (port_vals.iloc[-1]/port_vals.iloc[0]) - 1
    adr = drs.mean()
    sddr = drs.std()
    k = np.sqrt(252)
    sr = drs.mean()/drs.std() * k # drr is 0.0

    return cr, adr, sddr, sr



if __name__ == "__main__":
    #trades = TheoreticallyOptimalStrategy().testPolicy(symbol = "JPM")
    #test_code(trade)
    pass
