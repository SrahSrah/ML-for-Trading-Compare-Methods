"""
Template for implementing StrategyLearner  (c) 2016 Tucker Balch

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

Student Name: Tucker Balch (replace with your name)
GT User ID: shernandez43 (replace with your User ID)
"""

import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd
import numpy as np
import util as ut
import random
from QLearner import QLearner as ql
from indicators import Indicators
from marketsimcode import compute_portvals, get_stats


class StrategyLearner(object):

    LONG = 1
    CASH = 0
    SHORT = -1

    # constructor
    def __init__(self, verbose = False, impact=0.0):
        self.verbose = verbose
        self.impact = impact
        self.Q = ql(num_states = 10**5-1, num_actions = 3)

    def author():
        return "shernandez43"

    # this method should create a QLearner, and train it for trading
    def addEvidence(self, symbol = "IBM", \
        sd=dt.datetime(2008,1,1), \
        ed=dt.datetime(2009,1,1), \
        sv = 10000):

        # Set up function:
        prices, indicators = self.setup(sd, ed, symbol)

        cum_returns = [] # initialize cumulative returns
        # now iterate:
        epochs = 100
        orders = pd.Series(index=indicators.index)
        for e in range(epochs):
            pos = 0 # set initial position
            for i, date in enumerate(indicators.index):
                # 1. discretize the state:
                state = self.discretize(indicators.loc[date], pos + 1)
                # 2. if the first day, set the state, get the action:
                if date == indicators.index.values[0]:
                    action = self.Q.querysetstate(state)
                else:
                    old_price = prices[self.syms].iloc[i-1] # iloc example
                    new_price = prices[self.syms].loc[date] # loc example
                    reward = pos * ((new_price / old_price) - 1)
                    action = self.Q.query(state, reward)

                # 3. find new position, set it:
                new_pos = self.get_new_pos(pos, action - 1)
                orders.loc[date] = new_pos
                pos += new_pos

            # compute df_trades and portvals for the epoch:

            df_trades = self.get_trades_df(orders)
            port_vals = compute_portvals(prices, df_trades, start_val=sv, impact=self.impact)
            port_stats = get_stats(port_vals)
            cum_returns.append(port_stats[0])

            if self.verbose: print e, cum_returns[-1]

            # If converged, break
            if e > 11:
                if self.converged(cum_returns):
                    break

        # For Experiment2, store cumulative return and sharpe ratio
        self.cr = np.float(cum_returns[-1])
        self.sr = np.float(port_stats[-1])
        if self.verbose:
            plt.plot(cum_returns)
            plt.xlabel("Epoch")
            plt.ylabel("Cumulative Return")
            plt.title("Cumulative Return over Epoch")
            plt.savefig("AddEvidenceResults.jpg")



    # this method should use the existing policy and test it against new data
    def testPolicy(self, symbol = "IBM", \
        sd=dt.datetime(2009,1,1), \
        ed=dt.datetime(2010,1,1), \
        sv = 10000):

        # Set up function:
        prices, indicators = self.setup(sd, ed, symbol)

        # Run through the policy:
        pos = self.CASH # == 0
        orders = pd.Series(index=indicators.index)
        for date in indicators.index:
            # 1. Discretize the state
            state = self.discretize(indicators.loc[date], pos + 1)
            # 2. Get the prescribed action:
            action = self.Q.querysetstate(state)
            # 3. Get the new position, update old position:
            new_pos = self.get_new_pos(pos, action - 1)
            pos += new_pos
            # 4. Add new position to orders, don't worry, will be 0 if no action taken:
            orders.loc[date] = new_pos

        trades = self.get_trades_df(orders)

        if self.verbose: print type(trades) # it better be a DataFrame!
        if self.verbose: print trades

        return trades


    def setup(self, sd, ed, symbol):
        self.sd, self.ed = sd, ed
        self.syms=symbol
        self.dates = pd.date_range(sd, ed)
        prices_all = ut.get_data([self.syms], self.dates)  # automatically adds SPY
        prices = pd.DataFrame(prices_all[self.syms])  # only portfolio symbols

        # get indicators in format: [momentum, sma, low_bollinger, high_bollinger]
        # fill in nan's as needed:
        indicators = self.get_indicators(prices)
        indicators = indicators.ffill()
        indicators = indicators.bfill()
        # calculate thresholds for discretization:
        self.thresholds = self.get_thresholds(indicators, num_steps = 10)
        self.prices = prices
        return prices, indicators

    def get_indicators(self, prices):
        calc_indicators = Indicators(self.syms, self.sd, self.ed)
        calc_indicators.gen_all_indicators()
        momentum, sma, low_bollinger, high_bollinger = calc_indicators.get_normed_indicators()
        df = pd.DataFrame(index = prices.index)
        df['Momentum'] = momentum
        df['SMA'] = sma
        df['Bollinger Low'] = low_bollinger
        df['Bollinger High'] = high_bollinger
        return df

    def get_thresholds(self, indicators, num_steps = 10):
        """
        Calculate the thresholds for later discretization.
        Thresholds is a 2d array of size number_of_indicators x number_of_steps,
        and will be used to categorize each value of each indicator.
        """

        self.num_steps = num_steps
        step_size = indicators.shape[0]/num_steps
        thresholds = np.zeros(shape=(indicators.shape[1], num_steps))
        dummy_df = indicators.copy() # Don't want to reorder the orginal!
        i = 0
        for col in indicators.columns.values:
            dummy_df.sort_values(by=[col], inplace = True)
            for step in range(num_steps):
                if step < num_steps - 1:
                    thresholds[i, step] = dummy_df[col].iloc[(step+1)*step_size]
                else:
                    # if the last step, create custom bin:
                    thresholds[i, step] = dummy_df[col].iloc[-1]
            i += 1
        return thresholds

    def discretize(self, indicator_row, pos):
        # Find state based on given date of indicator data:
        s = pos * (self.num_steps ** len(indicator_row))
        for i in range(len(indicator_row)):
            thres = self.thresholds[i][self.thresholds[i] >= indicator_row[i]][0]
            thres = np.where(self.thresholds == thres)[1][0]
            s += thres * (self.num_steps)**i

        return s


    def get_new_pos(self, old_pos, action):
        new_pos = self.CASH
        if old_pos > self.SHORT & action == self.SHORT:
            new_pos = self.SHORT
        elif old_pos < self.LONG & action == self.LONG:
            new_pos = self.LONG
        return new_pos

    def get_trades_df(self, orders):
        shares = 1000
        trades = []
        for date in orders.index:
            if orders.loc[date] == self.SHORT:
                trades.append((date, self.SHORT*shares))
            elif orders.loc[date] == self.LONG:
                trades.append((date, self.LONG*shares))
            else:
                trades.append((date, self.CASH*shares))

        # This is how your notes say to set the index:
        df_trades = pd.DataFrame(trades, columns=["Date", self.syms])
        df_trades.set_index("Date", inplace=True)
        return df_trades



    def converged(self, returns, patience=10):
        if patience > len(returns):
            return False

        latest = np.array(returns[-patience:])
        if len(np.unique(latest)) == 1:
            return True

        max_r = np.max(returns)

        if max_r in latest:
            if max_r not in np.array(returns[:len(returns)-patience]):
                return False
            else:
                return True
        # if latest does not contain max_r, it has converged:
        else:
            return True





if __name__=="__main__":
    print "One does not simply think up a strategy"
    learner = StrategyLearner(verbose = True)
    # learner.addEvidence(symbol = "AAPL", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = 100000) # training phase
    # df_trades = learner.testPolicy(symbol = "AAPL", sd=dt.datetime(2010,1,1), ed=dt.datetime(2011,12,31), sv = 100000) # testing phase



    np.random.seed = 1481090000

    learner.addEvidence(symbol = "UNH", sd=dt.datetime(2008, 1, 1, 0, 0), ed=dt.datetime(2009, 12, 31, 0, 0), sv = 100000) # training phase
    df_trades = learner.testPolicy(symbol = "UNH", sd=dt.datetime(2010, 1, 1, 0, 0), ed=dt.datetime(2011, 12, 31, 0, 0), sv = 100000) # testing phase





    #
    #
    # insample_args = {'ed': , 'sd': , 'sv': 100000, 'symbol': 'UNH'}
    # outsample_args = {'ed': datetime.datetime(2011, 12, 31, 0, 0), 'sd': datetime.datetime(2010, 1, 1, 0, 0), 'sv': 100000, 'symbol': 'UNH'}
    # benchmark_type = 'stock', benchmark = -0.25239999999999996, impact = 0.0, train_time = 25, test_time = 5, max_time = 60
