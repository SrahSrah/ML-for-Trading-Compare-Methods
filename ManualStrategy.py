from util import get_data
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np
import pandas as pd
import marketsimcode as ms
from indicators import Indicators

class ManualStrategy():

    def __init__(self, title):
        self.title = title

    def author():
        return "shernandez43"

    def testPolicy(self, symbol = "JPM", sd=dt.datetime(2010,1,1), ed=dt.datetime(2011,12,31), sv = 100000):
        dates = pd.date_range(sd, ed)

        data = get_data([symbol], dates)
        self.data = data[[symbol]]
        self.sd = sd
        self.ed = ed
        self.symbol = symbol
        self.normed_data = data/data.iloc[0]

        benchmark = self.gen_benchmark()

        manstrat = self.gen_manual_trades()

        # create bench trades df:
        dummy = benchmark.copy()
        dummy[self.symbol] = benchmark["Shares"]
        bench_trades = dummy[self.symbol]
        bench_trades = bench_trades.to_frame()


        # PUTS MANSTRAT IN REQUIRED RETURN FORMAT:
        dummy = manstrat.copy()
        dummy[self.symbol] = manstrat["Shares"]
        dummy.loc[dummy["Order"] == "SELL", self.symbol] *= -1

        manstrat_trades = dummy[self.symbol]
        manstrat_trades = manstrat_trades.to_frame()


        return manstrat_trades, bench_trades

    def gen_manual_trades(self):

        indicators = Indicators(symbols = self.symbol, sd=self.sd, ed=self.ed)
        indicators.gen_all_indicators()
        momentum, sma, bollinger_minus, bollinger_plus = indicators.get_all_inidcators()


        # EXECUTE MANUAL STRATEGY
        cols = ["Symbol", "Order", "Shares"]
        trades = np.zeros(shape = (self.data.shape[0],3))
        trades = pd.DataFrame(trades, index = self.data.index.values, columns = cols)
        holdings = 0    # initial holdsing value:
        momentum_cutoff = 0.075
        for i in range(self.data.shape[0]):
            # if it's the first day, buy because the general trend of the economy is up:
            if i == 0 and holdings < 1000:
                to_buy = 1000 - holdings
                trades.ix[i] = (self.symbol, "BUY", to_buy)
                holdings = holdings + to_buy
            # first, buy/sell based on bollinger bands:
            # if price is below lower bollinger band, buy:
            elif bollinger_minus.ix[i] > self.data.ix[i].values[0] and holdings < 1000:
                to_buy = 1000 - holdings
                trades.ix[i] = (self.symbol, "BUY", to_buy)
                holdings = holdings + to_buy
            # elif price is above upper bollinger band, sell:
            elif bollinger_plus.ix[i] < self.data.ix[i].values[0] and holdings > -1000:
                to_sell = 1000 + holdings
                trades.ix[i] = (self.symbol, "SELL", to_sell)
                holdings = holdings - to_sell
            # if momentum slope is above a value, buy because we expect it to go up:
            elif self.get_momentum_slope(momentum, i) > momentum_cutoff and holdings < 1000:
                to_buy = 1000 - holdings
                trades.ix[i] = (self.symbol, "BUY", to_buy)
                holdings = holdings + to_buy
            # else if momentum slope is below a value, sell because we expect price to then go down:
            elif self.get_momentum_slope(momentum, i) < -momentum_cutoff and holdings > -1000:
                to_sell = 1000 + holdings
                trades.ix[i] = (self.symbol, "SELL", to_sell)
                holdings = holdings - to_sell
            # if sma is larger than the price, buy:
            elif self.sma_is_consistently_higher(sma, i) and holdings < 1000:
                to_buy = 1000 - holdings
                trades.ix[i] = (self.symbol, "BUY", to_buy)
                holdings = holdings + to_buy
            # elif sma is lower than price, sell:
            elif self.sma_is_consistently_lower(sma, i) and holdings > -1000:
                to_sell = 1000 + holdings
                trades.ix[i] = (self.symbol, "SELL", to_sell)
                holdings = holdings - to_sell



        trades["Symbol"] = self.symbol

        return trades

    # Returns true if sma is higher than price for a period of n days
    def sma_is_consistently_higher(self, sma, i, n_days = 3):
        if i < n_days:
            return False
        else:
            for day in range(n_days):
                if sma.ix[i-day] < self.data.ix[i-day].values[0]:
                    return False
            return True

    # Returns true if sma is lower than price for a period of n days:
    def sma_is_consistently_lower(self, sma, i, n_days = 3):
        if i < n_days:
            return False
        else:
            for day in range(n_days):
                if sma.ix[i-day] > self.data.ix[i-day].values[0]:
                    return False
            return True

    def get_momentum_slope(self, momentum, i, n_days = 5):
        if i < n_days:
            return 0
        else:
            return (momentum[i] - momentum[i-n_days])/n_days


    def gen_benchmark(self):
        columns = ["Symbol", "Order",
        "Shares"]
        index = self.data.index.values
        trades = np.zeros(shape = (len(index), len(columns)), dtype = object)
        trades[0][0], trades[0][1], trades[0][2] = self.symbol, "BUY", 1000
        trades = pd.DataFrame(trades, index = index, columns = columns)
        trades["Symbol"] = self.symbol

        return trades


    def plot(self, prices, benchmark, manual_trades, strategy_trades):

        manual_port = ms.compute_portvals(prices, manual_trades, commission=9.95, impact=0.005)
        bench_port = ms.compute_portvals(prices, benchmark, commission=9.95, impact=0.005)
        strat_port = ms.compute_portvals(prices, strategy_trades, commission=9.95, impact=0.005)
        manual_port = manual_port/manual_port.iloc[0]
        bench_port = bench_port/bench_port.iloc[0]
        strat_port = strat_port/strat_port.iloc[0]

        ms.get_port_stats(manual_port, self.title + " Normalized Manual Strategy")
        ms.get_port_stats(bench_port, self.title + " Normalized Benchmark")
        ms.get_port_stats(strat_port, self.title + " Normalized Strategy Learner")


        plt.figure()
        plt.plot(bench_port, label = "Benchmark", color ="green")
        plt.plot(manual_port, label = "Manual Strategy", color ="red")
        plt.plot(strat_port, label = "Strategy Learner", color = 'blue')
        plt.ylabel('Cumulative Return')
        plt.xlabel('Date')
        # longs = manual_trades.index[manual_trades[self.symbol] > 0]
        # for long in longs:
        #     plt.axvline(x = long, color = "blue")
        #
        # shorts = manual_trades.index[manual_trades[self.symbol] < 0]
        # for short in shorts:
        #     plt.axvline(x = short, color = "black")


        plt.title(self.title)
        plt.legend()
        plt.savefig(self.title + ".png")


if __name__ == "__main__":
    in_sample_manstrat = ManualStrategy(title = "In-Sample Results:")
    df_in_sample_trades = in_sample_manstrat.testPolicy(symbol = "JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = 100000)

    out_sample_manstrat = ManualStrategy(title = "Out-Sample Results:")
    df_out_sample_trades = out_sample_manstrat.testPolicy(symbol = "JPM", sd=dt.datetime(2010,1,1), ed=dt.datetime(2011,12,31), sv = 100000)
