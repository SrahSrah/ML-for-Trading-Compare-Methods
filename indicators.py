"""
indicators.py

3-4 stock indicators

-each indicator must generate a chart to be used in the final report
-normalize the data when you do this!



"""
from util import get_data
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np
import pandas as pd

class Indicators():
    # Generates indicators for later use in projects

    def __init__(self, symbols = "JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), plot = False):
        dates = pd.date_range(sd, ed)
        self.data = get_data([symbols], dates)
        self.normed_w_spy_data = self.data/self.data.ix[0]
        self.plot = plot
        self.data = self.data[symbols] # removes SPY
        self.data = self.data.to_frame() # converts the dreaded series to a frame
        self.normed_data = self.data/self.data.ix[0] #normed w/o SPY
        self.symbol = symbols
        if self.plot: self.plot_data(self.normed_w_spy_data, "Normalized " + symbols + " vs. SPY")

    def author():
        return "shernandez43"

    def gen_ind1_momentum(self, n_days = 5):
        self.data["Momentum"] = (self.data/self.data.shift(n_days)) - 1
        result = pd.concat([self.data["Momentum"], self.normed_data[self.symbol]], axis = 1)
        self.normed_data["Momentum"] = self.data["Momentum"]
        if self.plot: self.plot_data(result, "Momentum")


    def gen_ind2_sma(self,n_days = 5):
        self.n_days = n_days
        sma = self.data[self.symbol].rolling(window = n_days).mean()
        normed_sma = self.normed_data[self.symbol].rolling(window = n_days).mean()

        sma = self.fillNA(sma)
        normed_sma = self.fillNA(normed_sma)

        self.data["SMA"] = sma
        self.normed_data["SMA"] = normed_sma
        self.data["Price/SMA"] = self.data["SMA"]/self.data[self.symbol]

        result = pd.concat([self.normed_data["SMA"], self.normed_data[self.symbol]], axis = 1)

        if self.plot: self.plot_data(result, "Simple Moving Average")



    def gen_ind3_bollinger_bands(self):

        rolling_std = self.data.ix[:,0].rolling(window = self.n_days, min_periods = self.n_days).std()

        self.data["Upper Bollinger"] = self.data["SMA"] + rolling_std*2
        self.data["Lower Bollinger"] = self.data["SMA"] - rolling_std*2

        normed_rolling_std = self.normed_data.ix[:,0].rolling(window = self.n_days, min_periods = self.n_days).std()

        self.normed_data["Upper Bollinger"] = self.normed_data["SMA"] + normed_rolling_std*2
        self.normed_data["Lower Bollinger"] = self.normed_data["SMA"] - normed_rolling_std*2

        result = pd.concat([self.normed_data[self.symbol], self.normed_data["Upper Bollinger"], self.normed_data["SMA"], self.normed_data["Lower Bollinger"]], axis = 1)
        if self.plot: self.plot_data(result, "Bollinger Bands")



    def gen_all_indicators(self):
        self.gen_ind1_momentum()
        self.gen_ind2_sma()
        self.gen_ind3_bollinger_bands()




    def get_all_inidcators(self):
        return self.data["Momentum"], self.data["SMA"], self.data["Lower Bollinger"], self.data["Upper Bollinger"]

    def get_normed_indicators(self):
        return self.normed_data["Momentum"], self.normed_data["SMA"], self.normed_data["Lower Bollinger"], self.normed_data["Upper Bollinger"]


    def plot_data(self, df, title="Stock prices", xlabel="Date", ylabel="Price"):
        """Plot stock prices with a custom title and meaningful axis labels."""
        plt.figure()
        ax = df.plot(title=title, fontsize=12)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.savefig("indicators_" + title + ".png")


    def fillNA(self, df):
        df = df.fillna(method = "ffill")
        df = df.fillna(method = "bfill")
        return df


if __name__ == "__main__":
    my_indictors = Indicators()
    my_indictors.gen_ind1_momentum()
    my_indictors.gen_ind2_sma()
    my_indictors.gen_ind3_bollinger_bands()
