
import StrategyLearner as sl
import ManualStrategy as ml
import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd
import numpy as np
import util as ut
import random

class Experiment1():

    def __init__(self):
        pass

    def author(self):
        return 'shernandez43'

    def run(self):
        sd = dt.datetime(2008,1,1)
        ed = dt.datetime(2009,12,31)
        sv = 100000
        sym = 'JPM'
        s_learner = sl.StrategyLearner(verbose=False, impact=0.000)
        s_learner.addEvidence(symbol=sym, sd=sd, ed=ed, sv=sv)
        sl_trades = s_learner.testPolicy(symbol=sym, sd=sd, ed=ed, sv=sv)
        prices = s_learner.prices
        m_learner = ml.ManualStrategy(title = "Manual Strategy vs. Stratey Learner vs. Benchmark")
        ml_trades, bench_trades = m_learner.testPolicy(symbol=sym, sd=sd, ed=ed, sv=sv)

        m_learner.plot(prices, bench_trades, ml_trades, sl_trades)

if __name__=="__main__":
    exp = Experiment1()
    exp.run()
