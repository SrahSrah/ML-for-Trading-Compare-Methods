
import StrategyLearner as sl
import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd
import numpy as np
import util as ut
import random

class Experiment2():

    def __init__(self):
        pass

    def author(self):
        return 'shernandez43'

    def run(self):
        sd = dt.datetime(2008,1,1)
        ed = dt.datetime(2009,12,31)
        sv = 100000
        sym = 'JPM'
        impacts = [0.0, 0.0025, 0.005, 0.0075, 0.01]
        crs = []
        srs = []
        for impact in impacts:
            s = sl.StrategyLearner(verbose=False, impact=impact)
            s.addEvidence(symbol=sym, sd=sd, ed=ed, sv=sv)
            crs.append(s.cr)
            srs.append(s.sr)

        plt.plot(impacts, crs)
        plt.xlabel("Impact")
        plt.ylabel("Cumulative Return")
        plt.title("Cumulative Return vs. Impact")
        plt.savefig("Experiment2_cr.jpg")


        plt.figure()

        plt.plot(impacts, srs)
        plt.xlabel("Impact")
        plt.ylabel("Sharpe Ratio")
        plt.title("Sharpe Ratio vs. Impact")
        plt.savefig("Experiment2_sr.jpg")
        plt.show()






        #sl_trades = s_learner.testPolicy(symbol=sym, sd=sd, ed=ed, sv=sv)




        #prices = s_learner.prices






if __name__=="__main__":
    exp = Experiment2()
    exp.run()
