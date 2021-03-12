"""
Template for implementing QLearner  (c) 2015 Tucker Balch

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
GT User ID: tb34 (replace with your User ID)
GT ID: 900897987 (replace with your GT ID)
"""

import numpy as np
import random as rand

class QLearner(object):

    def __init__(self, \
        num_states=100, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.5, \
        radr = 0.99, \
        dyna = 0, \
        verbose = False):

        self.verbose = verbose
        self.num_actions = num_actions
        self.num_states = num_states
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna

        self.s = 0
        self.a = 0

        self.q = np.zeros(shape = (self.num_states, self.num_actions))

        #init dyna:
        if self.dyna != 0:
            self.R = -1 * np.ones((num_states, num_actions))
            self.Tc = 0.00001 * np.ones((num_states, num_actions, num_states))
            self.T = self.Tc/self.Tc.sum(axis = 2, keepdims = True)

    def author(self):
        return 'shernandez43'

    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table. This is also
        used to initialize the q-table
        @param s: The new state
        @returns: The selected action
        """

        self.s = s
        action = np.argmax(self.q[s, :])

        if self.verbose: print "s =", s,"a =",action

        return action

    def query(self,s_prime,r):
        """
        @summary: Update the Q table and roll the dice to see if you return a random action or not,
                  and return an action. Also update rar: rar = rar*radr
        @param s_prime: The new state
        @param r: The new state
        @returns: The selected action
        """

        # 1. Update Q table:
        self.q[self.s, self.a] = (1.0-self.alpha) * self.q[self.s, self.a] + \
                                self.alpha*(r + (self.gamma*np.max(self.q[s_prime,:])))


        # 2. Implement Dyna:
        if self.dyna != 0:
            # increment the count:
            self.Tc[self.s, self.a, s_prime] += 1

            #Normalize counts:
            counts = self.Tc[self.s, self.a, :]
            self.T[self.s, self.a, :] = counts / counts.sum()

            # Update Rewards:
            self.R[self.s, self.a] = (1 - self.alpha) * self.R[self.s, self.a] + self.alpha * r

            # Hallucinate:
            self.hallucinate()


        # 3. Choose action:
        action = self.get_action(s_prime)

        if self.verbose: print "s =", s_prime,"a =",action,"r =",r

        self.s = s_prime
        self.a = action

        return action

    def hallucinate(self):
        # first, generate some samples:
        samples_s = np.random.randint(0, self.num_states, self.dyna)
        samples_a = np.random.randint(0, self.num_actions, self.dyna)

        # simulate actions for each sample:
        for s, a in zip(samples_s, samples_a):
            s_prime = np.argmax(np.random.multinomial(1, self.T[s, a, :]))
            r = self.R[s, a] # reward of the sample action
            #update Q:
            self.q[s,a] = (1 - self.alpha) * self.q[s, a] + \
                self.alpha * (r + self.gamma * np.max(self.q[s_prime,:]))
  
    def get_action(self, s_prime):
        randnum = rand.uniform(0.0,1.0)
        if randnum <= self.rar:
            action = rand.randint(0, self.num_actions-1)
        else:
            action = np.argmax(self.q[s_prime, :])

        self.rar = self.rar * self.radr

        return action

if __name__=="__main__":
    print "Remember Q from Star Trek? Well, this isn't him"
