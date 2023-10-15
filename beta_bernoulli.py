import scipy.stats as stats
from Conj_Model import Conj_Model
import matplotlib.pyplot as plt
import numpy as np

def plot():
    n_posterior = [5, 25, 100, 250, 500] #for these n the posterior will be plotted
    model = Conj_Model(prior = stats.beta, hyperpara = [2, 2], likelikhood = stats.bernoulli, theta_0 = .7 , n = max(n_posterior))  # define a model
    y = model.sample_data() #sample some data
    fig, ax = plt.subplots(tight_layout=True)
    model.plot_analytic_posterior(y = y, n_posterior = n_posterior, prior = True, ax = ax)
    fig, ax = plt.subplots(tight_layout=True)
    model.plot_posterior_and_bernstein(y = y, ax=ax)

def plot_a():
    n_posterior = [5, 50, 100]  # for these n the posterior will be plotted
    model = Conj_Model(prior = stats.beta, hyperpara = [2, 2], likelikhood = stats.bernoulli, theta_0 = .7 , n = max(n_posterior))  # define a model
    y = model.sample_data() #sample some data
    fig, ax = plt.subplots(tight_layout=True)
    model.plot_posterior_and_bernstein_dashed(y = y, ax=ax, n_posterior=n_posterior, prior=True)

def plot_b():
    model = Conj_Model(prior=stats.beta, hyperpara=[1, 3], likelikhood=stats.bernoulli, theta_0=.7,
                   n=250)  # define a model
    model_orange = Conj_Model(prior=stats.beta, hyperpara=[3, 1], likelikhood=stats.bernoulli, theta_0=.7,
                   n=250)  # define a model
    y = model.sample_data() #sample some data
    for k in [10, 50 , 100, 250]:
        fig, ax = plt.subplots(tight_layout=True)
        model.plot_analytic_posterior(y = y, ax=ax, n_posterior=[k], prior=True)
        model_orange.plot_analytic_posterior_dashed(y = y, ax=ax, n_posterior=[k], prior=True)
        plt.ylim([0,16])




if __name__ == '__main__':
    plot_a()
    plt.show()





