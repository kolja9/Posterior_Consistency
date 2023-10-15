import scipy.stats as stats
from Conj_Model import Conj_Model
import matplotlib.pyplot as plt
import numpy as np

def plot():
    n_posterior = [5, 25, 100, 250, 500] #for these n the posterior will be plotted
    model = Conj_Model(prior = stats.gamma, hyperpara = [5, 2], likelikhood = stats.poisson, theta_0 = 4 , n = max(n_posterior))  # define a model
    y = model.sample_data() #sample some data
    fig, ax = plt.subplots(tight_layout=True)
    model.plot_analytic_posterior(y = y, n_posterior = n_posterior, prior = True, ax=ax)
    fig, ax = plt.subplots(tight_layout=True)
    model.plot_posterior_and_bernstein(y = y, ax=ax)

