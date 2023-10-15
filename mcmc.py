'''this script contatins functions to plot the markov chains and histogramms from the metropolis-hastings-algorithm
other functions can be defined easily for different models in the same way as below '''


from Conj_Model import Conj_Model
import scipy.stats as stats
import matplotlib.pyplot as plt
import twostaged_bb
import numpy as np

def plot_beta_ber_markov(r_list = [.01, 1, .1, .5], theta_0=.7,hyperpara=[2,2], n = 100, K = 10**3, theta_1=.5):
    model = Conj_Model(prior = stats.beta, hyperpara = hyperpara, likelikhood=stats.bernoulli, theta_0 = theta_0, n = n)
    y = model.sample_data()
    for r in r_list:
        fig, ax = plt.subplots(tight_layout=True)
        model.metropolis_hastings(y, K=K, theta_1=theta_1, r=r, Hist=False, N=None, ax = ax)

def plot_beta_ber_hist(K_list = [10**3, 10**4], theta_0=.7,hyperpara=[2,2], n = 100, r=.1, theta_1 = .5):
    model = Conj_Model(prior=stats.beta, hyperpara=hyperpara, likelikhood=stats.bernoulli, theta_0=theta_0, n=n)
    y = model.sample_data()
    for K in K_list:
        fig, ax = plt.subplots(tight_layout=True)
        model.metropolis_hastings(y = y, K = K, theta_1 = theta_1, r = r, Hist = True, N = int((np.log10(K) - 1) * 10), ax = ax) #use metropolis-hastings for different K and plot histogram

def plot_beta_ber_hist_appendix(r_list = [.1,.5], theta_0=.7,hyperpara=[2,2], n = 100, K=10**3, theta_1 = .5):
    model = Conj_Model(prior=stats.beta, hyperpara=hyperpara, likelikhood=stats.bernoulli, theta_0=theta_0, n=n)
    y = model.sample_data()
    for r in r_list:
        fig, ax = plt.subplots(tight_layout=True)
        model.metropolis_hastings(y = y, K = K, theta_1 = theta_1, r = r, Hist = True, N = int((np.log10(K) - 1) * 10), ax = ax) #use metropolis-hastings for different K and plot histogram

def plot_beta_ber_hist_appendix2(r_list = [.1,.5], theta_0=.7,hyperpara=[2,2], n = 100, K=10**4, theta_1 = .5):
    model = Conj_Model(prior=stats.beta, hyperpara=hyperpara, likelikhood=stats.bernoulli, theta_0=theta_0, n=n)
    y = model.sample_data()
    for r in r_list:
        fig, ax = plt.subplots(tight_layout=True)
        model.metropolis_hastings(y = y, K = K, theta_1 = theta_1, r = r, Hist = True, N = int((np.log10(K) - 1) * 10), ax = ax) #use metropolis-hastings for different K and plot histogram


def plot_gamma_poi_markov(r_list = [1, 10, 100, 1000], theta_0=4,hyperpara=[5,2], n = 100, K = 10**3, theta_1 = 3):
    model = Conj_Model(prior = stats.gamma, hyperpara = hyperpara, likelikhood=stats.poisson, theta_0 = theta_0, n = n)
    y = model.sample_data()
    for r in r_list:
        fig, ax = plt.subplots(tight_layout=True)
        model.metropolis_hastings(y, K=K, theta_1=theta_1, r=r, Hist=False, N=None, ax = ax)

def plot_gamma_poi_hist(K_list = [10**3, 10**4], theta_0=4,hyperpara=[5,2], n = 100, r = 100, theta_1=3):
    model = Conj_Model(prior=stats.gamma, hyperpara=hyperpara, likelikhood=stats.poisson, theta_0=theta_0, n=n)
    y = model.sample_data()
    for K in K_list:
        fig, ax = plt.subplots(tight_layout=True)
        model.metropolis_hastings(y = y, K = K, theta_1 = theta_1, r = r, Hist = True, N = int((np.log10(K) - 1) * 10), ax = ax) #use metropolis-hastings for different K and plot histogram

def plot_gamma_poi_hist_appendix(r_list = [10,100], theta_0=4,hyperpara=[5,2], n = 100, K=10**3, theta_1 = 3):
    model = Conj_Model(prior=stats.gamma, hyperpara=hyperpara, likelikhood=stats.poisson, theta_0=theta_0, n=n)
    y = model.sample_data()
    for r in r_list:
        fig, ax = plt.subplots(tight_layout=True)
        model.metropolis_hastings(y = y, K = K, theta_1 = theta_1, r = r, Hist = True, N = int((np.log10(K) - 1) * 10), ax = ax) #use metropolis-hastings for different K and plot histogram

def plot_gamma_poi_hist_appendix2(r_list = [10,100], theta_0=4,hyperpara=[5,2], n = 100, K=10**4, theta_1 = 3):
    model = Conj_Model(prior=stats.gamma, hyperpara=hyperpara, likelikhood=stats.poisson, theta_0=theta_0, n=n)
    y = model.sample_data()
    for r in r_list:
        fig, ax = plt.subplots(tight_layout=True)
        model.metropolis_hastings(y = y, K = K, theta_1 = theta_1, r = r, Hist = True, N = int((np.log10(K) - 1) * 10), ax = ax) #use metropolis-hastings for different K and plot histogram


def plot_2staged_bb_hist(n_coins = 10**3, n_flips = 25, alpha_0 = 2, beta_0 = 2, K = 10**4, alpha_1 = 1.5, beta_1 = 1.5, r=1000):
    fig, ax = plt.subplots(tight_layout=True)
    y = twostaged_bb.sample_ber_data(n_coins = n_coins, n_flips = n_flips, alpha_0 = alpha_0, beta_0 = beta_0)
    twostaged_bb.metropolis_hastings(y, K = K, alpha_1 = alpha_1, beta_1 = beta_1, r=r, Hist = True, markov_alpha=None, N=20, ax = ax)

def plot_2staged_bb_markov(n_coins=10 ** 3, n_flips=25, alpha_0=2, beta_0=2, K=10 ** 4, alpha_1=1.5, beta_1=1.5, r=1000):
    y = twostaged_bb.sample_ber_data(n_coins=n_coins, n_flips=n_flips, alpha_0=alpha_0, beta_0=beta_0)
    fig, ax = plt.subplots(tight_layout=True)
    twostaged_bb.metropolis_hastings(y, K=K, alpha_1=alpha_1, beta_1=beta_1, r=r, Hist=False, markov_alpha=True, N=None, ax=ax)
    fig, ax = plt.subplots(tight_layout=True)
    twostaged_bb.metropolis_hastings(y, K=K, alpha_1=alpha_1, beta_1=beta_1, r=r, Hist=False, markov_alpha=False, N=None, ax=ax)

