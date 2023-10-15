''' this module contains the class 'Model'
the objects of this class are bayesian, model, i.e. a posterior and a likelihood.
the method metropolis_hastings computes and plots the posterior numerically
some of the other methods are used in the method metroplis hastings '''

import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams['text.usetex'] = True

class Model:

    def __init__(self, prior, hyperpara, likelikhood, theta_0, n):
        self.prior = prior
        self.hyperpara = hyperpara
        self.likelikhood = likelikhood
        self.theta_0 = theta_0
        self.n = n

    '''
    this method samples data from the likelihood for a given theta_0 and returns a list containing the sample 
    '''
    def sample_data(self):
        y = []
        for i in range(self.n):
            y.append(self.likelikhood.rvs(self.theta_0))
        return y

    '''since for different model, different areas on the x-axis are of interest, we define a fct'''
    def x_lim(self):
        if self.prior == stats.beta and self.likelikhood == stats.bernoulli:
            return [0, 1]
        if self.prior == stats.gamma and self.likelikhood == stats.poisson:
            return [0, self.theta_0 * 1.5]
        else:
            return [self.theta_0 - 5, self.theta_0 + 5]  # this is a choice and can be modified

    ''' this method computes the likehlihood for iid models (the likelihood is a product of likelihoods)
    logp(y_{1:n}|theta) = log [prod_{i=1}^n p(y_i|theta)] = sum_{i=1}^n log p(y_i|theta)'''
    def prod_log_likelihood(self, y, theta):
        log_likelihood_theta = []
        for i in range(self.n):
            log_likelihood_theta.append(self.likelikhood.logpmf(y[i], theta))
        return sum(log_likelihood_theta)

    '''
    this method is the proposal distribution for the metropolis hastings algorithm
    it can be modified and suited for a given model
    for now it is only specified for the beta-bernoulli and gamma-poission model
    else the proposal is J = uniform(theta_k - r/2, theta_k + r/2, which might cause problems.')
    the parameter r can be varied to optimize 
    '''
    def proposal_distibution(self, theta_k, r):
        if self.prior == stats.beta:
            return stats.uniform.rvs(theta_k - r/2, r) % 1 #important: modular 1 since we dont want values outside of [0,1]
        if self.prior == stats.gamma:
            return stats.gamma.rvs(theta_k * r, loc = 0, scale = 1/r)
        else:
            return stats.uniform.rvs(theta_k - r/2, r)

    '''for computations the logarithm of proposal density function is usefull'''
    def log_J(self, theta, theta_star, r):
        if self.prior == stats.beta:
            return stats.uniform.logpdf(theta_star, theta - r/2, r) % 1 #modular 1 since we dont want values outside of [0,1]
        if self.prior == stats.gamma:
            return stats.gamma.logpdf(theta_star, theta * r, loc = 0, scale = 1/r)
        else:
            return stats.uniform.logpdf(theta_star, theta - r/2, r)

    '''this method computes the log (!) of the acceptance probabilty from the metropolis hastings algorithm '''
    def log_acceptance_prob(self, theta, theta_star, y, r):
        if self.prior == stats.beta and self.likelikhood == stats.bernoulli:
            return np.log( theta_star / theta ) * (sum(y) + self.hyperpara[0] - 1) \
                         + np.log((1 - theta_star) / (1 - theta)) * (self.n - sum(y) + self.hyperpara[1] - 1)
        if self.prior == stats.gamma and self.likelikhood == stats.poisson:
            return (self.hyperpara[1] + self.n ) * (theta - theta_star) + (self.hyperpara[0] + sum(y) - 1) * np.log(theta_star / theta)\
                    + stats.gamma.logpdf(theta, theta_star * r, loc = 0, scale = 1/r) - stats.gamma.logpdf(theta_star, theta * r,  loc = 0, scale = 1/r)
        else:
            return self.prod_log_likelihood(y, theta_star) + self.prior.logpdf(theta_star, self.hyperpara[0], self.hyperpara[1]) + (self.log_J(theta, theta_star, r)) \
                    - self.prod_log_likelihood(y, theta) - self.prior.logpdf(theta, self.hyperpara[0], self.hyperpara[1]) - (self.log_J(theta_star, theta, r))

    '''this method computes the posterior using metropolis hastings
    if Hist == True, we get the approximatied posterior in comparism to the analytic; if Hist == Flase we get a illustration of the Markov chain'''
    def metropolis_hastings(self, y, K, theta_1, r, Hist, N, ax):
        #plt.rcParams['text.usetex'] = True  # for LaTeX
        #fig, ax = plt.subplots(tight_layout=True)  # for subplots
        k = 0  #step
        rej_prop = 0  #to count rejected proposals
        theta = [theta_1]

        if Hist == False:
            plt.legend(handles=[Line2D([], [], marker='o', color='green', label='angenommen', linestyle='None'),
                                Line2D([], [], marker='o', color='r', label='abgelehnt', linestyle='None')], loc='upper left', fontsize=12)
            ax.set_xlabel(r'\textrm{Schritte} $k$', fontsize=20)
            ax.set_ylabel(r'$\vartheta_k$ \textrm{aus der Markov-Kette} ', fontsize=20)
            plt.ylim(self.x_lim())

        while k < K:
            theta_star = self.proposal_distibution(theta[k], r)  # draw theta* from proposal distribution
            xi = stats.uniform.rvs()  # uniform rv on (0,1)
            if np.log(xi) <= self.log_acceptance_prob(theta[k], theta_star, y, r):
                theta.append(theta_star)
                if Hist == False:
                    plt.plot([k], [theta_star], ".", color="green")  # green point when proposal accepted
            else:
                rej_prop += 1
                theta.append(theta[k])
                if Hist == False:
                    plt.plot([k], [theta[k]], ".", color="green")  # green point for the next step
                    plt.plot([k], [theta_star], ".", color="red")  # red point for the rejected proposal
            k += 1
        if Hist == True:

            ax.set_xlabel(r'\textrm{Parameter} $\vartheta$', fontsize=20)
            ax.set_ylabel(r'\textrm{Posterior} $p(\vartheta|y_{1:n})$', fontsize=20)
            plt.hist(theta, N, density=True, histtype='step', linewidth=1, color='k', label='Numerische Lösung mit MCMC')  # plot histogram for numerical solution using mcmc
            plt.ylim(0, plt.ylim()[1] * 1.05) #adjust y-lim s.t. the numerical sol fits (y-lim was suitable for analytic sol)
            plt.xlim(self.x_lim())
            plt.legend(loc='upper left', fontsize=12)

        print(f'for r={r}, from a total of {k} steps, {rej_prop} proposals were rejected')