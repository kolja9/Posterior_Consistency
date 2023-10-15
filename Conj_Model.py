'''this module contains a subclass of the superclass Model. This subclasses objects are conjugated models
since the parameter updates are a little different for each model, this code isnt working for all conjugated models but only for
beta-bernoulli and gamma-poisson. however, other models can be added'''

from Model import Model
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
from colour import Color

plt.rcParams['text.usetex'] = True  # for LaTeX

class Conj_Model(Model):

    ''' this function updates the parameters of the posterior
    it only works for models used in the bachelor thesis, but other models can be added easily'''
    def updated_parameters(self, y):
        alpha, beta = [self.hyperpara[0]], [self.hyperpara[1]]  # list for parameters of posterior
        if self.prior == stats.beta and self.likelikhood == stats.bernoulli:
            for j in range(len(y)):
                alpha.append(self.hyperpara[0] + sum(y[:j]))    #update parameters --- note that hyperpara[0] is alpha and ..[1] is beta
                beta.append(self.hyperpara[1] + j - sum(y[:j]))
        elif self.prior == stats.gamma and self.likelikhood == stats.poisson:
            for j in range(len(y)):
                alpha.append(self.hyperpara[0] + sum(y[:j]))   #update parameters --- note that hyperpara[0] is alpha and ..[1] is beta
                beta.append(self.hyperpara[1] + j)
        return alpha, beta

    ''' this function shall plot the analytic posterior.
    it only works for models used in the bachelor thesis, but other models can be added easily'''
    def plot_analytic_posterior(self, y, n_posterior, prior, ax):
        if len(n_posterior) == 1:
            colors = ['green']
        else:
            colors = list(Color("red").range_to(Color("green"), len(n_posterior)))  # create color cradient
            colors = [color.rgb for color in colors]  # convert to RGB values since matplotlib doesnt support with coulor.color
        x = np.linspace(self.x_lim()[0], self.x_lim()[1], 10 ** 3)

        if self.prior == stats.beta and self.likelikhood == stats.bernoulli:
            if prior == True:
                ax.plot(x, self.prior.pdf(x, self.hyperpara[0], self.hyperpara[1]), color='grey',
                    label='Prior')  # plot prior
                plt.axvline(x=self.theta_0, color='k', linestyle='--')  # line at theta_0

            for i in range(len(n_posterior)):
                k = n_posterior[i]
                if prior == False:
                    label_text = 'Analytische Lösung'

                else:
                    label_text = r'$n =$ ' + str(k) + r', $k =$ ' + str(self.updated_parameters(y)[0][k] - self.hyperpara[0])
                ax.plot(x, self.prior.pdf(x, self.updated_parameters(y)[0][k], self.updated_parameters(y)[1][k]), color=colors[i], label=label_text)
            plt.xticks([0, self.theta_0, 1], [0, r'$\vartheta_0 = $' + str(self.theta_0), 1], color='k', fontsize=12)

        elif self.prior == stats.gamma and self.likelikhood == stats.poisson:
            if prior == True:
                ax.plot(x, self.prior.pdf(x, self.hyperpara[0], loc = 0, scale = 1 / self.hyperpara[1]), color='grey',
                    label='Prior')  # plot prior
                plt.axvline(x=self.theta_0, color='k', linestyle='--')  # line at theta_0

            for i in range(len(n_posterior)):
                k = n_posterior[i]
                if prior == False:
                    label_text = 'Analytische Lösung'

                else:
                    label_text = r'$n =$ ' + str(k) + r', $k =$ ' + str(self.updated_parameters(y)[0][k] - self.hyperpara[0])
                ax.plot(x, self.prior.pdf(x, self.updated_parameters(y)[0][k], loc = 0, scale = 1 / self.updated_parameters(y)[1][k]), color=colors[i], label = label_text)
                plt.xticks([0, 1, 2, 3, self.theta_0, 5, 6], [0, 1, 2, 3, r'$\vartheta_0 = $' + str(self.theta_0), 5, 6], color='k', fontsize=12)

        else:
            print('warning: no analytic solution available')

        ax.set_xlabel(r'\textrm{Parameter} $\vartheta$', fontsize=20)
        ax.set_ylabel(r'\textrm{Posterior} $p(\vartheta|y_{1:n})$', fontsize=20)
        plt.legend(loc='upper left', fontsize=12)
        plt.xlim(self.x_lim())
        plt.gca().set_ylim(bottom=0)


    def plot_posterior_and_bernstein(self, y, ax):
        x = np.linspace(self.x_lim()[0], self.x_lim()[1], 10 ** 3)
        self.plot_analytic_posterior(y, [self.n], prior = False, ax = ax)

        if self.prior == stats.beta and self.likelikhood == stats.bernoulli:
            ax.plot(x, stats.norm.pdf(x, loc=(self.updated_parameters(y)[0][self.n]-self.hyperpara[0]) / self.n,
                    scale=np.sqrt((self.theta_0 * (1 - self.theta_0)) / self.n)), color='k', label='Normalverteilung aus Bernstein-von Mises')

        elif self.prior == stats.gamma and self.likelikhood == stats.poisson:
            ax.plot(x, stats.norm.pdf(x, loc=(self.updated_parameters(y)[0][self.n]-self.hyperpara[0]) / self.n,
                    scale=np.sqrt(self.theta_0 ** 2 / (self.n * 4))), color='k', label='Normalverteilung aus Bernstein-von Mises')

        plt.axvline(x=self.theta_0, color='k', linestyle='--')  # line at theta_0
        ax.set_xlabel(r'\textrm{Parameter} $\vartheta$', fontsize=20)
        ax.set_ylabel(r'\textrm{Posterior} $p(\vartheta|y_{1:n})$', fontsize=20)
        plt.legend(loc='upper left', fontsize=12)
        plt.xlim(self.x_lim())
        plt.gca().set_ylim(bottom=0)

    '''the metropolis hastings method for conjugated models also plots the analytic solution'''
    def metropolis_hastings(self, y, K, theta_1, r, Hist, N, ax):
        super().metropolis_hastings(y, K, theta_1, r, Hist, N, ax)
        if Hist == True:
            self.plot_analytic_posterior(y=y, n_posterior=[self.n], prior=False, ax = ax) #for conjugated model the analytic posterior can be plotted aswell

    '''this method plots multiple posteriors in different colors and the normal distr from bernstein von mises in comparism (same color but dashed)
    of course, this function is simular to others (DRY is violated) --- this could be improved, but the result would be the same'''
    def plot_posterior_and_bernstein_dashed(self, y, n_posterior, prior, ax):
        if len(n_posterior) == 1:
            colors = ['green']
        else:
            colors = list(Color("red").range_to(Color("green"), len(n_posterior)))  # create color cradient
            colors = [color.rgb for color in colors]  # convert to RGB values since matplotlib doesnt support with coulor.color
        x = np.linspace(self.x_lim()[0], self.x_lim()[1], 10 ** 3)

        if self.prior == stats.beta and self.likelikhood == stats.bernoulli:
            if prior == True:
                ax.plot(x, self.prior.pdf(x, self.hyperpara[0], self.hyperpara[1]), color='grey',
                    label='Prior')  # plot prior

            for i in range(len(n_posterior)):
                k = n_posterior[i]
                if prior == False:
                    label_text = 'Analytische Lösung'

                else:
                    label_text = r'$n =$ ' + str(k) + r', $k =$ ' + str(self.updated_parameters(y)[0][k] - self.hyperpara[0])

                ax.plot(x, self.prior.pdf(x, self.updated_parameters(y)[0][k], self.updated_parameters(y)[1][k]), color=colors[i], label=label_text)
                ax.plot(x, stats.norm.pdf(x, loc=(self.updated_parameters(y)[0][k]-self.hyperpara[0]) / k,
                                          scale=np.sqrt((self.theta_0 * (1 - self.theta_0)) / k)), color=colors[i], linestyle='dashed')

        elif self.prior == stats.gamma and self.likelikhood == stats.poisson:
            if prior == True:
                ax.plot(x, self.prior.pdf(x, self.hyperpara[0], loc = 0, scale = 1 / self.hyperpara[1]), color='grey',
                    label='Prior')  # plot prior

            for i in range(len(n_posterior)):
                k = n_posterior[i]
                if prior == False:
                    label_text = 'Analytische Lösung'

                else:
                    label_text = r'$n =$ ' + str(k) + r', $k =$ ' + str(self.updated_parameters(y)[0][k] - self.hyperpara[0])
                    ax.plot(x, self.prior.pdf(x, self.updated_parameters(y)[0][k], loc = 0, scale = 1 / self.updated_parameters(y)[1][k]), color=colors[i], label = label_text)
                    ax.plot(x,
                            stats.norm.pdf(x, loc=(self.updated_parameters(y)[0][k] - self.hyperpara[0]) / k,
                                           scale=np.sqrt(self.theta_0 ** 2 / (k * 4))), color=colors[i], linestyle='dashed')

        else:
            print('warning: no analytic solution available')

        ax.set_xlabel(r'\textrm{Parameter} $\vartheta$', fontsize=20)
        ax.set_ylabel(r'\textrm{Posterior} $p(\vartheta|y_{1:n})$', fontsize=20)
        plt.legend(loc='upper left', fontsize=12)
        plt.xlim(self.x_lim())
        plt.gca().set_ylim(bottom=0)

    '''this method plots multiple posteriors in different colors with a dashed line
    of course, this function is simular to others (DRY is violated) --- this could be improved, but the result would be the same'''
    def plot_analytic_posterior_dashed(self, y, n_posterior, prior, ax):
        if len(n_posterior) == 1:
            colors = ['green']
        else:
            colors = list(Color("red").range_to(Color("green"), len(n_posterior)))  # create color cradient
            colors = [color.rgb for color in
                      colors]  # convert to RGB values since matplotlib doesnt support with coulor.color

        x = np.linspace(self.x_lim()[0], self.x_lim()[1], 10 ** 3)
        ax.plot(x, self.prior.pdf(x, self.hyperpara[0], self.hyperpara[1]), color='grey',
                label='Prior', linestyle='dashdot')  # plot prior
        if self.prior == stats.beta and self.likelikhood == stats.bernoulli:
            plt.axvline(x=self.theta_0, color='k', linestyle='--')  # line at theta_0

            for i in range(len(n_posterior)):
                k = n_posterior[i]
                if prior == False:
                    label_text = 'Analytische Lösung'

                else:
                    label_text = r'$n =$ ' + str(k) + r', $k =$ ' + str(
                        self.updated_parameters(y)[0][k] - self.hyperpara[0])
                ax.plot(x, self.prior.pdf(x, self.updated_parameters(y)[0][k], self.updated_parameters(y)[1][k]),
                        color=colors[i], label=label_text, linestyle='dashdot')
            plt.xticks([0, self.theta_0, 1], [0, r'$\vartheta_0 = $' + str(self.theta_0), 1], color='k',
                       fontsize=12)

        elif self.prior == stats.gamma and self.likelikhood == stats.poisson:
            plt.axvline(x=self.theta_0, color='k', linestyle='--')  # line at theta_0
            ax.plot(x, self.prior.pdf(x, self.hyperpara[0], loc=0, scale=1 / self.hyperpara[1]), color='grey',
                    label='Prior',linestyle='dashdot')  # plot prior
            for i in range(len(n_posterior)):
                k = n_posterior[i]
                if prior == False:
                    label_text = 'Analytische Lösung'

                else:
                    label_text = r'$n =$ ' + str(k) + r', $k =$ ' + str(
                        self.updated_parameters(y)[0][k] - self.hyperpara[0])
                ax.plot(x, self.prior.pdf(x, self.updated_parameters(y)[0][k], loc=0,
                                          scale=1 / self.updated_parameters(y)[1][k]), color=colors[i],
                        label=label_text, linestyle='dashed')
                plt.xticks([0, 1, 2, 3, self.theta_0, 5, 6],
                           [0, 1, 2, 3, r'$\vartheta_0 = $' + str(self.theta_0), 5, 6], color='k', fontsize=12)

        else:
            print('warning: no analytic solution available')

        ax.set_xlabel(r'\textrm{Parameter} $\vartheta$', fontsize=20)
        ax.set_ylabel(r'\textrm{Posterior} $p(\vartheta|y_{1:n})$', fontsize=20)
        plt.xlim(self.x_lim())
        plt.ylim(0, plt.ylim()[1] * 1.5)  # adjust y-lim s.t. the numerical sol fits (y-lim was suitable for analytic sol)

