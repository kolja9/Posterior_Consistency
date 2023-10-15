'''this module contains a function that plots 2 graphics of posteriors which have different priors
it illustrates the necessity of the firs condtion of schwartz theorem --- that theta_0 shall be included in the KL-Support of the prior
since this code was written before the class Model was created, it is not used (could be modified easily)'''

import matplotlib.pyplot as plt
import scipy.stats as stats
from colour import Color
import numpy as np
import scipy.integrate as integr

theta_0_values = [.3, .7]  #choose differnet values of the true parameter theta_0 to compare

plt.rcParams['text.usetex'] = True

alpha_prior = 2             #prior
beta_prior = 2

colors = list(Color("red").range_to(Color("green"), 6))  # create color cradient
colors = [color.rgb for color in colors]  # convert to RGB values since matplotlib doesnt support with coulor.color

def plot_subplot(theta_0):

    fig, ax = plt.subplots(tight_layout=True)

    if theta_0 < .5:
        n_posterior = [5, 25, 100, 250]

    elif theta_0 >= .5:
        n_posterior = [5, 25, 100, 250]

    Y = []
    count_heads = [0]

    for i in range(max(n_posterior)):
        Y.append(stats.bernoulli.rvs(theta_0))  #bernoulli RV
        count_heads.append(Y.count(1))

    x = np.linspace(0, .5, 10 ** 3)
    x_2 = np.linspace(.5, 1, 10 ** 3)

    ax.plot(x, x * 0 + 1, color='grey', label='Prior')  # plot prior
    ax.plot(x_2, x_2 * 0, color='grey')

    for i in range(len(n_posterior)):
        k = n_posterior[i]
        integral = integr.quad(lambda x: (x ** count_heads[k] * (1 - x) ** (k - count_heads[k])), 0, .5)[0] #integrate to get the normalizing factor from the bayes rule
        ax.plot(x, x ** count_heads[k] * (1 - x) ** (k - count_heads[k]) / integral, color=colors[i],
                label='n = ' + str(k) + ', k = ' + str(count_heads[k]))
        ax.plot(x_2, x_2 * 0, color=colors[i])

    plt.axvline(x = theta_0, color='k', linestyle='--')  # line at theta_0
    ax.set_xlabel(r'\textrm{Parameter} $\vartheta$', fontsize=20)
    ax.set_ylabel(r'\textrm{Posterior} $p(\vartheta|y_{1:n})$', fontsize=20)
    plt.xticks([0, theta_0, 1], [0, r'$\vartheta_0 = $' + str(theta_0), 1], color='k', fontsize=12)
    plt.xlim([0, 1])
    plt.ylim([0, 20])
    plt.gca().set_ylim(bottom=0)

    if theta_0 > .5:
        plt.legend(loc='upper left', fontsize=12)

    if theta_0 <= .5:
        plt.legend(loc='upper right', fontsize=12)


def plot():
    for theta_0 in theta_0_values:
        plot_subplot(theta_0)
    plt.show()

