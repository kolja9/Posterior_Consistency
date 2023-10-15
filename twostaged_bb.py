'''this script contains the function metropolis hastings that perforoms the metropolis hastings algorithm for the two staged beta bernoulli modell
the class Model is not used since its objects are one-dim. models. However this could be modified.'''

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import scipy.special as special
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D
import scipy.integrate as integr

plt.rcParams['text.usetex'] = True

n_coins = 10**3 #number of coins
n_flips = 25    #number of flips per coin
alpha_0 = 2      #prior
beta_0 = 2

def sample_ber_data(alpha_0, beta_0, n_coins, n_flips):
    y = []
    y_coins = []
    for i in range(n_coins):
        y_coins.append(stats.beta.rvs(alpha_0, beta_0))
        y_flips = []
        for j in range(n_flips):
            y_flips.append(stats.bernoulli.rvs(y_coins[i]))
        y.append(y_flips)
    return y

'''proposal distribution for metropolos hastins algo --- here it is a exponential or a gamma distribution 
with previos theta_k as  expected value however, optimimazition is possible'''

def proposal_distibution(alpha_k, beta_k, r):
    alpha_star = stats.gamma.rvs(alpha_k * r, scale = 1/r)
    beta_star = stats.gamma.rvs(beta_k * r, scale = 1/r)
    return alpha_star, beta_star

def log_acceptance_prob(y, alpha, beta, alpha_star, beta_star, r):
    sum_logs = 0
    for i in range(n_coins):
        sum_logs += np.log(special.beta(alpha_star + sum(y[i]), beta_star + n_flips - sum(y[i])) / special.beta(alpha + sum(y[i]), beta + n_flips - sum(y[i])))
    return n_coins * np.log(special.beta(alpha, beta) / special.beta(alpha_star, beta_star)) \
           + sum_logs + alpha + beta - alpha_star - beta_star\
           + stats.gamma.logpdf(alpha_star, alpha * r, scale = 1/r) - stats.gamma.logpdf(alpha, alpha_star * r, scale = 1/r)\
           + stats.gamma.logpdf(beta_star, beta * r, scale = 1/r) - stats.gamma.logpdf(beta, beta_star * r, scale = 1/r)

'''metropolis hastings
Hist is boolean and states whether the histogram is showed (true) or the markov chain (false)
markov_alpha is also boolean and states whether the alpha components from the markov chain are shown (true) or the beta (false) (note that if hist==true, the variable markov_alpha has nor impact)'''
def metropolis_hastings(y, K, alpha_1, beta_1, r, Hist, markov_alpha, N, ax):    #if Hist == True, we get the approximatied posterior in comparism to the analytic; if Hist == Flase we get a illustration of the Markov chain
    k = 0 #to count accepted steps
    rej_prop = 0 #to count the proposals
    alpha = [alpha_1] #list for accepted points
    alpha_rej = [] #list for rejected points
    beta = [beta_1]
    beta_rej = []
    while k < K:
        alpha_star, beta_star = proposal_distibution(alpha[k], beta[k], r)[0], proposal_distibution(alpha[k], beta[k], r)[1]    #draw theta* from proposal distribution
        xi = stats.uniform.rvs()      #uniform rv on (0,1)
        if np.log(xi) <= log_acceptance_prob(y, alpha[k], beta[k], alpha_star, beta_star, r):
            alpha.append(alpha_star)
            beta.append(beta_star)
            if Hist == False and markov_alpha == True:
                plt.plot([k], [alpha_star], ".", color="green")  # green point when step accepted
            if Hist == False and markov_alpha == False:
                plt.plot([k], [beta_star], ".", color="green")  # green point when step accepted
        else:
            rej_prop += 1
            alpha.append(alpha[k])
            beta.append(beta[k])
            alpha_rej.append(alpha_star)
            beta_rej.append(beta_star)
            if Hist == False and markov_alpha == True:
                plt.plot([k], [alpha[k]], ".", color="green")  # green point for the next step
                plt.plot([k], [alpha_star], ".", color="red")  # red point when propsal rejected
            if Hist == False and markov_alpha == False:
                plt.plot([k], [beta[k]], ".", color="green")  # green point for the next step
                plt.plot([k], [beta_star], ".", color="red")  # red point when proposal rejected
        k += 1

    if Hist == False:
        if markov_alpha == True:
            ax.set_ylabel(r'$\alpha_k$ \textrm{aus der Markov-Kette}', fontsize=20)
        else:
            ax.set_ylabel(r'$\beta_k$ \textrm{aus der Markov-Kette}', fontsize=20)
        plt.legend(handles=[Line2D([], [], marker='o', color='green', label='angenommen', linestyle='None'),
                            Line2D([], [], marker='o', color='r', label='abgelehnt', linestyle='None')],
                   loc='upper left', fontsize=12)
        ax.set_xlabel(r'\textrm{Schritte} $k$', fontsize=20)
        plt.ylim([1,3])

    if Hist == True:
        #formating scatter plot
        ax.scatter(alpha_rej, beta_rej, 5, c='red', marker='.')
        ax.scatter(alpha, beta, 5, c = 'green', marker = '.')
        ax.set_aspect(1.)
        divider = make_axes_locatable(ax)
        ax_histx = divider.append_axes("top", 1.2, pad=.3, sharex=ax)
        ax_histy = divider.append_axes("right", 1.2, pad=.3, sharey=ax)

        ax_histx.hist(alpha, N, density=True, histtype='step', linewidth=1, color='k',
                 label='Numerische Lösung mit MCMC')
        ax_histy.hist(beta, N, density=True, histtype='step', linewidth=1, color='k',
                 label='Numerische Lösung mit MCMC', orientation='horizontal')

        ax_histx.text(-0.1, 0.5, r'$p(\alpha|y_{1:n})$', transform=ax_histx.transAxes,
                      ha='right', va='center', rotation=90, size = 18) #additional label
        ax_histy.text(0.5, -0.1, r'$p(\beta|y_{1:n})$', transform=ax_histy.transAxes,
                      ha='center', va='top', size = 18) #additional label

        ax.set_xlabel(r'$\alpha$', fontsize=20)
        ax.set_ylabel(r'$\beta$', fontsize=20)
    print(f'mean of alpha in the numeric solution: {np.mean(alpha)}')
    print(f'mean of beta in the numeric solution: {np.mean(beta)}')
    print(f'from a total of {k} steps, {rej_prop} proposals were rejected')

    #compute KL-divergence
    d_KL = integr.quad(lambda x: (stats.beta.pdf(x, alpha_0, beta_0) * np.log(
        stats.beta.pdf(x, alpha_0, beta_0) / stats.beta.pdf(x, np.mean(alpha), np.mean(beta)))), 0, 1)
    print(f'the KL-divergence of the true and the approximated beta-distribution is {d_KL[0]} (with an error of {d_KL[1]}).')


