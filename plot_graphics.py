#this script can plot any graphic contained in the bachelor thesis
#run the script and type in the number of the graphic you want to see!
#all the modules imported below should be included in the same folder as this one

import matplotlib.pyplot as plt
from beta_bernoulli import plot as plot1
from gamma_poi import plot as plot2
from mcmc import plot_beta_ber_markov as plot3
from mcmc import plot_beta_ber_hist as plot4
from mcmc import plot_gamma_poi_markov as plot5
from mcmc import plot_gamma_poi_hist as plot6
from bad_prior import plot as plot7
from mcmc import plot_2staged_bb_markov as plot8
from mcmc import plot_2staged_bb_hist as plot9
from mcmc import plot_beta_ber_hist_appendix as plot10
from mcmc import plot_beta_ber_hist_appendix2 as plot11
from mcmc import plot_gamma_poi_hist_appendix as plot12
from mcmc import plot_gamma_poi_hist_appendix2 as plot13
from beta_bernoulli import plot_a as plot14
from beta_bernoulli import plot_b as plot15

import numpy as np
np.random.seed(3)  # fixes RVs (remove if you want to draw new RVs)


def main():
    try:
        x = int(input('which graphic would you like to see? Please enter a number between 1 and 15 (or 0 to exit): '))
        if x == 0:
            pass

        elif 1 <= x <= 15:
            plot_function = globals()[f"plot{x}"]
            plot_function()
            plt.show()
        else:
            print('ERROR --- invalid input')

    except ValueError:
        print('ERROR --- invalid input')


if __name__ == "__main__":
    main()