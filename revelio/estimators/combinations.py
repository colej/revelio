import numpy as np
from sys import exit
from progress.bar import Bar
from revelio import periodograms as rper

"""
This file contains functions to combine frequencies obtained
from data taken in multiple filters into a single, weighted
frequency of maximum variability estimate.
"""

def get_weights_equal(N_filters):
    """
    @input (int): N_filters --> total number of filters to be combined

    ----------------------------------------------------------------------------
    ----------------------------------------------------------------------------

    @output (array): weights --> weight per filter

    ----------------------------------------------------------------------------
    ----------------------------------------------------------------------------

    This function returns weights assuming that all filters have the same
    influence; i.e. the weight for all filters f is: w_f = 1/N_filters
    """

    return np.array([1./N_filters for ii in range(N_filters)])


def get_weights_n_obs(n_obs):
    """
    @input (array): N_filters --> array containing number of observations per
                                  filter.

    ----------------------------------------------------------------------------
    ----------------------------------------------------------------------------

    @output (array): weights --> weight per filter

    ----------------------------------------------------------------------------
    ----------------------------------------------------------------------------

    This function returns weights according to the total number of data npoints
    obtained in that filter; i.e. the weight for a filter f is: w_f = n_obs_f/n_obs_total
    """
    total_obs = np.sum(n_obs)

    return np.array([n_obs[ii]/total_obs for ii in range(len(n_obs))])


def get_weights_variance(variances):

    """
    @input (float): variance --> variance on the frequency estimate for a given
                                 filter

    ----------------------------------------------------------------------------
    ----------------------------------------------------------------------------

    @output (float): weights --> weight per filter

    ----------------------------------------------------------------------------
    ----------------------------------------------------------------------------

    This function returns a weight for the filter following optimality theory
    such that the weight for a given filter f is: w_f = 1/(sigma_f)^2
    """

    return 1./variances


def get_combined_frequency(frequencies, variances, n_obs=None, type_weights='equal'):
    """
    @input (array): frequencies --> frequency estimates per filter
    @input (array): variances --> variance estimates per filter
    @input (list/array): n_obs --> number of observations per filter. Only to be
                                  used if type_weights == 'nobs'
    @input (str): type_weights --> the method for determining the weights per filter
    ----------------------------------------------------------------------------
    ----------------------------------------------------------------------------

    @output (float): combined_frequency --> weighted combined frequency across
                                            all filters
    @output (float): combined_variance --> weighted combined variance across
                                            all filters

    ----------------------------------------------------------------------------
    ----------------------------------------------------------------------------

    This function takes the frequencies and their squared uncertainties (variances)
    obtained per filter and combines them according to their variances.
    """

    ## Determine the weights
    ## We can set the weight of a filter according to the variance of the
    ## frequency estimated from the Monte Carlo iterations.
    if type_weights=='variance':
        weights = get_weights_variance(variances)

    ## Or we can give wights according to the number of observations per filter
    elif type_weights=='n_obs':
        weights = get_weights_n_obs(n_obs)

    ## Otherwise, we can give equal weight to each filter, we use "equal"
    else:
        weights = get_weights_equal(len(frequencies))

    ## we calculate the combined frequency according to:
    ## nu_hat = [ sum_f->F ( weights_f * frequencies_f) ] / [ sum_f->F ( weights_f ) ]
    numerator = weights * frequencies

    combined_frequency = numerator.sum()/weights.sum()
    combined_variance = 1./weights.sum()

    return combined_frequency, combined_variance
