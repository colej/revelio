import numpy as np
from sys import exit
from progress.bar import Bar
from revelio import periodograms as rper

"""
This file contains functions to combine frequencies obtained
from data taken in multiple filters into a single, weighted
frequency of maximum variability estimate.
"""

def get_weights_simple(N_filters):
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


def get_weights_per_filter(variances):

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

    return 1./variance


def get_combined_frequency(frequencies, variances, simple_weights=False):
    """
    @input (array): frequencies --> frequency estimates per filter
    @input (array): variances --> variance estimates per filter

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
    ## If we want to give equal weight to each filter, we use "simple_weights"
    if simple_weights:
        weights = get_weights_simple(len(frequencies))

    ## Otherwise, we set the weight of a filter according to the variance of the
    ## frequency estimated from the Monte Carlo iterations.
    else:
        weights = get_weights_per_filter(variances)

    ## we calculate the combined frequency according to:
    ## nu_hat = [ sum_f->F ( weights_f * frequencies_f) ] / [ sum_f->F ( weights_f ) ]
    numerator = weights * frequencies

    combined_frequency = numerator.sum()/weights.sum()
    combined_variance = 1./weights.sum()

    return combined_frequency, combined_variance
