import numpy as np
from sys import exit
from progress.bar import Bar
from revelio import periodograms as rper

def get_mean_frequency(frequencies):

    """
    @input (array): frequencies --> Array of frequencies of highest significance
                                    from each the Monte Carlo iterations.

    ----------------------------------------------------------------------------
    ----------------------------------------------------------------------------

    @output (float): f_hat --> Mean frequency calculated from /frequencies/.

    ----------------------------------------------------------------------------
    ----------------------------------------------------------------------------

    This function accepts an array
    """

    return (1./len(frequencies)) * frequencies.sum()


def get_variance_frequency(frequencies, mean_frequency):

    """
    @input (array): frequencies --> Array of frequencies of highest significance
                                    from each the Monte Carlo iterations.
    @input (float): mean_frequency --> The mean frequency used to calculate the
                                        variance.
    ----------------------------------------------------------------------------
    ----------------------------------------------------------------------------

    @output (float): var_hat --> Frequency variance calculated from /frequencies/.

    ----------------------------------------------------------------------------
    ----------------------------------------------------------------------------

    This function accepts an array
    """

    return (1./(len(frequencies)-1.)) * ((frequencies-mean_frequency)**2).sum()


def sample(obs, sigma, N_samples):

    """
    @input (array): obs --> observed fluxes/magnitudes.
    @input (array): sigma --> uncertainties on fluxes/magnitudes.
    @input (int): N_samples --> number of samples to draw for
                                Monte-Carlo simulation.

    ----------------------------------------------------------------------------
    ----------------------------------------------------------------------------

    @output (2D array) obs_mc: (N_samples, len(obs)) array of lightcurves
                             perturbed within the observational uncertainties.

    ----------------------------------------------------------------------------
    ----------------------------------------------------------------------------

    For each data point obs_i with corresponding uncertainty
    sigma_i, this function draws N_samples from a Normal distribution
    with N~(0,sigma_i) and returns an array of (N_samples, len(obs)) sampled
    lightcurves.
    """

    eps = np.array( [ np.random.normal(0., sigma_i, N_samples) for sigma_i in sigma ] ).T
    obs_mc = obs + eps

    return obs_mc


def get_periodogram(module, periodogram, times, obs, sigma=None, min=None,
                    max=None, use_frequency=True, oversample_factor=1., nbins=8,
                    phase_bins=10, mag_bins=5):
    """
    @input (module): module --> module that contains the periodogram.
    @input (str): periodogram --> name of the periodogram to be used.
    @input (array): times --> times of observations.
    @input (array): obs --> observed fluxes/magnitudes.
    @input (array): sigma --> uncertainties on fluxes/magnitudes.
    @input (float): min --> minimum frequency/period to calculate periodogram for.
    @input (float): max --> maximum frequency/period to calculate periodogram for.
    @input (boolean): use_frequency --> If True, sample periodogram in frequency.
                                        If False, sampler periodogram in period.
    @input (float): oversample_factor --> factor by which we increase the frequency
                                          step in the periodogram.
    @input (int): nbins --> ONLY USED WITH AOV PERIODOGRAM. Number of bins used
                            to calculate variance in lightcurves.
    @input (int): phase_bins --> ONLY USED WITH CONDITIONAL ENTROPY PERIODOGRAM.
                                 Number of phase bins used to calculate
                                 the periodogram.
    @input (int): mag_bins --> ONLY USED WITH CONDITIONAL ENTROPY PERIODOGRAM.
                                 Number of magnitude/flux bins used to calculate
                                 the periodogram.

    ----------------------------------------------------------------------------
    ----------------------------------------------------------------------------

    @output (array): nu --> 1D array of the frequencies at which the
                            periodogram is calculated.
    @output (array): theta --> 1D array of the periodogram.
    @output (float): nu_max --> frequency of most significant variability.

    ----------------------------------------------------------------------------
    ----------------------------------------------------------------------------

    This function calls a periodogram
    """
    periodogram_func = getattr(module, periodogram)
    if periodogram == 'aov':
        nu, theta = periodogram_func(times, obs, sigma=None, min=min,
                                     max=max, use_frequency=use_frequency,
                                     oversample_factor=oversample_factor,
                                     nbins=nbins)
        idx = np.argmax(theta)
        nu_max = nu[idx]
    elif periodogram == 'conditional_entropy':
        nu, theta = periodogram_func(times, obs, sigma=None, min=min, max=max,
                                     use_frequency=use_frequency,
                                     oversample_factor=oversample_factor,
                                     phase_bins=phase_bins, mag_bins=mag_bins)
        idx = np.argmin(theta)
        nu_max = nu[idx]
    else:
        raise ValueError('{} is not an accepted periodogram'.format(periodogram))
        exit()

    return nu, theta, nu_max

def run_monte_carlo(times, obs, sigma=None, min=None, max=None, use_frequency=True,
                    oversample_factor=1., nbins=8, phase_bins=10, mag_bins=5,
                    N_samples=100, periodogram='aov', snr_window=2.,
                    snr_range=False):

    """
    @input (array): times --> times of observations.
    @input (array): obs --> observed fluxes/magnitudes.
    @input (array): sigma --> uncertainties on fluxes/magnitudes.
    @input (float): min --> minimum frequency/period to calculate periodogram for.
    @input (float): max --> maximum frequency/period to calculate periodogram for.
    @input (boolean): use_frequency --> If True, sample periodogram in frequency.
                                        If False, sampler periodogram in period.
    @input (float): oversample_factor --> factor by which we increase the frequency
                                          step in the periodogram.
    @input (int): nbins --> ONLY USED WITH AOV PERIODOGRAM. Number of bins used
                            to calculate variance in lightcurves.
    @input (int): phase_bins --> ONLY USED WITH CONDITIONAL ENTROPY PERIODOGRAM.
                                 Number of phase bins used to calculate
                                 the periodogram.
    @input (int): mag_bins --> ONLY USED WITH CONDITIONAL ENTROPY PERIODOGRAM.
                                 Number of magnitude/flux bins used to calculate
                                 the periodogram.
    @input (int): N_samples --> number of samples to draw for
                                Monte-Carlo simulation.
    @input (str): periodogram --> algorithm to use for calculating
                                  the periodogram.

    ## TO BE INCORPORATED LATER
    @input (float): snr_window --> window in c/d over which we calculate
                                   the signal-to-noise ratio of a given
                                   peak.
    @input (boolean): snr_range --> if True, use a 5 c/d window between
                                    f~[ f_nyquist - 5, f_nyquist ] to calculate
                                    the noise level. If False, use snr_window.

    ----------------------------------------------------------------------------
    ----------------------------------------------------------------------------


    ----------------------------------------------------------------------------
    ----------------------------------------------------------------------------

    This routine takes a time series of observations /obs/, with associated
    uncertainties /sigma/, taken at times /times/ and calculates /N_samples/
    periodograms after perturbing the observations obs_i within the
    uncertainties sigma_i.

    We collect the highest signal-to-noise peak in each of the N_samples
    periodograms and calculate the resulting mean frequency and the weighted
    uncertainty of the mean frequency.

    We use either a moving window or a high-frequency range to calculate the
    noise level of the periodogram for the signal-to-noise calculations.

    -- NOTE --
    The /obs/ array is not mean/median subtracted before being passed to this
    function. You must substract the mean/medain of the observations before
    passing it to the periodogram, otherwise all of the power will go into
    the zeroeth frequency bin.
    """

    obs_ = obs-np.median(obs)

    ## First we calculate the nominal periodogram of the original, unperturbed
    ## observations and determine the frequency of highest significance.
    nu_nominal, theta_nominal, \
    nu_max_nominal = get_periodogram(rper, periodogram, times, obs_,
                                     sigma=None, min=min, max=max,
                                     use_frequency=True, oversample_factor=1.,
                                     nbins=8, phase_bins=10, mag_bins=5)

    ## Report the frequency of maximum variability
    print('The frequency of most significant variability from the nominal,')
    print('unperturbed data is: f={:.3f} c/d'.format(nu_max_nominal))
    print('or: \t p={:.5f} d \t p={:.5f} minutes'.format(1./nu_max_nominal,
                                                         1440./nu_max_nominal))

    ## Perform monte carlo simulations and run a periodogram on
    ## each perturbed lightcurve.
    ## This could take a while.
    print('\n We will now run {:d} Monte Carlo iterations'.format(N_samples))
    mc_samples = sample(obs_, sigma, N_samples)

    suffix = '%(percent).1f%% - %(eta)ds'
    mc_frequencies = []
    with Bar('Iterating', fill = '#', suffix=suffix, max=N_samples) as bar:
        for obs_ii in mc_samples:
            nu_ii, theta_ii, \
            nu_max_ii = get_periodogram(rper, periodogram, times, obs_ii,
                                        sigma=None, min=min, max=max,
                                        use_frequency=True, oversample_factor=1.,
                                        nbins=8, phase_bins=10, mag_bins=5)
            mc_frequencies.append(nu_max_ii)
            bar.next()

    ## Now we want to calculate the mean frequency and variance on the MC
    ## frequencies.
    mean_frequency = get_mean_frequency(mc_frequencies)
    var_frequency = get_variance_frequency(mc_frequencies)
    std_frequency = np.sqrt(var_frequency)

    print('The mean frequency calculated from {:i} Monte Carlo iteration is'.format(N_samples))
    print('f={:.3f} c/d'.format(mean_frequency))
    print('or: \t p={:.5f} d ; \t p={:.5f} minutes'.format(1./mean_frequency,
                                                         1440./mean_frequency))

    dt = 1./mean_frequency - 1./(mean_frequency-std_frequency)
    print('\n The uncertainty is: {:.3f} c/d'.format(std_frequency))
    print('or \t dt={:.5f} d ; \t p={:.5f} minutes')

    return mean_frequency, std_frequency
