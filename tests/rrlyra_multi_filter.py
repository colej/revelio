import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import revelio.periodograms.aov as aov
import revelio.periodograms.conditional_entropy as ce

import revelio.estimators.combinations as rec

from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

plt.rcParams.update({
    "text.usetex": True,
        "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})

plt.rcParams.update({
    "pgf.rcfonts": False,
    "pgf.texsystem": "pdflatex",
    "pgf.preamble": "\n".join([
         r"\usepackage{amsmath}",
         r"\usepackage[utf8x]{inputenc}",
         r"\usepackage[T1]{fontenc}",
         r"\usepackage{cmbright}",
    ]),
})

plt.rcParams['xtick.labelsize']=18
plt.rcParams['ytick.labelsize']=18


def run_aov(times, signal, sigma,
            min=0.1, max=5,
            oversample_factor=3., nbins=10):

    nu, theta = aov.aov_periodogram( times, signal, sigma=sigma, min=min,
                                     max=max, oversample_factor=oversample_factor,
                                     nbins=nbins)

    ## We want to identify the frequency of maximum variability
    nu_max = nu[np.argmax(theta)]

    return nu, theta, nu_max


if __name__ == '__main__':

    ## Load in the data for our example
    df = pd.read_csv('rrlyra_example.csv')

    ## In this example, we want to use data from multiple
    ## filters -- first, we'll look at the data from the
    ## u, q, and i bands as these have the most data.

    ## Lets extract the data per filter and put them all in
    ## their own dataframe

    ## u-band
    dfu    = df.loc[df['Filter']=='u']
    utimes = np.array(dfu['MJD'].tolist())
    uflux  = np.array(dfu['Flux [μJy]'].tolist())
    uferr  = np.array(dfu['Fluxerr [μJy]'].tolist())

    ## q-band
    dfq    = df.loc[df['Filter']=='q']
    qtimes = np.array(dfq['MJD'].tolist())
    qflux  = np.array(dfq['Flux [μJy]'].tolist())
    qferr  = np.array(dfq['Fluxerr [μJy]'].tolist())

    ## i-band
    dfi    = df.loc[df['Filter']=='i']
    itimes = np.array(dfi['MJD'].tolist())
    iflux  = np.array(dfi['Flux [μJy]'].tolist())
    iferr  = np.array(dfi['Fluxerr [μJy]'].tolist())


    ## First, we'll use the Analysis of Variance algorithm
    ## to search for periodic signals in the u, q, and i bands
    nu_aov_u, theta_aov_u, nu_max_aov_u = run_aov( utimes, uflux, uferr)
    nu_aov_q, theta_aov_q, nu_max_aov_q = run_aov( qtimes, qflux, qferr)
    nu_aov_i, theta_aov_i, nu_max_aov_i = run_aov( itimes, iflux, iferr)


    ## Now lets plot the periodograms and the phase folded lightcurves on
    ## the frequency of most significant variability for each filter.
    fig_aov, axes_aov = plt.subplots(2,3, num=1, figsize=(12,8))
    axes_aov[0][0].plot(nu_aov_u, theta_aov_u)
    axes_aov[0][0].set_xlabel(r'${\rm Frequency~[d^{-1}]}$', fontsize=18)
    axes_aov[0][0].set_ylabel(r'${\Theta}$', fontsize=18)

    axes_aov[0][1].plot(nu_aov_q, theta_aov_q)
    axes_aov[0][1].set_xlabel(r'${\rm Frequency~[d^{-1}]}$', fontsize=18)

    axes_aov[0][2].plot(nu_aov_i, theta_aov_i)
    axes_aov[0][2].set_xlabel(r'${\rm Frequency~[d^{-1}]}$', fontsize=18)


    axes_aov[1][0].errorbar((utimes*nu_max_aov_u)%1., uflux, yerr=uferr, ls='',
                         marker='x', ms=1, label=r'${\rm u-band}$', color='dodgerblue')
    axes_aov[1][0].set_xlabel(r'${\rm Phase}$', fontsize=18)
    axes_aov[1][0].set_ylabel(r'${\rm Flux}$', fontsize=18)

    axes_aov[1][1].errorbar((qtimes*nu_max_aov_q)%1., qflux, yerr=qferr, ls='',
                         marker='x', ms=1, label=r'${\rm q-band}$', color='darkorange')
    axes_aov[1][1].set_xlabel(r'${\rm Phase}$', fontsize=18)
    axes_aov[1][2].errorbar((itimes*nu_max_aov_i)%1., iflux, yerr=iferr, ls='',
                         marker='x', ms=1, label=r'${\rm i-band}$', color='firebrick')
    axes_aov[1][2].set_xlabel(r'${\rm Phase}$', fontsize=18)

    fig_aov.tight_layout()
    plt.show()

    print('The AoV periodogram found the frequency of variability to be:')
    print('\t u-band')
    print('\t\t f={:.5f} c/d'.format(nu_max_aov_u))
    print('\t\t p={:.5f} d'.format(1./nu_max_aov_u))
    print('\t q-band')
    print('\t\t f={:.5f} c/d'.format(nu_max_aov_q))
    print('\t\t p={:.5f} d'.format(1./nu_max_aov_q))
    print('\t i-band')
    print('\t\t f={:.5f} c/d'.format(nu_max_aov_i))
    print('\t\t p={:.5f} d'.format(1./nu_max_aov_i))

    ## Now lets combine the frequencies identified per filter!
    ## First, we'll see what the combined frequency is giving equal weight to
    ## all filters
    frequencies = [nu_max_aov_u, nu_max_aov_q, nu_max_aov_i]
    n_obs = [len(utimes), len(qtimes), len(itimes)]
    variances = [1.,1.,1.]
    nu_max_combined, \
    var_combined = rec.get_combined_frequency(frequencies, variances,
                                              n_obs=n_obs, type_weights='equal')

    print('The combined frequency assuming all filters have the same weight is:')
    print('\t\t f={:.7f} c/d'.format(nu_max_combined))

    nu_max_combined, \
    var_combined = rec.get_combined_frequency(frequencies, variances,
                                              n_obs=n_obs, type_weights='n_obs')

    print('The combined frequency assuming the filters are weighted by the number of observations is:')
    print('\t\t f={:.7f} c/d'.format(nu_max_combined))

    ## You can of course do the same thing for the conditional entropy periodogram,
    ## but we'll leave that up to you!
