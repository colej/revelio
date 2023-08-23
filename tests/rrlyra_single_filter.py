import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import revelio.periodograms.aov as aov
import revelio.periodograms.conditional_entropy as ce

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


if __name__ == '__main__':

    ## Load in the data for our example
    df = pd.read_csv('rrlyra_example.csv')

    ## In this example, we want to use only the data from
    ## the MeerLICHT q-band, so lets extract it and put it in
    ## a new dataframe
    dfq    = df.loc[df['Filter']=='q']
    qtimes = np.array(dfq['MJD'].tolist())
    qflux  = np.array(dfq['Flux [μJy]'].tolist())
    qferr  = np.array(dfq['Fluxerr [μJy]'].tolist())
    # qflux  = np.array(dfq['Mag_Opt'].tolist())
    # qferr  = np.array(dfq['Magerr_Opt'].tolist())

    ## It is good practice to calculate the periodogram with respect to
    ## the time-midpoint of the dataset to minimize the uncertainty between
    ## the number of variability cycles between 0 and the starting time of
    ## the dataset.
    # qtimes -= min(qtimes)
    qflux -= np.median(qflux)

    ## Now, let's use the Analysis of Variance algorithm
    ## to search for periodic signals!
    nu_aov, theta_aov = aov.aov_periodogram(qtimes, qflux,
                                            sigma=qferr, min=0.01, max=3.,
                                            oversample_factor=5., nbins=10)

    ## We want to identify the frequency of maximum variability
    nu_max_aov = nu_aov[np.argmax(theta_aov)]

    ## Now lets plot the periodogram and the lightcurve phase folded on
    ## the frequency of most significant variability.
    fig_aov, axes_aov = plt.subplots(1,2, num=1, figsize=(9,4.5))
    axes_aov[0].plot(nu_aov, theta_aov)
    axes_aov[0].set_xlabel(r'${\rm Frequency~[d^{-1}]}$', fontsize=18)
    axes_aov[0].set_ylabel(r'${\Theta}$', fontsize=18)

    axes_aov[1].errorbar((qtimes*nu_max_aov)%1., qflux, yerr=qferr, ls='',
                         marker='x', ms=1, label=r'${\rm q-band}$')
    axes_aov[1].set_xlabel(r'${\rm Phase}$', fontsize=18)
    axes_aov[1].set_ylabel(r'${\rm Flux}$', fontsize=18)
    axes_aov[1].legend(loc='best')

    fig_aov.tight_layout()
    plt.show()


    ## Now, let's use the Conditional Entropy algorithm
    ## to search for periodic signals!
    nu_ce, theta_ce = ce.ce_periodogram(qtimes, qflux,
                                            sigma=None, min=0.01, max=4.,
                                            oversample_factor=4., phase_bins=8, mag_bins=5)

    ## We want to identify the frequency of maximum variability
    ## Be careful, we're minimising the entropy, so we want to use
    ## np.argmin now!
    nu_max_ce = nu_ce[np.argmin(theta_ce)]

    ## Next lets plot the periodogram and the lightcurve phase folded on
    ## the frequency of most significant variability.
    fig_ce, axes_ce = plt.subplots(1,2, num=2, figsize=(9,4.5))
    axes_ce[0].plot(nu_ce, theta_ce)
    axes_ce[0].set_xlabel(r'${\rm Frequency~[d^{-1}]}$', fontsize=18)
    axes_ce[0].set_ylabel(r'${\Theta}$', fontsize=18)

    axes_ce[1].errorbar((qtimes*nu_max_ce)%1., qflux, yerr=qferr, ls='',
                         marker='x', ms=1, label=r'${\rm q-band}$')
    axes_ce[1].set_xlabel(r'${\rm Phase}$', fontsize=18)
    axes_ce[1].set_ylabel(r'${\rm Flux}$', fontsize=18)
    axes_ce[1].legend(loc='best')

    fig_ce.tight_layout()
    plt.show()

    print('The AoV periodogram found the frequency of variability to be:')
    print('\t f={:.5f} c/d'.format(nu_max_aov))
    print('\t p={:.5f} d'.format(1./nu_max_aov))
    print('The CE periodogram found the frequency of variability to be:')
    print('\t f={:.5f} c/d'.format(nu_max_ce))
    print('\t p={:.5f} d'.format(1./nu_max_ce))
