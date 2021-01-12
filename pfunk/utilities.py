import numpy as np
from scipy.stats import norm
from scipy.stats import halfcauchy
from scipy.stats import lognorm
from scipy.stats import spearmanr
from scipy import interpolate
from scipy.stats import spearmanr
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sbs
import scipy.stats.kde as kde 
import dynesty


def hpd_grid(sample, alpha=0.32, roundto=2):
    """Calculate highest posterior density (HPD) of array for given alpha. 
    The HPD is the minimum width Bayesian credible interval (BCI). 
    The function works for multimodal distributions, returning more than one mode
    Parameters
    ----------
    
    sample : Numpy array or python list
        An array containing MCMC samples
    alpha : float
        Desired probability of type I error (defaults to 0.05)
    roundto: integer
        Number of digits after the decimal point for the results
    Returns
    ----------
    hpd: array with the lower 
    
    written by Osvaldo Martin
    taken from https://github.com/aloctavodia/BAP/blob/master/first_edition/code/Chp1/hpd.py
    
    """
    sample = np.asarray(sample)
    sample = sample[~np.isnan(sample)]
    # get upper and lower bounds
    l = np.min(sample)
    u = np.max(sample)
    density = kde.gaussian_kde(sample)
    x = np.linspace(l, u, 2000)
    y = density.evaluate(x)
    #y = density.evaluate(x, l, u) waitting for PR to be accepted
    xy_zipped = zip(x, y/np.sum(y))
    xy = sorted(xy_zipped, key=lambda x: x[1], reverse=True)
    xy_cum_sum = 0
    hdv = []
    for val in xy:
        xy_cum_sum += val[1]
        hdv.append(val[0])
        if xy_cum_sum >= (1-alpha):
            break
    hdv.sort()
    diff = (u-l)/20  # differences of 5%
    hpd = []
    hpd.append(round(min(hdv), roundto))
    for i in range(1, len(hdv)):
        if hdv[i]-hdv[i-1] >= diff:
            hpd.append(round(hdv[i-1], roundto))
            hpd.append(round(hdv[i], roundto))
    hpd.append(round(max(hdv), roundto))
    ite = iter(hpd)
    hpd = list(zip(ite, ite))
    modes = []
    for value in hpd:
         x_hpd = x[(x > value[0]) & (x < value[1])]
         y_hpd = y[(x > value[0]) & (x < value[1])]
         modes.append(round(x_hpd[np.argmax(y_hpd)], roundto))
    return hpd, x, y, modes



def spec_factor_correlations(samples, spec_parameter=1, n=0):
    """
    def spec_factor_correlations(samples, spec_parameter=1, n=0)

    Given an array "samples" of shape (# of samples, # of parameters)
    select the parameter related to the spectroscopic factor and plot
    all correlations. If a non-zero n is given use the spearman-rank-order
    coefficient to find the top n most important parameters.
    """

    spectroscopic_array = samples[:, spec_parameter]
    other_parameters = np.delete(samples, spec_parameter, axis=1)
    n_parameters = other_parameters.shape[1]  # Number of parameters
    # Initialize the plot
    correlations = np.zeros(n_parameters+1)
    for i in range(n_parameters+1):
        correlations[i] = spearmanr(samples[:, i], spectroscopic_array)[0]
    return correlations

def confidence_bands(lines, percentiles=[16, 50, 84]):
    angles = np.asarray(lines[0].theta[:])
    all_sigmas = np.zeros([len(angles), len(lines)])
    for i, ele in enumerate(lines):
        for j in range(len(angles)):
            all_sigmas[j][i] = ele.sigma[j]
    return np.percentile(all_sigmas, percentiles, axis=1)


def plot_ci(lines, levels=[68.0, 95.0], data=None, colors=None, alpha=1.0,
            linestyle='-', linewidth=1.0, hatch=None, zorder=0, markersize=5.0):
    """
    Set up a plot with 68%, 95%, and 99% credibility bands.
    '#e0ecf4', 
    """

    # Purple son
    if colors:
        fill_colors = colors
    else:
        fill_colors = reversed(["#a6cee3", "#6a3d9a"])

    for color, i in zip(fill_colors, levels):
        # Get the intervals
        low = 50. - (i/2.0)
        high = 50. + (i/2.0)
        ci = confidence_bands(lines, percentiles=[low, 50.0, high])
        # Spline and plot
        angles = np.arange(len(ci[0, :]))
        spline_angles = np.linspace(0, len(angles), 100000)

        lower_spline = interpolate.UnivariateSpline(angles,
                                                    ci[0, :], s=0)
        lower = lower_spline(spline_angles)

        upper_spline = interpolate.UnivariateSpline(angles,
                                                    ci[-1, :], s=0)
        upper = upper_spline(spline_angles)

        plt.fill_between(spline_angles,
                         lower, upper, hatch=hatch, linestyle=linestyle,
                         color=color, zorder=zorder, alpha=alpha, linewidth=linewidth)
        zorder = zorder-1
    if data:
        plt.errorbar(data.theta, data.sigma, yerr=data.erry,
                     linestyle='None', marker='o',
                     markersize=markersize, zorder=1, color="#ff7f00")
    plt.ylim(1e-3, 1e2)
    plt.yscale('log')
    plt.xlim(-.2, 180.0)
    plt.xlabel(r'$\theta_{c.\!m.}$(deg)', fontsize=32)
    plt.ylabel(r'$\sigma(\theta_{c.\!m.})$(mb/sr)', fontsize=32)
    plt.tight_layout()

def plot_ci_comp(lines, colors, alpha=0.6):
    """Quickly generate multiple ci plots using just the 68% interval.

    :param lines: cross section object
    :param colors: list for each line
    :returns: plots 
    :rtype: plots?

    """
    if len(lines) != len(colors):
        colors = colors[:len(lines)] # match the lengths
    
    for ele, c in zip(lines, colors):
        plot_ci(ele, colors=[c], alpha=alpha, levels=[68.0])


def cross_section_residual(lines, levels=[68.0, 95.0], data=None, cs_true=None, credibility=68.0):
    angle_range = np.asarray(lines[0].theta[:], dtype='float')
    # ci = confidence_bands(lines, percentiles=[low, 50.0, high])
    if colors:
        fill_colors = colors
    else:
        # Purple son
        fill_colors = reversed(['#e0ecf4', '#9ebcda', '#8856a7'])
    # Setup the three levels of credibility.

    # Median will be the unity line
    plt.hlines(1.0, angle_range.min()-5.0, angle_range.max()+5.0,
               color='k', linestyle='--')
    zorder = 0
    for color, i in zip(fill_colors, levels):
        low = 50. - (i/2.0)
        high = 50. + (i/2.0)
        ci = confidence_bands(lines, percentiles=[low, 50.0, high])
        angles = np.arange(len(ci[0, :]))
        spline_angles = np.linspace(0, len(angles), 100000)

        lower_spline = interpolate.splrep(angles, (ci[0, :])/ci[1, :])
        lower = interpolate.splev(spline_angles, lower_spline)

        upper_spline = interpolate.splrep(angles, (ci[-1, :]/ci[1, :]))
        upper = interpolate.splev(spline_angles, upper_spline)
        plt.fill_between(spline_angles,
                             lower, upper,
                             color=color, alpha=1.0, zorder=zorder)
        zorder = zorder-1
    # plt.fill_between(spline_angles,
    #                      lower, upper,
    #                      color='#8856a7', alpha=1.0)
    if cs_true:
        plt.plot(cs_true.theta, (cs_true.sigma/ci[1, :]), color='#e66101')
    if data:
        med_spline = interpolate.splrep(angle_range, ci[1, :])
        med = interpolate.splev(data.theta, med_spline)
        plt.errorbar(data.theta, (data.sigma/med), yerr=data.erry,
                     linestyle='None', marker='o', color='#e66101', markersize=5.0)
        print("The sort of Chi is:",
              np.sum(((data.sigma-med)/data.erry)**2.0)*(1.0/(len(data.theta-6.))))
    plt.xlabel(r'$\theta_{cm}$', fontsize=32)
    plt.xlim(-0.2, 180.2)


def parameters_values(samples):

    """
    Given an array of samples returns each parameters
    16, 50, 84 percentile values. Returns list of tuples.
    """
    values = [(v[1], v[2]-v[1], v[1]-v[0]) for v in zip(*np.percentile(samples,
                                    [16, 50, 84],
                                    axis=0))]
    return values

def make_samples_dynesty(results):
    """Uses dynesty's resampling function 
       to return equally weighted samples.
          
    :param results: the results dictionary from dynesty.  
    :returns: equally weighted samples
    :rtype: numpy array

    """

    # Construct the sample weights
    weights = np.exp(results['logwt'] - results['logz'][-1])
    # Resample
    samples = dynesty.utils.resample_equal(results['samples'], weights)
    return samples

def evidence_unc(results):
    """Copy and paste job of a function to 
    estimate the statistical and sampling uncertainties
    of nested sampling evidence estimation.

    :param results: object returned from dynesty sampler
    :returns: results object with well defined 'logzerr' 
    :rtype: some sort of dictionary thingy

    """

    lnzs = np.zeros((100, len(results.logvol)))
    for i in range(100):
        r = dynesty.utils.simulate_run(results, approx=True)
        lnzs[i] = np.interp(-results.logvol, -r.logvol, r.logz)
    lnzerr = np.std(lnzs, axis=0)
    results['logzerr'] = lnzerr
    return results

def log_norm_parameters(samples):
    """Estimate the lognormal parameters based on the 
    given samples. Gaussian parameters are extracted from
    the log of the samples, and then translated into the appropriate
    lognormal parameters. There is no explicit fitting.

    :param samples: numpy array of the samples
    :returns: median, factor_unc
    :rtype: float64

    """
    mu = np.log(samples).mean()
    std = np.log(samples).std()

    med = np.exp(mu)
    fu = np.exp(std)
    
    x = np.linspace(samples.min(), samples.max(), 10000)
    
    plt.hist(samples, bins=50, density=True)
    plt.plot(x, lognorm.pdf(x, std, scale=med))
    plt.show()
    print('Median =', med)
    print('f.u. =', fu)
    print('Lognormal Credibility Interval =', med,'+/-',(med*fu-med),(med-(med/fu)))
    # Show 68% ci jsut for comparison.
    center = np.percentile(samples, 50.0) 
    upper = np.percentile(samples, 84.0) - center 
    lower = center - np.percentile(samples, 16.0) 
    print('Credibility Interval of Samples =', center,'+/-',upper,lower)
    return med, fu

def mixed_transition_sf(sf, alpha):
    """Function for convince that gives 
    the two spectroscopic factors for a mixed 
    l transition that is parameterized by a single
    spectroscopic factor and a mixing parameter, alpha.
    Returns the two spectroscopic factors.

    :param sf: Combined spectroscopic factor samples.
    :param alpha: Samples of alpha
    :returns: sf1, sf2
    :rtype: numpy arrays

    """

    sf1 = sf*alpha
    sf2 = sf*(1-alpha)
    return sf1, sf2

