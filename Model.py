"""
This module is used to create Model objects 
for MCMC sampling. These functions and classes
should provide all the tools to set up a Bayesian
model for FRESCO by defining its log-prior and
log-likelihood functions.

Caleb Marshall NCSU/TUNL 2019
"""
import numpy as np
from scipy.stats import norm
from scipy.stats import uniform
from scipy.stats import halfcauchy
from scipy.stats import lognorm
from scipy import interpolate
import os
import sys
import FrescoClasses as fc


class ScalePrior():

    """
    Handles the priors for normalizing variables.
    Takes an array of values for the factor uncertainty
    and creates a "frozen" pdf. lnprior method returns
    the sum input values.
    """

    def __init__(self, fu):
        # First, dumb?, way I thought to allow scaler or np.array
        self.factor_unc = np.asarray(fu)
        try:
            self.prior_len = len(fu)
        except TypeError:
            self.prior_len = 1

        self.pdf = lognorm(np.log(fu))

    def lnprior(self, x):
        return np.sum(self.pdf.logpdf(x))

class PotPrior():

    """
    Handles the priors for optical potential parameters.
    Takes two arrays for the Gaussian means and widths, 
    and defines a frozen pdf. 
    lnprior method returns the sum of input values.
    """

    def __init__(self, means, widths):
        self.gaussian_means = np.asarray(means)
        self.gaussian_widths = np.asarray(widths)
        
        try:
            self.prior_len = len(means)
        except TypeError:
            self.prior_len = 1

        self.pdf = norm(loc=self.gaussian_means,
                        scale=self.gaussian_widths)

    def lnprior(self, x):
        return np.sum(self.pdf.logpdf(x))

class Priors():

    """
    Handles the evaluation of the prior probabilities.
    Right now there is just support for Gaussian priors
    and factor uncertainties. The assumed structure of the
    vector x to be evaluated is [f.u, Gaussian] with Gaussian
    being an array of means and widths. Instances of ScalePrior
    and PotPrior should be input as a list in the order expected
    from the likelihood.
    """

    def __init__(self, priors):
        self.prior_len = []
        self.prior_functions = []
        for ele in priors:
            self.prior_len.append(ele.prior_len)
            self.prior_functions.append(ele.lnprior)
        self.slice_indices = self.make_slices()
            
    def make_slices(self):
        # Make a list of tuples, (i, j),
        # for array indices to splice i.e x[i:j] 
        start = 0
        slices = []
        for ele in self.prior_len:
            slices.append((start, start+ele))
            start += ele
        return slices

    def lnprior(self, x):
        # This list comprehension uses the list of tuples to
        # to create and feed slices of the x into the list of
        # prior objects. For example an elastic scattering data set
        # with normalization would split into x[0:1] for the norm
        # then x[1:last] for potential priors.
        values = [f(x[i[0]:i[1]]) for i, f
                  in zip(self.slice_indices, self.prior_functions)]
        return np.sum(values)

class FrescoEval():
    """
    Handles the swapping of the new values, reading
    in of cross section, and the spline for the likelihood
    evaluation. Takes an fc.NamelistInput object and filename
    for the cross section.
    """
    def __init__(self, fresco, filename='fort.201'):
            self.fresco = fresco
            self.filename = filename
            
    def run_fresco(self, x):
        self.fresco.swap_values(x)
        fc.filerun('new_input')

    def read_fresco(self):
        try:
            cross = fc.read_cross(self.filename)
        except IOError:
            print("Nonphysical Value, FRESCO failed.")
            return -1.0*np.inf
        os.remove(self.filename)
        spline = interpolate.UnivariateSpline(cross.theta,
                                              cross.sigma,
                                              s=0)
        return spline


class LnLikeElastic(FrescoEval):

    """
    Class that inherits the fresco processing features of
    FrescoEval, and then includes experimental data to 
    form a simple Gaussian likelihood 
    (which I have called chi^2 for no good reason).

    Initialize with a fc.NamelistInput instance and fc.DataObject instance
    (or path to the data file)

    normalization adjusts the likelihood function to include a factor_unc.
    """

    def __init__(self, fresco, data, normalized=False):
        FrescoEval.__init__(self, fresco)

        # This block makes sure we have a fc.DataObject
        try:
            self.data = fc.read_data(data)
        except ValueError:
            self.data = data

        # Whatever chi^2 function is chosen we still call lnlike
        if normalized:
            self.lnlike = self.norm_fresco_chi
        else:
            self.lnlike = self.fresco_chi
            
    def fresco_chi(self, x):
        self.run_fresco(x)
        spline = self.read_fresco()
        try:
            theory = spline(self.data.theta)
        except TypeError:
            return spline #  read_fresco returned -inf 
        likelihood = norm.logpdf(theory,
                                 loc=(self.data.sigma),
                                 scale=self.data.erry)
        likelihood = np.sum(likelihood) 
        return likelihood


    def norm_fresco_chi(self, x):
        self.run_fresco(x[1:])  # Norm is x[0]
        spline = self.read_fresco()
        try:
            theory = spline(self.data.theta)
        except TypeError:
            return spline
        likelihood = norm.logpdf(theory,
                                 loc=(self.data.sigma*x[0]),
                                 scale=(self.data.erry*x[0]))
        likelihood = np.sum(likelihood)
        return likelihood

    
class Model():

    """
    This class is used to specify the prior probabilities
    and likelihood function for a Bayesian model. 
    It handles the input to the prior and likelihood
    instances.
    """

    def __init__(self, fresco_path, fresco_names,
                 fresco_positions, **kwargs):
        self.fresco = fc.NamelistInput(fresco_path)
        self.fresco.names = fresco_names
        self.fresco.name_positions = fresco_positions
        # Model gets built up assuming potential priors
        # first and then additional variables on top
        self.fresco.initial_values()
        self.x0 = self.fresco.x0
        self.norm_priors = []
        self.pot_priors = []
        self.likelihood = []
        

    # These series of methods create all of the elements
    # needed for evaluating the lnprob.
        
    def create_norm_prior(self, fu):
        try:
            norm_x0 = np.ones(len(fu))
        except TypeError:
            norm_x0 = [1.0]
        self.norm_priors.append(ScalePrior(fu))

    def create_pot_prior(self, means, widths):
        self.pot_priors.append(PotPrior(means, widths))

    def create_prior(self):
        self.priors = Priors(self.norm_priors + self.pot_priors)
        
    def create_elastic_likelihood(self, data, normalized=False):
        self.likelihood.append(LnLikeElastic(self.fresco,
                                             data,
                                             normalized=normalized))

    # lnprob is to be called by emcee. dynesty will require
    # future updates.
    
    def lnprob(self, x):
        probability = self.priors.lnprior(x)
        if np.isinf(probability):
            return -1.0 * np.inf
        for ele in self.likelihood:
            probability += ele.lnlike(x)
        return probability
