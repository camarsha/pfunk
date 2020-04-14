"""
This module is used to create Model objects 
for MCMC sampling. These functions and classes
should provide all the tools to set up a Bayesian
model for FRESCO by defining its log-prior and
log-likelihood functions.

Caleb Marshall NCSU/TUNL 2019
"""
import numpy as np
import dynesty as dy
from scipy.stats import norm
from scipy.stats import halfnorm
from scipy.stats import uniform
from scipy.stats import halfcauchy
from scipy.stats import lognorm
from scipy.stats import t
from scipy.stats import gamma
from scipy import interpolate
import os
import sys
from . import fresco_classes as fc
#from pydream.parameters import SampledParam 


class FlatPrior():

    def __init__(self, lower, upper):
        self.means = np.array(lower)
        self.upper = np.array(upper)
        self.pdf = uniform(loc=self.means, scale=(self.upper - self.means))
        self.prior_len = 1

    def lnprior(self, x):
        return np.sum(self.pdf.logpdf(x))

    def prior_transform(self, x):
        return self.pdf.ppf(x)

    def prior_rvs(self):
        return self.pdf.rvs()

class ScatterPrior():


    def __init__(self):
        self.means = np.array([0.0])
        self.widths = np.array([1.0])
        self.pdf = halfnorm(self.means, scale=self.widths)
        self.prior_len = 1

    def lnprior(self, x):
        return np.sum(self.pdf.logpdf(x))

    def prior_transform(self, x):
        return self.pdf.ppf(x)

    def prior_rvs(self):
        return self.pdf.rvs()


class DPrior():

    """
    Handles the priors for normalizing variables.
    Takes an array of values for the factor uncertainty
    and creates a "frozen" pdf. lnprior method returns
    the sum input values.
    """

    def __init__(self, means, widths):
        # First, dumb?, way I thought to allow scaler or np.array
        self.means = np.asarray(means)
        self.widths = np.asarray(widths)
        try:
            self.prior_len = len(means)
        except TypeError:
            self.prior_len = 1

        self.pdf = norm(loc=means, scale=widths)

    def lnprior(self, x):
        return np.sum(self.pdf.logpdf(x))

    def prior_transform(self, x):
        return self.pdf.ppf(x)

    def prior_rvs(self):
        return self.pdf.rvs()


class ScalePrior():

    """
    Handles the priors for normalizing variables.
    Takes an array of values for the factor uncertainty
    and creates a "frozen" pdf. lnprior method returns
    the sum input values.
    """

    def __init__(self, means, widths):
        # First, dumb?, way I thought to allow scaler or np.array
        self.means = np.asarray(means)
        self.widths = np.asarray(widths)
        try:
            self.prior_len = len(means)
        except TypeError:
            self.prior_len = 1

        self.pdf = halfnorm(loc=means, scale=widths)

    def lnprior(self, x):
        return np.sum(self.pdf.logpdf(x))

    def prior_transform(self, x):
        return self.pdf.ppf(x)

    def prior_rvs(self):
        return self.pdf.rvs()


class PotPrior():

    """
    Handles the priors for optical potential parameters.
    Takes two arrays for the Gaussian means and widths, 
    and defines a frozen pdf. 
    lnprior method returns the sum of input values.
    """

    def __init__(self, means, widths):
        self.means = np.asarray(means)
        self.widths = np.asarray(widths)
        
        try:
            self.prior_len = len(means)
        except TypeError:
            self.prior_len = 1

        self.pdf = norm(loc=self.means,
                        scale=self.widths)

    def lnprior(self, x):
        return np.sum(self.pdf.logpdf(x))

    def prior_transform(self, x):
        return self.pdf.ppf(x)

    def prior_rvs(self):
        return self.pdf.rvs()
    
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
        self.x0 = []
        self.prior_transforms = []
        self.prior_rvs = []
        for ele in priors:
            self.prior_len.append(ele.prior_len)
            self.prior_functions.append(ele.lnprior)
            self.x0.append(ele.means)
            self.prior_transforms.append(ele.prior_transform)
            self.prior_rvs.append(ele.prior_rvs)
        try:
            self.x0 = np.concatenate(self.x0)
        except ValueError:
            self.x0 = np.asarray(self.x0)
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

    def transform_prior(self, x):
        """
        For use with nested sampler. Returns a vector
        of transformed values from the uniform distribution
        input values.
        """
        values = [f(x[i[0]:i[1]]) for i, f
                  in zip(self.slice_indices, self.prior_transforms)]
        return np.concatenate(values)
    
    def prior_sample(self):
        values = []
        for ele in self.prior_rvs:
            temp = ele()
            if not isinstance(temp, np.ndarray):
                temp = [temp]
            values.append(temp)
        return np.concatenate(values)
        
class FrescoEval():
    """
    Handles the swapping of the new values, reading
    in of cross section, and the spline for the likelihood
    evaluation. Takes an fc.NamelistInput object and filename
    for the cross section.
    """
    def __init__(self, filename, remove=True):
            self.filename = filename
            self.remove = remove
            
    def read_fresco(self):
        try:
            cross = fc.read_cross(self.filename)
        except IOError:
            return -1.0*np.inf
        if self.remove:
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

    def __init__(self, filename, data, norm_index=False, scatter_index=False,
                 remove=True):
        FrescoEval.__init__(self, filename, remove=remove)

        # This block makes sure we have a fc.DataObject
        try:
            self.data = fc.read_data(data)
        except ValueError:
            self.data = data

        # Whatever function is chosen we still call lnlike
        if isinstance(norm_index, int):
            self.i = norm_index
            self.lnlike = self.norm_fresco_chi
        else:
            self.lnlike = self.fresco_chi

        if isinstance(scatter_index, int):        
            self.scatter_index = scatter_index
            if isinstance(norm_index, int):    
                self.lnlike = self.norm_scatter_chi
            else:
                self.lnlike = self.scatter_chi

            
    def fresco_chi(self, x):
        spline = self.read_fresco()
        try:
            theory = spline(self.data.theta)
        except TypeError:
            return spline #  read_fresco returned -inf 
        likelihood = norm.logpdf(self.data.sigma,
                                 loc=(theory),
                                 scale=self.data.erry)
        likelihood = np.sum(likelihood) 
        return likelihood


    def norm_fresco_chi(self, x):
        spline = self.read_fresco()
        try:
            theory = spline(self.data.theta)
        except TypeError:
            return spline
        n = 10.0**(x[self.i])
        likelihood = norm.logpdf(self.data.sigma,
                                 loc=theory*n,
                                 scale=self.data.erry)
        likelihood = np.sum(likelihood)
        return likelihood

    def norm_scatter_chi(self, x):
        spline = self.read_fresco()
        try:
            theory = spline(self.data.theta)
        except TypeError:
            return spline
        n = 10.0**(x[self.i])
        scale = np.sqrt((self.data.erry)**2.0 +
                        (n*theory*x[self.scatter_index])**2.0)
        likelihood = norm.logpdf(self.data.sigma,
                                 loc=(theory*n),
                                 scale=scale)
        likelihood = np.sum(likelihood)
        return likelihood

    
class LnLikeTransfer(LnLikeElastic):

    """
    Transfer likelihood functions that assume fresco has already
    been run (i.e elastic likelihood has already been called).
    """

    def __init__(self, filename, data, sf_index, scatter_index=None,
                 norm_index=None, remove=True):

        # The assignement of lnlike is the same, but we redefine their forms below.
        LnLikeElastic.__init__(self, filename, data, norm_index=norm_index,
                               scatter_index=scatter_index, remove=remove)

        self.sf_index = sf_index
                
    def fresco_chi(self, x):
        """
        Likelihood for transfer cross section. Always
        needs spectroscopic factor to be defined.
        """
        spline = self.read_fresco()
        try:
            theory = spline(self.data.theta)
        except TypeError:
            return spline #  read_fresco returned -inf
        sf = np.prod(x[self.sf_index])
        likelihood = norm.logpdf(self.data.sigma,
                                 loc=(theory*sf),
                                 scale=self.data.erry)
        likelihood = np.sum(likelihood) 
        return likelihood

    def norm_fresco_chi(self, x):
        spline = self.read_fresco()
        try:
            theory = spline(self.data.theta)
        except TypeError:
            return spline
        sf = np.prod(x[self.sf_index])
        n = 10.0**(x[self.i])
        likelihood = norm.logpdf(self.data.sigma,
                                 loc=(theory*sf*n),
                                 scale=(self.data.erry))
        likelihood = np.sum(likelihood)
        return likelihood

    def norm_scatter_chi(self, x):
        spline = self.read_fresco()
        try:
            theory = spline(self.data.theta)
        except TypeError:
            return spline
        sf = np.prod(x[self.sf_index])
        n = 10.0**(x[self.i])
        scale = np.sqrt((self.data.erry)**2.0 +
                        (n*sf*theory*x[self.scatter_index])**2.0)
        likelihood = norm.logpdf(self.data.sigma,
                                 loc=(theory*sf*n),
                                 scale=scale)
        likelihood = np.sum(likelihood)
        return likelihood

    def scatter_chi(self, x):
        """
        Likelihood for transfer cross section. Always
        needs spectroscopic factor to be defined.
        """
        spline = self.read_fresco()
        try:
            theory = spline(self.data.theta)
        except TypeError:
            return spline #  read_fresco returned -inf
        sf = np.prod(x[self.sf_index])
        scale = np.sqrt((self.data.erry)**2.0 +
                        (sf*theory*x[self.scatter_index])**2.0)
        likelihood = norm.logpdf(self.data.sigma,
                                 loc=(theory*sf),
                                 scale=scale)
        likelihood = np.sum(likelihood) 
        return likelihood

class LnLikeTransferTwoL(LnLikeElastic):

    """
    Transfer likelihood functions that assume fresco has already
    been run (i.e elastic likelihood has already been called).
    For the two l case we assume one fresco file still produced
    all of the cross sections.
    """

    def __init__(self, filename1, filename2, data,
                 sf_index1, sf_index2, scatter_index=None,
                 norm_index=None, remove=True):
        self.cs_eval1 = FrescoEval(filename1, remove=remove)
        self.cs_eval2 = FrescoEval(filename2, remove=remove)

        self.sf_index1 = sf_index1
        self.sf_index2 = sf_index2
        
        # This block makes sure we have a fc.DataObject
        try:
            self.data = fc.read_data(data)
        except ValueError:
            self.data = data

        # Whatever function is chosen we still call lnlike
        if isinstance(norm_index, int):
            self.i = norm_index
            self.lnlike = self.norm_fresco_chi
        else:
            self.lnlike = self.fresco_chi

        if isinstance(scatter_index, int):        
            self.scatter_index = scatter_index
            if isinstance(norm_index, int):    
                self.lnlike = self.norm_scatter_chi
            else:
                self.lnlike = self.scatter_chi

                
    def fresco_chi(self, x):
        """
        Likelihood that reads both cross sections
        and adds them together to compare to the
        data.
        """
        spline1 = self.cs_eval1.read_fresco()
        spline2 = self.cs_eval2.read_fresco()
        try:
            theory1 = spline1(self.data.theta)
            theory2 = spline2(self.data.theta)
        except TypeError:
            return spline1  # read_fresco returned -inf
        sf1 = np.prod(x[self.sf_index1])
        sf2 = np.prod(x[self.sf_index2])
        theory_total = sf1*theory1 + sf2*theory2
        likelihood = norm.logpdf(self.data.sigma,
                                 loc=(theory_total),
                                 scale=self.data.erry)
        likelihood = np.sum(likelihood) 
        return likelihood

    def norm_fresco_chi(self, x):
        """
        For this case we have a common factor, n, 
        which is shared by both cross sections.
        """
        spline1 = self.cs_eval1.read_fresco()
        spline2 = self.cs_eval2.read_fresco()
        try:
            theory1 = spline1(self.data.theta)
            theory2 = spline2(self.data.theta)
        except TypeError:
            return spline1  # read_fresco returned -inf
        sf1 = np.prod(x[self.sf_index1])
        sf2 = np.prod(x[self.sf_index2])
        n = 10.0**(x[self.i])
        theory_total = sf1*n*theory1 + sf2*n*theory2
        
        likelihood = norm.logpdf(self.data.sigma,
                                 loc=(theory_total),
                                 scale=(self.data.erry))
        likelihood = np.sum(likelihood)
        return likelihood

    def norm_scatter_chi(self, x):
        """
        Error estimate, f, is multiplied by the total cross section.
        """
        spline1 = self.cs_eval1.read_fresco()
        spline2 = self.cs_eval2.read_fresco()
        try:
            theory1 = spline1(self.data.theta)
            theory2 = spline2(self.data.theta)
        except TypeError:
            return spline1  # read_fresco returned -inf
        
        sf1 = np.prod(x[self.sf_index1])
        sf2 = np.prod(x[self.sf_index2])
        n = 10.0**(x[self.i])
        theory_total = sf1*n*theory1 + sf2*n*theory2
        
        scale = np.sqrt((self.data.erry)**2.0 +
                        (theory_total*x[self.scatter_index])**2.0)
        likelihood = norm.logpdf(self.data.sigma,
                                 loc=(theory_total),
                                 scale=scale)
        likelihood = np.sum(likelihood)
        return likelihood

    def scatter_chi(self, x):
        """
        Scatter without the normalization.
        """
        spline1 = self.cs_eval1.read_fresco()
        spline2 = self.cs_eval2.read_fresco()
        try:
            theory1 = spline1(self.data.theta)
            theory2 = spline2(self.data.theta)
        except TypeError:
            return spline1  # read_fresco returned -inf
        
        sf1 = np.prod(x[self.sf_index1])
        sf2 = np.prod(x[self.sf_index2])
        theory_total = sf1*theory1 + sf2*theory2

        scale = np.sqrt((self.data.erry)**2.0 +
                        (theory_total*x[self.scatter_index])**2.0)
        likelihood = norm.logpdf(self.data.sigma,
                                 loc=(theory_total),
                                 scale=scale)
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
        self.fresco_path = fresco_path
        self.fresco = fc.NamelistInput(fresco_path)
        self.fresco.create_names_positions(fresco_names,
                                           fresco_positions)
        # Model gets built up assuming potential priors
        # first and then additional variables on top
        self.fresco.initial_values()
        self.x0_init = self.fresco.x0[:]
        self.norm_priors = []
        self.pot_priors = []
        self.spec_priors = []
        self.scatter_priors = []
        self.likelihood = []
        self.transfer_likelihood = []
        self.norm_len = 0
        self.sf_len = 0
        self.scatter_len = 0
        self.dream_priors = []

    # These series of methods create all of the elements
    # needed for evaluating the lnprob.
        
    def create_norm_prior(self, means, widths):
        try:
            norm_x0 = np.ones(len(means))
        except TypeError:
            norm_x0 = np.array([1.0])
        self.norm_priors.append(FlatPrior(means, widths))

        
    def create_spec_prior(self, means, widths, gaus=False):
        try:
            spec_x0 = np.ones(len(means))
        except TypeError:
            spec_x0 = np.array([1.0])
        if gaus:
            self.spec_priors.append(DPrior(means, widths))
        else:
            self.spec_priors.append(ScalePrior(means, widths))

    def create_scatter_prior(self):
        self.scatter_priors.append(ScatterPrior())
    
    def create_pot_prior(self, means, widths):
        self.pot_priors.append(PotPrior(means, widths))


    def create_prior(self):
        # This establishes the canonical order for our inputs
        # Normalization first, spectroscopic factors next, transfer scatter and finally fresco variables
        prior_list = self.norm_priors + self.spec_priors + self.scatter_priors + self.pot_priors
        self.priors = Priors(prior_list)
        # We now set variables that will slice up the array passed by the sampler
        self.norm_len = len(self.norm_priors)
        self.sf_len = len(self.spec_priors)
        self.scatter_len = len(self.scatter_priors)
        self.x0 = self.priors.x0[:]
        # For compatibility with pydreams
        # for ele in prior_list:
        #     self.dream_priors.append(SampledParam(ele.pdf))

    def create_elastic_likelihood(self, filename, data, norm_index=None,
                                  scatter_index=None, remove=True):
        self.likelihood.append(LnLikeElastic(filename,
                                             data,
                                             norm_index=norm_index,
                                             scatter_index=scatter_index, remove=remove))

    def create_transfer_likelihood(self, filename, data, sf_index,
                                   scatter_index=None, norm_index=None, remove=True):
        self.transfer_likelihood.append(LnLikeTransfer(filename,
                                                       data,
                                                       sf_index,
                                                       scatter_index=scatter_index,
                                                       norm_index=norm_index, remove=remove))

    def create_two_l_transfer_likelihood(self, filename1, filename2,
                                         data, sf_index1, sf_index2,
                                         scatter_index=None, norm_index=None, remove=True):
        self.transfer_likelihood.append(LnLikeTransferTwoL(filename1,
                                                           filename2,
                                                           data,
                                                           sf_index1,
                                                           sf_index2,
                                                           scatter_index=scatter_index,
                                                           norm_index=norm_index, remove=remove))
        
    def create_likelihood(self):
        # If transfer reactions are just file reads after
        self.likelihood = self.likelihood + self.transfer_likelihood


    def run_fresco(self, x):
        x_slice = x[(self.norm_len + self.sf_len + self.scatter_len):]
        if x_slice.size > 0:
            self.fresco.swap_values(x_slice)
        # This handles the case of just normalization and spec factors
        # i.e fresco parameters are fixed.
        else:
            self.fresco.create_original()
        fc.filerun('new_input')
        
    # lnprob is to be called by emcee. dynesty will require
    # lnlike method.
    
    def lnprob(self, x):
    
        # We run fresco once and then read each cross
        # section file.

        probability = self.priors.lnprior(x)
        if np.isinf(probability):
            return -1.0 * np.inf
        # Slice array to get the fresco parameters
        self.run_fresco(x)
        for ele in self.likelihood:
            probability += ele.lnlike(x)
        if np.isnan(probability):
            return -1.0 * np.inf
        return probability

    def lnlikefunc(self, x):
        probability = 0.0
        self.run_fresco(x)
        for ele in self.likelihood:
            probability += ele.lnlike(x)
        if np.isnan(probability):
            return -1.0 * np.inf
        return probability
