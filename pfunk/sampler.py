import numpy as np
import emcee
import dynesty
import zeus

class Sampler():

    def __init__(self, model, nwalker=50, nstep=100):
        self.ndim = len(model.x0)
        self.nwalker = nwalker
        self.nstep = nstep
        self.model = model
    

    def ball_init(self, scatter=1.e-2):
        """
        Initialize a random ndim ball around model.x0 parameters
        according to 'scatter' parameter, i.e the larger scale is
        the larger the ball is (can be).
        """
        self.p0 = ((self.model.x0*np.ones([self.nwalker, self.ndim])) +
                   (scatter*np.random.randn(self.nwalker, self.ndim)))

    def cont_init(self, lower=1.15, upper=1.55, scatter=1.e-2):
        """
        Ball initialization except that the real well depth, V,
        and radius parameter, r, are randomly scattered along
        the Vr**n = c relationship. V and r are assumed to be in the
        indices 0 and 1.
        """
        self.ball_init()
        scale = upper - lower  # Put into form scipy uniform expects.
        self.p0[:, 1] = uniform.rvs(lower, scale=scale, size=self.nwalker)
        # See there are fit n, c parameters.
        try:
            top = (self.model.c + np.random.randn(self.nwalker))
            bottom = self.p0[:, 1]**(self.model.n +
                                     (scatter * np.random.randn(self.nwalker)))
            self.p0[:, 0] = top/bottom
        except TypeError:
            print("Model parameters n and c have not been initialized.")

    def check_p0(self):
        """Check the lnprob values of the starting values.

        :returns: lnprob values of the each p0
        :rtype: np.array

        """

        vals = np.apply_along_axis(self.model.lnprob, 1, self.p0)
        return vals

    def prior_sample_init(self):
        self.ball_init()
        for ele in self.p0:
            ele = self.model.priors.prior_sample()
        
    def free_sig_init(self, scatter=1.e-2):
        """
        Does a ball initialization on potential and sigma parameters.
        """
        self.p0 = np.zeros([self.nwalker, self.ndim])
        # Potential parameters go first
        pot_values = ((self.model.x0 * np.ones([self.nwalker, len(self.model.x0)])) +
                      (scatter * np.random.randn(self.nwalker, len(self.model.x0))))
        self.p0[:, :len(self.model.x0)] = pot_values
        # Scaled sigma^2 values from "true" sigma values from data file
        n = len(self.model.sig_square)
        sig_values = ((self.model.sig_square * np.ones([self.nwalker, n])) +
                      (scatter * np.random.randn(self.nwalker, n)))
        self.p0[:, len(self.model.x0):] = sig_values

    def run_ensemble(self):
        """
        Run the sampler with the default stretch move.
        """
        self.sampler = emcee.EnsembleSampler(self.nwalker,
                                             self.ndim,
                                             self.model.lnprob)
        self.sampler.run_mcmc(self.p0, self.nstep, progress=True)

    # def run_differential_ev(self):
    #     """
    #     Run the sampler using the pydream module.
    #     https://github.com/LoLab-VU/PyDREAM
    #     """
        
    #     self.sampler = run_dream(self.model.dream_priors,
    #                              self.model.lnlikefunc,
    #                              nchains=self.nwalker,
    #                              niterations=self.nstep,
    #                              start = self.p0)
                                              

    def run_nested(self, nlive=250, dlogz=.01, sample='slice', bound='multi'):
        self.sampler = dynesty.NestedSampler(self.model.lnlikefunc,
                                             self.model.priors.transform_prior,
                                             ndim=self.ndim,
                                             bound=bound,
                                             sample=sample,
                                             nlive=nlive)
        self.sampler.run_nested(dlogz=dlogz)

    def run_dynamic_nested(self, evidence=False, posterior=False,
                           nlive=250, sample='slice'):
        self.sampler = dynesty.DynamicNestedSampler(self.model.lnlikefunc,
                                                    self.model.priors.transform_prior,
                                                    ndim=self.ndim,
                                                    bound='multi',
                                                    sample=sample, nlive=nlive)
        if evidence and posterior:
            print('Both, really? Going to default')
            self.sampler.run_nested()
        elif evidence:
            # evidence focused dynamic run
            self.sampler.run_nested(wt_kwargs={'pfrac': 0.0},
                                    stop_kwargs={'pfrac': 0.0})
        elif posterior:
            # evidence focused dynamic run
            self.sampler.run_nested(wt_kwargs={'pfrac': 1.0})    
        else:
            # Default behavior, 80/20 weight split and 100% posterior. 
            self.sampler.run_nested()

    def run_slice_ensemble(self):
        self.sampler = zeus.sampler(self.nwalker, self.ndim, self.model.lnprob)
        self.sampler.run_mcmc(self.p0, self.nstep)
