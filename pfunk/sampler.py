import numpy as np
import emcee



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

    def multimodal_init(self, modes, r_limits=[], scatter=1.e-2):
        """
        This divides the walkers evenly between the modes given.
        The modes are in terms of n, c parameters like in cont_init
        function. "r_limits" is a list that contains the r range
        to use the Vr^n = c relationship over.
        """

        return None

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

    def run_differential_ev(self):
        """
        Run the sampler with a differential evolution move.
        Following the suggestions of Braak and Vrugt 2008
        the DE move is mixed with a snooker move. This mixture
        is a 90/10 split.
        """
        move = ((emcee.moves.DEMove(), .9), (emcee.moves.DESnookerMove(), .1))
        self.sampler = emcee.EnsembleSampler(self.nwalker,
                                             self.ndim,
                                             self.model.lnprob,
                                             moves=move)
        self.sampler.run_mcmc(self.p0, self.nstep, progress=True)

    def run_dynest(self):
        self.sampler = dynesty.DynamicNestedSampler(self.model.lnlike,
                                                    self.model.prior_transform,
                                                    ndim=self.ndim,
                                                    bound='none',
                                                    sample='rslice',
                                                    slices=10, nlive=200)
        self.sampler.run_nested(wt_kwargs={'pfrac': 1.0})

