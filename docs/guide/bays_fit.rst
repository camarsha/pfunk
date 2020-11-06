A Simple Bayesian Fit
=====================

With some of the basics out of the way, we can
now look at the kinds of calculations pfunk was intended for. Bayesian
statistics uses the cross section data to update the prior probability distributions
for the optical model potentials via:

.. math::
   
   P(\mathcal{U}|D) \propto P(D|\mathcal{U})P(\mathcal{U}),

where :math:`\mathcal{U}` is the collection of optical model parameters and :math:`D`
are the data. We setup the model as before:

.. code-block:: python

	fresco_path = './48Ca_elastic_new.in'
	fresco_names = ['p1', 'p2', 'p3', 'p4', ('p5', 'p5'), ('p6', 'p6'), 'p4']
	fresco_positions = [54, 55, 56, 57, (58, 66), (59, 67), 65]
	model = pfunk.model.Model(fresco_path, fresco_names, fresco_positions)

Now we must assign priors to the potential parameters, which is done using the
method :meth:`pfunk.model.Model.create_pot_prior`. This method will create a normal
prior for each of the parameters defined in by the :code:`fresco_names` and :code:`fresco_positions`.
As will be shown later, these priors need to be informative in order to eliminate the pathological behavior
of the parameters, but for now lets just center them around the global values and give them :math:`100 \%`
widths. This can be easily with the command:

.. code-block:: python
		
   model.create_pot_prior(model.fresco.x0, model.fresco.x0)

Since we just have priors on the potentials in this simple example, we can go ahead and tell
pfunk we are done defining the priors with :meth:`pfunk.model.Model.create_priors` :

.. code-block:: python
		
   model.create_priors()

The likelihood function is defined precisely as before:

.. code-block:: python
		
   model.create_elastic_likelihood('fort.201', elastic_data_path)
   model.create_likelihood()

Now our model has a defined log-probability that can be called by whatever minimization or
sampling method we want to use. The simplest way to sample this posterior is to use `emcee <https://github.com/dfm/emcee>`_ .
:code:`emcee` is an ensemble sampler and has the best results when the walkers are initialized in a high probability region
of the posterior. With this in mind, we again use :class:`pfunk.model_fit.MAPFit`, but this time we want to minimize the
posterior not just the likelihood. Since pfunk is designed with Bayesian methods in mind, this is the default behavior
of this class. Thus, we do the following:

.. code-block:: python

   fit = pfunk.model_fit.MAPFit(model, percent_range=5.0)
   fit.run_anneal(max_iter=1000)

Notice that we did not have to specify :code:`model.x0` this time. When priors are defined for the parameters, the model
object automatically defines :code:`model.x0` based on the means of the normal distribution. Once the minimization has
finished, update :code:`model.x0` based on the results of the fit.

.. code-block:: python

   model.x0 = fit.results.x[:]

Now with the setup all taken care of we can define our sampler. The methods to call the samplers are all defined in
:class:`pfunk.sampler.Sampler`. Initial an object by giving it the model instance and setting up the number of walkers
and steps you want:

.. code-block:: python

   sampler = pfunk.sampler.Sampler(model)
   sampler.nwalker = 200
   sampler.nstep = 1000

Each of these walkers needs to be given an initial starting position. To spread them out in a random ball around the
MAP fit value call :meth:`pfunk.sampler.Sampler.ball_init`, check to make sure all these walkers have finite log-probabilities
with :meth:`pfunk.sampler.Sampler.check_p0`. Now all that is left to do is run the sampler.

.. warning::

   Any sampling will be computationally expensive! Depending on the FRESCO calculation and number of likelihood calls
   expect anywhere from hours to days of run time!

To run :code:`emcee` just call :meth:`pfunk.sampler.Sampler.run_ensemble`:

.. code-block::
   
   sampler.run_ensemble()
