Nested Sampling
===============

As seen in the last section, a reasonable fit to the data can be made
just considering optical model uncertainties, but there appear to be multiple
modes in the posterior and low angle data is not well described. Before expanding
the Bayesian model to rectify these issues it is useful to gain more information
about the structure of the posterior. The best way to do this is using nested sampling
to explore a larger portion of the posterior. pfunk uses the nested sampling package
`dynesty <https://github.com/joshspeagle/dynesty>`_ to do this. This calculation
is setup in the same way MCMC calculation with the exception that we do not need
to do a MAP fit. Nested sampling requires the likelihood and priors to be calculated separately.
Additionally, the priors need to be mapped from :math:`[0,1)` to the pdf of the priors. However, pfunk does
all of this automatically, so just setup the model:

.. code-block:: python

   
   fresco_path = '48Ca_elastic_new.in'
   fresco_names = ['p1', 'p2', 'p3', 'p4', ('p5', 'p5'), ('p6', 'p6'), 'p4']
   fresco_positions = [54, 55, 56, 57, (58, 66), (59, 67), 65]
   elastic_data_path = '48Ca_p_p.dat'

   model = pfunk.model.Model(fresco_path, fresco_names, fresco_positions) 
   model.create_pot_prior(model.fresco.x0, model.fresco.x0) 
   model.create_prior() 

   model.create_elastic_likelihood('fort.201', elastic_data_path)
   model.create_likelihood()

Now create an instance of a sampler object

.. code-block:: python

   sampler = pfunk.sampler.Sampler(model)

We want to use static nested sampling for this calculation. The default
stopping condition is :math:`\Delta Z = 0.01` with :math:`250` live points. 


.. code-block:: python

   sampler.run_nested()


