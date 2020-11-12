Nested
======

.. code:: ipython3

    %matplotlib inline
    import pfunk
    import matplotlib.pyplot as plt
    import numpy as np
    import corner

.. code:: ipython3

    fresco_path = '48Ca_elastic_new.in'
    fresco_names = ['p1', 'p2', 'p3', 'p4', ('p5', 'p5'), ('p6', 'p6'), 'p4']
    fresco_positions = [54, 55, 56, 57, (58, 66), (59, 67), 65]
    elastic_data_path = '48Ca_p_p.dat'


.. code:: ipython3

    model = pfunk.model.Model(fresco_path, fresco_names, fresco_positions) # create model
    model.create_pot_prior(model.fresco.x0, model.fresco.x0) # 100% priors from global values
    model.create_prior() # Finish setting up the prior functions
    
    model.create_elastic_likelihood('fort.201', elastic_data_path)
    model.create_likelihood()


.. code:: ipython3

    sampler = pfunk.sampler.Sampler(model)

.. code:: ipython3

    sampler.run_nested()


.. parsed-literal::

    8324it [7:05:14,  3.07s/it, +250 | bound: 203 | nc: 1 | ncall: 1416350 | eff(%):  0.605 | loglstar:   -inf <  3.893 <    inf | logz: -24.725 +/-  0.450 | dlogz:  0.000 >  0.010]


.. code:: ipython3

    s = pfunk.utilities.make_samples_dynesty(sampler.sampler.results)

.. code:: ipython3

    labels = ['$V$', '$r_0$', '$a_0$', '$W$', '$r_i$', '$a_i$', '$W_S$']
    p = corner.corner(s, bins=50, smooth=1.5, labels=labels,
                      show_titles=True, title_kwargs={'fontsize':17.0},
                      label_kwargs={'fontsize':17.0})




.. image:: nested_files/nested_6_0.png

And with this result we can better understand why the MCMC samples showed some pathological
behaviour. 
