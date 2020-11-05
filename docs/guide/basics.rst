Maximum Likelihood Fit ( i.e., a :math:`\chi^2`-fit)
====================================================

The first example will be a simple :math:`\chi^2`-fit, where the
optical model parameters will be adjusted to reproduce observed elastic
scattering data. In this case I am going to look at the :math:`^{48}` Ca
:math:`(p, p)` at :math:`25` MeV. This data comes from Ref. [Mc1986]_. 
Just selecting the ... optical potential, we first have to create a FRESCO
file that is compatible with pfunk. In particular, pfunk swaps variable values
based on line numbers. While this can seem cumbersome, the upshot is that
swapping based on line number eliminates any computational overhead with doing
find and replace. Below the relevant lines are highlighted.
      
.. code-block:: none
  :linenos:
  :lineno-start: 0
  :emphasize-lines: 55,56,57,58,59,60,66,67,68

     48Ca(p,p) 25.0 MeV
     NAMELIST
     &fresco
     hcm = 0.1
     rmatch = 40.0
     jtmax = 20
     absend = 0
     jump(1:6) = 0, 0, 0, 0, 0, 0
     jbord(1:6) = 0, 0, 0, 0, 0.0, 0.0
     kqmax = 1
     thmin = 0.0
     thmax = 180.0
     thinc = 1.0
     it0 = 1
     smats = 2
     xstabl = 1
     elab(1:4) = 25.0, 0, 0, 0
     nlab(1:3) = 0, 0, 0
     /

     &partition
     namep = 'p'
     massp = 1
     zp = 1
     namet = 'Ca'
     masst = 48
     zt = 20
     qval = 0.0
     pwf = .false.
     nex = 1
     /


     &states
     jp = 0.5
     bandp = 1
     cpot = 1
     jt = 0.0
     bandt = 1
     /


     &partition
     /

     &pot
     kp = 1
     at = 48
     rc = 1.25
     /

     &pot
     kp = 1
     type = 1
     p1 = 52.2
     p2 = 1.17
     p3 = 0.75
     p4 = 2.8
     p5 = 1.32
     p6 = 0.63
     /

     &pot
     kp = 1
     type = 2
     p4 = 7.6
     p5 = 1.32
     p6 = 0.63
     /

     &pot
     kp = 1
     type = 3
     shape = 0
     p1 = 6.2
     p2 = 1.01
     p3 = 0.75
     /

     &pot
     /

     &overlap
     /

     &coupling
     /


In order for pfunk to swap these values we need to specify the input file path, line number, and variable name.
The most convenient way to do this for the line number and names is using lists. 

.. code-block:: python

	fresco_path = './48Ca_elastic_new.in'
	fresco_names = ['p1', 'p2', 'p3', 'p4', ('p5', 'p5'), ('p6', 'p6')]
	fresco_positions = [54, 55, 56, 57, (58, 66), (59, 67)]

Note that the line numbers above start at zero. The names and numbers in parentheses
tell pfunk to swap one number in at two positions in the input file. This is done because 
often global optical potentials use the same radius and diffuseness parameters
for the imaginary surface and volume potentials:

.. math::

   W_{total}(r) = (4 a_i W_s + W)f(r;r_i, a_i).

The parenthesis enforce this condition by swapping the enclosed parameters
simultaneously.


Now it is time to initialize our model. The first step is to create an instance
of a :class:`pfunk.model.Model` object:

.. code-block:: python

	model = pfunk.model.Model(fresco_path, fresco_names, fresco_positions)


Since this is a :math:`\chi^2`-fit the only remaining task is to assign a likelihood
function. The trick is that a :math:`\chi^2`-fit is simply the log-likelihood of a
normal distribution. There are several methods in the model object to assist in
the creation of the necessary probability functions. For a likelihood function
all that needs to be specified is the path of the fresco output, the path to the data
that is to be fit, and any additional model parameters (which I will come back to later).
The data we want to fit  should be in a text file
with labeled columns of angle, cross section, and uncertainty. As an example here are
the first four lines of the data file.

.. code-block:: none

   theta        sigma       erry
   10.19108     0.80674     0.080674
   12.95117     0.89819     0.089819
   15.28662     1.012       0.1012

Next, by default, the FRESCO elastic scattering cross section is written to fort.201.
To add a likelihood function based on elastic scattering call the method
:meth:`pfunk.model.Model.create_elastic_likelihood`:

.. code-block:: python

   model.create_elastic_likelihood('fort.201', elastic_data_path)

Finally, pfunk needs to know that we are done adding likelihood functions so
that it can create the total likelihood function.

.. code-block:: python

   model.create_likelihood()

We now what to fit the optical model parameters to the data. Minimization is
provided by the module :mod:`pfunk.model_fit` , and specifically a model can be
minimized using the class :class:`pfunk.model_fit.MAPFit`. First the model needs
to have starting parameters. The easiest way to do this if we just have potential
parameters is to pull them from the :class:`pfunk.fresco_classes.NamelistInput` object which is generated automatically by the model. This object stores the initial parameters, and this list can be copied to the model objects initial parameters with: 

.. code-block:: python

   model.x0 = model.fresco.x0[:]

We now initialize the fit object:


.. code-block:: python

   fit = model.funk.model_fit.MAPFit(model, lnlike=True, percent_range=5.0)

where we have told MAPFit object to minimize just the likelihood via :code:`lnlike=True` and to
consider a global minimization in a range of :code:`5.0*model.x0` up and down with :code:`percent_range=5.0`.
Because :code:`model.x0` controls the search range of the global minimization, different starting values
can be chosen by changing the array :code:`fit.x0_custom`. With this we simply need to run the minimization, which
can be done using basin hopping, :meth:`pfunk.model_fit.MAPFit.run`, differential evolution,
:meth:`pfunk.model_fit.MAPFit.run_differential`, or dual annealing, :meth:`pfunk.model_fit.MAPFit.run_anneal`.
Annealing usually provides the best compromise between speed and search range:

.. code-block:: python

   fit.run_anneal(max_iter=1000)

This will print out a message after each iteration, but the message will not make too much sense. The first value
is the current value of likelihood function, which you should check to make sure everything is still going smoothly.
After this finishes the minimized parameter values can be found in the array :code:`fit.results.x`. The below notebook
shows how to do this calculation and plot the results.


   
.. [Mc1986] https://journals.aps.org/prc/abstract/10.1103/PhysRevC.33.1624.
