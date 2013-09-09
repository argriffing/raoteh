
requires
--------
 * python
 * numpy
 * scipy (for generic expm, xlogy)
 * networkx (for infinite dimensional sparse matrices)
 * pyfelscore (a non-standard python module,
   for special-cased expm and expm-frechet)


organization
------------

At the time of writing,
the functions implemented in this project are organized into three sections:
 * discrete-time, discrete-space Markov chains
 * continuous-time, discrete-space Markov jump processes
 * a specific continuous-time Bayesian network ('tolerance' Markov jump process)

These abstractions have been generalized from the traditional
'path' or 'interval' domain to tree-like domains
for which the same kinds of algorithms apply;
to the degree that the computational complexity depends on treewidth,
there is no difference in this regard between path and tree domains.

For each section, two sets of functions have been implemented:
 * deterministic calculations such as likelihood or closed-form expectation
 * stochastic calculations like sampling for Monte-Carlo

Tests should be available for most of the deterministic calculations
and for some of the stochastic calculations.


how observations are handled
----------------------------

The notation for partial observations of the process state
has been ad-hoc, and has been proceeding through the following
three stages:
 * a sparse map from a node to its observed state
 * a map from a node to a set of states allowed by the observation
 * a map from a node to a map from allowed states to associated likelihoods

The last of these three stages is essentially
the setup that is standard for hidden Markov models.


installation
------------

Personally I install this module by changing the working directory
to the checked out git source directory and running the following command.

    $ python setup.py install --user

I've had trouble installing with `pip` so maybe something is wrong with
my setup.py files.

