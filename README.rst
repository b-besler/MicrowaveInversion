Microwave Inversion
===================
Algorithms for inverse microwave problems.

|Build Status|_

.. |Build Status| image:: https://dev.azure.com/brendonbesler/brendon_besler/_apis/build/status/b-besler.MicrowaveInversion?branchName=master
.. _Build Status: https://dev.azure.com/brendonbesler/brendon_besler/_build/latest?definitionId=1&branchName=master

Developer Install
=================
Creates conda environment `mwi` to work from

.. code-block:: bash

  $ conda create -n mwi python=3.7
  $ conda activate mwi
  $ conda install --file requirements.txt
  $ pip install -e .

To remove the conda environment:

.. code-block:: bash

  $ conda deactivate
  $ conda env remove -n mwi
