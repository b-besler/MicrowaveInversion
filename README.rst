Microwave Inversion
===================
Algorithms for inverse microwave problems.

|Build Status|_

.. |Build Status| image:: https://dev.azure.com/brendonbesler/brendon_besler/_apis/build/status/b-besler.MicrowaveInversion?branchName=master
.. _Build Status: https://dev.azure.com/brendonbesler/brendon_besler/_build/latest?definitionId=1&branchName=master

Install
=================
Clone repository to local machine:

.. code-block:: base

  $ git clone https://github.com/b-besler/MicrowaveInversion.git

Create conda environment `mwi` to work from:

.. code-block:: bash

  $ conda create -n mwi python=3.7
  $ conda activate mwi
  $ conda install --file requirements.txt
  $ pip install -e .

To remove the conda environment:

.. code-block:: bash

  $ conda deactivate
  $ conda env remove -n mwi
Running Tests
=============

.. code-block:: bash

  $ nosetests tests/ #while in MicrowaveInversion directory
Running Inverse Solve
=====================

The inverse solver can be called via the command line as follows:
 
.. code-block:: bash

  $ inverse path-to-config/model1_config.json path-to-config/model2_config.json path-to-config/meas_config.json path-to-config/image_config.json path-to-output

See example folder for example configuration files.
 

