.. bead documentation master file, created by
   sphinx-quickstart on Tue Feb 11 21:09:46 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

BEAD documentation
==================

**Background Enrichment for Anomaly Detection** || *{Background Enrichis pour (la Détection d'Anomalies) Anomalie Détection}*

BEAD is a Python package that uses deep learning based methods for anomaly detection in HEP data for new physics searches.
BEAD has been designed with modularity in mind, to enable usage of various unsupervised latent variable models for any task.

BEAD has five main running modes:

   1. Data handling: Deals with handling file types, conversions between them and pre-processing the data to feed as inputs to the DL models.

   2. Training: Train your model to learn implicit representations of your background data that may come from multiple sources/generators to get a single, encriched latent representation of it.

   3. Inference: Using a model trained on an enriched background, feed in samples you want to detect anomalies in and watch the magic happen.

   4. Plotting: After running Inference, or Training you can generate plots similar towhat is shown in the paper. These include performance plots as well as different visualizations of the learned data.

   5. Diagnostics: Enabling this mode allows running profilers that measure a host of metrics connected to the usage of the compute node you run on to help you optimize the code if needed(using CPU-GPU metrics).

For more information, follow the install instructions here: https://github.com/PRAkTIKal24/BEAD and from ./BEAD/bead/ run the command ```poetry run bead -h``` to get detailed instructions on running the package and the available customizations

.. note::

   This project is under active development.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`