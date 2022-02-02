# Dinkelbach2022_ANNarchyAutoTuning

Source code of simulations and analyses from Dinkelbach, Bouhlal, Vitay & Hamker (2022) submitted to *Frontiers in Neuroinformatics*

## Authors

* Helge Ülo Dinkelbach (helge.dinkelbach@gmail.com)
* Badr-Eddine Bouhlal (badreddinebouhlal@gmail.com)

## Pre-requisites

Pre-requisites are the following Python packages: csv, matplotlib, ANNarchy >= 4.7.1, Keras >= 2.6.0 (we run all experiments with Keras 2.6.0, we are unaware how large issues with up/down-ward compability are).

## Folder Structure

- **dense_vs_sparse**:

    Contains files which are related to the comparison of the dense- and compressed sparse row matrix format.

- **generate_data**:

    Generate the data set for the training of the neural network, see section *spmv dataset generation for more details*.

- **ml**

    Contains all the files related to the neural network and the corresponding evaluations.

- **datasets**

    Intended to store measurement results obtained by the generate_data/measure.py script.

- **figures**

    Inteneded to store the figures produced by several scripts.

## Spmv Dataset Generation :

For the most evaluations, we need to produce the dataset first. The procedure is split into two parts: first the configurations and then the measurement. This splitting allows to create a set of configurations which can be run on several machines and allows to generate a comparable dataset.

1st step: execute *python configurations.py [number]* where number is the number of configurations which should be generated

2nd step: call run.sh, by default the resulting .csv file is stored in *datasets*. You can modify both storage folder and name in the .sh but please note, that the other scripts in *ml* need to be adjusted too.

## Reproduction of the figures

- Figure 3:

    - usage: python [number rows=int] [number columns=int] [fmt=dense|csr|auto] [paradigm=openmp|cuda] [SIMD(openmp)=0(on)|1(off)]

    - for multiple data points you can simply re-run the same configuration multiple times as the results will be written continuously in a file (please see *run.sh* for an example)

    - to recreate the figure from the article, you might want to use *sh run.sh* and then *python plot_dense_vs_sparse.py* this will create Figure 3.

- Figure 4:

    - after the dataset was generated, call the *plot_relative_performance* script in *generate_data* which should generate Fig. 4

- Figure 5:

    - this figure is a result of the *predict.py* script in *ml*

- Figure 6:

    - this figure is a result of the *cross_validation.py* script in *ml*

- Figure 7:

    - this figure is a result of the *plot_cross_validation.py* script in *ml* after execution of cross_validation.py with different dataset sizes (see the run_var_ds.sh for details)
