# Dinkelbach2021_ANNarchyAutoTuning

Source code of simulations and analyses from Dinkelbach, Bouhlal, Vitay & Hamker (2021) submitted to *Frontiers in Neuroinformatics*

## Authors

* Helge Ülo Dinkelbach (helge.dinkelbach@gmail.com)
* Badr-Eddine Bouhlal ()

## Structure

- **dense_vs_sparse**:

    - A comparison of a dense- and compressed sparse row matrix format.

        - pre-requisites (Python packages): csv, matplotlib, ANNarchy >= 4.7.1

        - usage: python [number rows=int] [number columns=int] [fmt=dense|csr|auto] [paradigm=openmp|cuda] [SIMD(openmp)=0(on)|1(off)]

        - for multiple data points you can simply re-run the same configuration multiple times as the results will be written continuously in a file (please see run.sh for an example)

        - to recreate the figure from the article, you might want to use *sh run.sh* and then *python plot_dense_vs_sparse.py*

        - to modify the tested configurations, you can simply modify

- **generate_data**:

    - Generate the data set for the training of the neural network

        - pre-requisites (Python packages): csv, ANNarchy >= 4.7.1

        - usage of *python []* to create 1 data point (repeated executions will fill the same dataset file)

    - to reproduce the relative performance plot (Fig. 4) you can run datasets/plot_relative_performance.py followed by the (relative/absolute) path + name of the .csv

- **ml**

    - Contains all the files related to the neural network / evaluations

        - pre-requisites (Python packages): csv, ANNarchy >= 4.7.1, Keras >= 2.6.0 (we run all experiments with 2.6.0, we are unaware how large issues with up/down-ward compability are)

- **datasets**

    - intended to store measurement results obtained by the generate_data/measure.py script

- **figures**

    - inteneded to store the figures produced by several scripts

## Reproduction of the figures

- dense vs sparse (Fig. 3) :

    repeated execution of *measure.py* in *dense_vs_sparse/* (see run.sh for details). Execution of plot.py will create Fig. 3

- spmv dataset generation :

    - first we need to produce the dataset:

        - execute *python configurations.py [number]* where number is the number of configurations which should be generated

        - then call run.sh (you can modify storage folder/name within this script)
    
- Figure 4:

    - after the dataset was generated, call the *plot_relative_performance* script in *generate_data* which should generate Fig. 4

- Figure 5:

    - this figure is a result of the *predict.py* script in *ml*

- Figure 6:

    - this figure is a result of the *cross_validation.py* script in *ml*

- Figure 7:

    - this figure is a result of the *plot_cross_validation.py* script in *ml* after execution of cross_validation.py with different dataset sizes (see the run_var_ds.sh for details)
