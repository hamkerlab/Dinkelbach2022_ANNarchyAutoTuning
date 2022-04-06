# Dinkelbach2022_ANNarchyAutoTuning

Source code of simulations and analyses from Dinkelbach, Bouhlal, Vitay & Hamker (2022) submitted to *Frontiers in Neuroinformatics*

## Authors

* Helge Ülo Dinkelbach (helge.dinkelbach@gmail.com)
* Badr-Eddine Bouhlal (badreddinebouhlal@gmail.com)

## Pre-requisites

Pre-requisites are the following Python packages: csv, pandas, matplotlib, ANNarchy >= 4.7.1, TensorFlow >= 2.6.2 (we run all experiments with TensorFlow 2.6.2, we are unaware how large issues with up/down-ward compability are), Optuna 2.10.0 library and scikit-learn package 0.23.1.

For more informations to the frameworks, please visit the project homepages:

- ANNarchy: https://www.tu-chemnitz.de/informatik/KI/projects/ANNarchy/

- Optuna: https://optuna.org/

- TensorFlow: https://www.tensorflow.org/

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

## SpMV Dataset Generation

For the most evaluations, we need to produce the dataset first. The procedure is split into two parts: first the configurations and then the measurement.  This splitting allows to create a set of configurations which can be run on several machines and allows to generate a comparable dataset.

The data generation can be started by simply call *run.sh*, by default the resulting .csv file is stored in *datasets*. You can modify both storage folder and name in the .sh but please note, that the other scripts in *ml* need to be adjusted too.

If you want to modify the compiler flags, or the CUDA compiler is not installed in a default place you can use the *annarchy.json* file to influence the compilation of ANNarchy.

## Training the ML models

After generating the dataset we can train the machine learning models. This split up into two tasks, first we need to find a network configuration using the Optuna package part of the *optimize_network.py* script in *ml*:

    python optimize_network.py [dataset] [folder name]

The optimized configuration is then stored in *folder name*. The training of a model is then performed using the *trainging.py* script in *ml*:

    python training.py [dataset] [folder name]

where the *dataset* is the filename as well as either absolute or relative path (e.g. ../datasets/my_data.csv) and folder name is the folder where the learned keras network should be stored. This should be the same folder as the optimized network was stored.

For our default dataset, you can use the *run_training.sh* script in *ml*. To configure the training (e. g. number of epochs, trials for optuna optimization, k for the k-fold cross validation) you can modify the lines 35-40 in *optimize_network.py* and the lines 34-35 in *training.py*

## Reproduction of the figures

- Figure 3:

    - usage: python measure.py [number rows=int] [number columns=int] [fmt=dense|csr|auto] [paradigm=openmp] [SIMD(openmp)=0(on)|1(off)] [target_folder=str]

    - for multiple data points you can simply re-run the same configuration multiple times as the results will be written continuously in a file (please see *run.sh* for an example)

    - to recreate the figure from the article, you might want to use *sh run_cpu.sh* and then *python plot_cpu.py* this will create Figure 3.

- Figure 4:

    - usage: python measure.py [number rows=int] [number columns=int] [fmt=dense|csr|auto] [paradigm=cuda] [target_folder=str]

    - for multiple data points you can simply re-run the same configuration multiple times as the results will be written continuously in a file (please see *run.sh* for an example)

    - to recreate the figure from the article, you might want to use *sh run_gpu.sh* and then *python plot_gpu.py* this will create Figure 4.

- Figure 5:

    - for the provided default dataset call the *plot_relative_performance.py* script in *datasets* which should generate Fig. 5 and store it in *figures*.
      Please note, for an own dataset you need to adapt line 13 in the *plot_relative_performance.py* script.

- Figure 6:

    - this figure is a result of the *predict.py* script in *ml*

- Figure 7:

    - this figure is a result of the *cross_validation.py* script in *ml*

- Figure 8:

    - this figure is a result of the *plot_cross_validation_var_ds.py* script in *ml* which calls several times *cross_validation.py* with different dataset sizes (see the *run_var_ds.sh* script for details)

- Supplementary figures:

    - run *plot_memory_estimation.py* to generate Figure S1 of the supplementary material. The dataset can be changed in line 16.

    - run *plot_data_analysis.py* to generates Figure S2 and S3 of the manuscript. The datasets can be changed in line 66 and 67.

## Creating/Adding an own dataset

By default, the configuration of the article (*article.csv*) is used. If you want to create an own set, you need to execute *python configurations.py [number]* where number is the number of configurations which should be generated. Then you need to replace *article.csv* in line 4 of the shell script (*run.sh*) by *configurations.csv*
