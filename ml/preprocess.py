#
#   Author: Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>, Badr-Eddine Bouhlal
#
import numpy
import pandas

use_gflops_as_target = True

def norm_vector(x):
    """
    It's beneficial to rescale the input/output space.
    """
    # interval [0..1]
    return (x-numpy.amin(x))/(numpy.amax(x)-numpy.amin(x))

def time_to_gflops(times, nnz):
    """
    transforms the computation time in seconds to GFLOPs.

    ATTENTION: num_iter needs to be adjusted, if the simulation period is changed!
    """
    num_iter = 1000.0
    return ( num_iter*2.0*(nnz) / times / (10**9) )

def preprocess_regression(filename, csv_labels, frac=0.8):
    """
    Create a training and test set from a given dataset.

    Params:

    * filename: dataset as csv
    * frac: fraction of training set (by default 80%)

    Returns:

    * 4 pandas data frames   
        
        * X_train, X_test, t_train, t_test: required for NN

    * and three lists:

        * min_values, max_values: need to be stored, so that the predictor can also normalize
        * indices: integer indices of those data rows which are part of the trainingsset
    """
    # which columns are features/outputs
    feature_labels = csv_labels[:7]
    output_labels = csv_labels[7:10]    # we concentrate here only on csr/ellr/dense

    # read in csv data
    print(csv_labels)
    print(feature_labels)
    print(output_labels)
    DataFrame = pandas.read_csv( filename, sep=',', names=csv_labels, usecols=[*range(0, 10)])

    # Normalize the feature values per column.
    # The output labels are already in interval [0..1]
    NormDataFrame = DataFrame.copy()
    min_values = []
    max_values = []

    # Transform output (times->gflops)
    if use_gflops_as_target:
        for output in output_labels:
            # HD: I tested an output normalization but this worsened the simulation results
            gflops = time_to_gflops(NormDataFrame[output], NormDataFrame['overall number nonzeros'])
            NormDataFrame[output] = gflops

    else:
    # HD:
    # when we record time in seconds, then the values are quite small
    # one could try to rescale them, but this will interfere  e. g. with single-CPU
    # I still believe, that the usage of GFLOPs is the better way ...
        #for output in output_labels:
        #    NormDataFrame[output] = NormDataFrame[output]*1000
        pass

    # Normalize features
    for feature in feature_labels:
        # store min/max in advance
        min_values.append(numpy.amin(NormDataFrame[feature]))
        max_values.append(numpy.amax(NormDataFrame[feature]))

        # norm the vector
        NormDataFrame[feature] = norm_vector(NormDataFrame[feature])

    # Split the Data into training and test set
    train = NormDataFrame.sample(frac=frac) # random_state: provide np.random.RandomState to fixate
    test = NormDataFrame.drop(train.index)

    # seperate features / output
    X_train = train[feature_labels] 
    X_test = test[feature_labels] 

    t_train = train[output_labels]
    t_test = test[output_labels]

    return X_train, X_test, t_train, t_test, min_values, max_values, train.index