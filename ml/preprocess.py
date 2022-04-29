#
#   Author: Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>, Badr-Eddine Bouhlal
#
import numpy
import pandas
import labels
from sklearn.model_selection import train_test_split

def preprocess_complete_data(filename, use_gflops_as_target=True):
    """
    Read out a pandas file and transform computation into GFLOPs if requested.
    """
    data = pandas.read_csv( filename, sep=',', names=labels.csv_labels, header=None, index_col=False)
    print("Target data:")
    if use_gflops_as_target:
        for label in labels.outputs:
            time_in_sec = data[label]
            nnz = data['overall number nonzeros']
            
            data[label] = _time_to_gflops(time_in_sec, nnz)
            print("  ", label, "GFLOPs: min =", numpy.amin(data[label]), ", max =", numpy.amax(data[label]))
    else:
        print(label, "time: min =", numpy.amin(data[label]), ", max =", numpy.amax(data[label]))

    return data

def preprocess_regression(filename, frac=0.8, use_gflops_as_target=True):
    """
    Create a training and test set from a given dataset.

    Params:

    * filename: dataset as csv
    * frac: fraction of training set (by default 80%)

    Returns:

    * 4 numpy arrays required for NN training (X_train, X_test, t_train, t_test)

    * 1 numpy array with the indices of test data points

    """
    data = preprocess_complete_data(filename, use_gflops_as_target)

    X = data[labels.features]
    y = data[labels.outputs]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-frac)
    test_indices = y_test.index

    # pandas data frames -> numpy arrays
    X_train = X_train.to_numpy().astype(numpy.float32)
    X_test = X_test.to_numpy().astype(numpy.float32)
    y_train = y_train.to_numpy().astype(numpy.float32)
    y_test = y_test.to_numpy().astype(numpy.float32)
    test_indices = test_indices.to_numpy().astype(numpy.int32)

    return X_train, X_test, y_train, y_test, test_indices

def get_evaluation_data(filename, indices, use_gflops_as_target=True):
    """
    Retrieve the test data points used during model training.

    Returns:

    * 2 numpy arrays required for NN validation

    * 1 numpy array containing the labels selected by "auto" in ANNarchy (transformed to integers)
    """
    data = preprocess_complete_data(filename, use_gflops_as_target)

    X = data[labels.features].to_numpy().astype(numpy.float32)
    y = data[labels.outputs].to_numpy().astype(numpy.float32)
    
    # replace labels by their integer ID
    idx = data.index[data['auto (label)']=="csr"].tolist()
    data.loc[idx]=0

    idx = data.index[data['auto (label)']=="ellr"].tolist()
    data.loc[idx]=1

    idx = data.index[data['auto (label)']=="dense"].tolist()
    data.loc[idx]=2
    
    # pandas data frames -> numpy arrays
    auto_test = data["auto (label)"].to_numpy().astype(int)[indices]
    X_test = X[indices]
    y_test = y[indices]

    return X_test, y_test, auto_test

def _time_to_gflops(times, nnz, num_iter = 1000.0):
    """
    Helper function: transforms the computation time in seconds to GFLOPs.

    ATTENTION: num_iter needs to be adjusted, if the simulation period is changed!
    """
    return ( ((num_iter*2.0*nnz) / times) / (10**9) )
