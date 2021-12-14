#
#   Author: Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>, Badr-Eddine Bouhlal
#
"""
HD: I'm aware the fact, that this is unhandy. But I'm too short on time yet, to
restructure the implementation now.

In future, this labels should be stored in the .csv directly
"""

import csv

csv_labels_K20m = [
    # classic stats
    "number rows", "number columns", "overall number nonzeros", "density",
    # nonzeros distribution
    "mean nnz per row", "min nnz per row", "max nnz per row", 
    # GPUs
    "csr (gpu)", "ellr (gpu)", "dense (gpu)", "auto (label)", "auto (times)"
]

csv_labels_RTX3080 = [
    # classic stats
    "number rows", "number columns", "overall number nonzeros", "density",
    # nonzeros distribution
    "mean nnz per row", "min nnz per row", "max nnz per row", 
    # GPUs
    "csr (gpu)", "ellr (gpu)", "dense (gpu)", "auto (label)", "auto (times)"
]

def get_csv_labels(dataset_name):
    "returns a list of labels which contains labels for all columns in the .csv"
    if dataset_name == "nvidia_K20m.csv":
        csv_labels = csv_labels_K20m
    elif dataset_name == "nvidia_RTX3080.csv":
        csv_labels = csv_labels_RTX3080
    else:
        raise ValueError("No csv labels set for:", dataset_name)

    return csv_labels

def get_csv_labels_output(dataset_name):
    "returns a list of labels used for the neural network, in our case csr, ellr, dense"
    if dataset_name == "nvidia_K20m.csv":
        csv_labels = csv_labels_K20m[7:10]
    elif dataset_name == "nvidia_RTX3080.csv":
        csv_labels = csv_labels_RTX3080[7:10]
    else:
        raise ValueError("No csv labels set for:", dataset_name)

    return csv_labels