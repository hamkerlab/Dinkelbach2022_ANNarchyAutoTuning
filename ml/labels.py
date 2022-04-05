#
#   Author: Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>, Badr-Eddine Bouhlal
#
"""
HD: I'm aware the fact, that this is unhandy. But I'm too short on time yet, to
restructure the implementation now.

In future, this labels should be stored in the .csv directly
"""

import csv

csv_labels = [
    # classic stats
    "number rows", "number columns", "overall number nonzeros", "density",
    # nonzeros distribution
    "mean nnz per row", "min nnz per row", "max nnz per row", 
    # GPUs
    "csr (gpu)", "ellr (gpu)", "dense (gpu)", "auto (label)", "auto (times)"
]

features = [
     "number rows", "number columns", "overall number nonzeros", "density",
     "mean nnz per row", "min nnz per row", "max nnz per row",
]

outputs = [
     "csr (gpu)", "ellr (gpu)", "dense (gpu)",
]
