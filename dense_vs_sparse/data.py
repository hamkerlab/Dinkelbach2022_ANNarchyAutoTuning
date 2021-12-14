#
#   Author: Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com
#
#   This file is intended to synchronize the configuration between measure/plot

# a qualifying name, in this case the GPU which was measured.
# Will be the leading part of the result file.
dataset = "dense_vs_sparse"

# The probabilities which should be measured. In case of a squared (!) matrix
# this will equal the density
conf = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
