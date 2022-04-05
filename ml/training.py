#
#   Author: Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>, Badr-Eddine Bouhlal
#
import pandas
import numpy
import csv
import sys
import tensorflow.keras as keras
import matplotlib.pylab as plt
import optuna
import pickle
import preprocess
import labels
from sklearn.model_selection import KFold

if len(sys.argv) < 3:
    print("Expected at least two arguments:")
    print("")
    print(" - a dataset file (*.csv) either relative or absolute path")
    print(" - a target folder to store the model")
    print(" - optional: the number of elements used from the dataset")
    exit()

#
# Dataset
dataset = sys.argv[1]
model_folder = sys.argv[2]
num_red_datapoints = -1
if len(sys.argv) == 4:
    num_red_datapoints = int(sys.argv[3])

#
# Experiment parameter
NUM_EPOCHS_PER_TRIAL = 150
BATCH_SIZE = 128
USE_GFLOPS_AS_TARGET = True

#
# The dataset is split during the training (k-fold cross validation)
data = preprocess.preprocess_complete_data(dataset, USE_GFLOPS_AS_TARGET)
X = data[labels.features].to_numpy().astype(numpy.float32)
y = data[labels.outputs].to_numpy().astype(numpy.float32)

if num_red_datapoints != -1:
    X = X[:num_red_datapoints]
    y = y[:num_red_datapoints]

print(dataset, model_folder, num_red_datapoints)

# load the model configuration determined by optuna
with open(model_folder+'/model_config.data', 'rb') as f:
    model_dict = pickle.load(f)
learning_rate = numpy.recfromtxt(model_folder+'/learning_rate.csv')

# create the model          
model = keras.Sequential.from_config(model_dict)

# TODO: is this correct ???
norm_layer = model.get_layer("normalization")
norm_layer.adapt(X)

# compile the model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
    loss=keras.losses.MeanSquaredError(), 
    metrics=['mse']
)

print("-- Model Structure --")
model.summary()

X_train, X_test, y_train, y_test, index_test = preprocess.preprocess_regression(dataset, 0.8, USE_GFLOPS_AS_TARGET)
history = model.fit(X_train, y_train, epochs=NUM_EPOCHS_PER_TRIAL, batch_size=BATCH_SIZE, verbose=0)

# compute the prediction
y_pred = model.predict(X_test, batch_size=BATCH_SIZE)
if USE_GFLOPS_AS_TARGET:
    best_data = numpy.argmax(y_test, axis=1)
    best_predicted = numpy.argmax(y_pred, axis=1)
else:
    best_data = numpy.argmin(y_test, axis=1)
    best_predicted = numpy.argmin(y_pred, axis=1)
print("Accuracy: ", numpy.sum(best_predicted == best_data) / y_test.shape[0])

# Save model and training indices (for prediction quality analysis)
model.save(model_folder)
with open(model_folder+"/indices.csv", mode='w') as Datafile:
    Exe_writer = csv.writer(Datafile, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    Exe_writer.writerow(index_test)
