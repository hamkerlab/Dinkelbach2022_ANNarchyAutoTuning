#
#   Author: Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>, Badr-Eddine Bouhlal
#
import pandas
import numpy
import csv
import os
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
print(dataset, model_folder, num_red_datapoints)

#
# Experiment parameter
NUM_TRIALS_OPTUNA = 150
NUM_EPOCHS_PER_TRIAL = 200
BATCH_SIZE = 128
USE_GFLOPS_AS_TARGET = True
NUM_FOLDS_CV = 5

#
# The dataset is split during the training (k-fold cross validation)
data = preprocess.preprocess_complete_data(dataset, USE_GFLOPS_AS_TARGET)
X = data[labels.features].to_numpy().astype(numpy.float32)
y = data[labels.outputs].to_numpy().astype(numpy.float32)

if num_red_datapoints != -1:
    X = X[:num_red_datapoints]
    y = y[:num_red_datapoints]

print(dataset, model_folder, num_red_datapoints)

def create_model(trial):
    """
    Create a model using an optuna Trial object
    """
    learning_rate = trial.suggest_float('learning_rate', 1e-7, 1e-2, log=True)
    nb_layers = trial.suggest_int('nb_layers', 2, 5, log=True)

    keras.backend.clear_session()
  
    normalizer = keras.layers.Normalization(input_shape=[len(labels.features)], axis=-1)
    normalizer.adapt(X)
    
    model = keras.Sequential()
    
    model.add(normalizer)

    for n in range(nb_layers):
        num_hidden = trial.suggest_int(f'n_units_l{n}', 64, 256, log=True)
        model.add(keras.layers.Dense(num_hidden, activation='relu'))
    
    model.add(keras.layers.Dense(3))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.MeanSquaredError(), 
        metrics=['mse']
    )

    return model

def train(trial):
    """
    Objective function, either using cross validation or a single model instance.

    The k-fold cross-validation might improve the prediction in the sense that
    a "lucky" draw of the test set could not influence too much.
    """
    accuracy = []

    if NUM_FOLDS_CV == 1:
        # create the network
        model = create_model(trial)

        # training and testset
        X_train, X_test, y_train, y_test, _ = preprocess.preprocess_regression(dataset, 0.8, USE_GFLOPS_AS_TARGET)
        
        # training
        history = model.fit(X_train, y_train, epochs=NUM_EPOCHS_PER_TRIAL, batch_size=BATCH_SIZE, verbose=0)

        # compute the prediction
        y_hat = model.predict(X_test, batch_size=BATCH_SIZE)

        if USE_GFLOPS_AS_TARGET:
            best_data = numpy.argmax(y_test, axis=1)
            best_predicted = numpy.argmax(y_hat, axis=1)
        else:
            best_data = numpy.argmin(y_test, axis=1)
            best_predicted = numpy.argmin(y_hat, axis=1)

        return numpy.sum(best_predicted == best_data) / y_test.shape[0]

    else:
        kfold = KFold(n_splits=NUM_FOLDS_CV, random_state=None, shuffle=False)
        
        for train_index, test_index in kfold.split(X, y):
            # create the network
            model = create_model(trial)

            # training and testset
            X_train = X[train_index]
            y_train = y[train_index]
            X_test = X[test_index]
            y_test = y[test_index]

            # training
            history = model.fit(X_train, y_train, epochs=NUM_EPOCHS_PER_TRIAL, batch_size=BATCH_SIZE, verbose=0)

            # compute the prediction
            y_hat = model.predict(X_test, batch_size=BATCH_SIZE)

            if USE_GFLOPS_AS_TARGET:
                best_data = numpy.argmax(y_test, axis=1)
                best_predicted = numpy.argmax(y_hat, axis=1)
            else:
                best_data = numpy.argmin(y_test, axis=1)
                best_predicted = numpy.argmin(y_hat, axis=1)

            # accuracy is our optimization goal
            accuracy.append( numpy.sum(best_predicted == best_data) / y_test.shape[0] )

        print(accuracy)
        return numpy.mean(accuracy)

study = optuna.create_study(direction='maximize')
study.optimize(train, n_trials=NUM_TRIALS_OPTUNA)

best_trial = study.best_trial
model = create_model(best_trial)

print("-- Model Structure --")
model.summary()

if not os.path.exists(model_folder):
    os.makedirs(model_folder)

# Save model structure and learning rate
with open(model_folder+"/learning_rate.csv", mode='w') as Datafile:
    Exe_writer = csv.writer(Datafile, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    Exe_writer.writerow([model.optimizer.get_config()['learning_rate']])

model_config = model.get_config()
with open(model_folder+'/model_config.data', 'wb') as f:
    pickle.dump(model_config, f)
