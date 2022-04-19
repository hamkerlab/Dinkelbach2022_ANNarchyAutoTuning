#
#   Author:    Badr-Eddine Bouhlal, Helge Ülo Dinkelbach
#
import tensorflow
import numpy
import csv
import sys
import pickle
import pandas
import matplotlib.pylab as plt
from sklearn.model_selection import KFold

import preprocess
import labels

#
# Experiment setup
k = 5                    # number of folds, i. e. how many partitions
repeats = range(1,11)    # number of repetition of the k-fold algorithm
if len(sys.argv) < 3:
    print("Expected argument: \n")
    print(" - dataset file (*.csv) either relative or absolute path.")
    print(" - folder containing the architecture definition.")
    print(" - optional: number of data points for evaluation")
    exit()

# get the data
dataset_file = sys.argv[1]
model_folder = sys.argv[2]

# preprocess the data
data = preprocess.preprocess_complete_data(dataset_file, True)

# The complete dataset is split into 80% training and 20% test data we need to merge the
# whole Dataset, because the split will be automaticaly performed during the cross validation
inputs = data[labels.features].to_numpy().astype(numpy.float32)
targets = data[labels.outputs].to_numpy().astype(numpy.float32)

if len(sys.argv) == 4:
     red_data_points = int(sys.argv[3])
     print("Use a reduced set of", red_data_points, "data points")
     inputs = inputs[:red_data_points]
     targets = targets[:red_data_points]
else:
     red_data_points = 0
     print("Use all available ", inputs.shape[0], "data points")

scores_best1 =[]

#number of repeats of the cross validation 
for r in repeats:
     print("Start Repeat", r)
     
     crossValidation_pred_Best1 = []
     kfold = KFold(n_splits=k, random_state=None, shuffle=False)
     fold = 0
     print(f"############ Cross-validation using {k} folds")
     for train_index, test_index in kfold.split(inputs, targets): 

          best1 = 0   
               
          fold+=1
          
          Sum_best1 = 0

          #split the data within the crosss validation --> each round with 80% for the training and 20% for the testing 
          x_train = inputs[train_index]
          y_train = targets[train_index]
          x_test = inputs[test_index]
          y_test = targets[test_index]

          tensorflow.keras.backend.clear_session()

          # load the model configuration determined by optuna
          with open(model_folder+'/model_config.data', 'rb') as f:
               model_dict = pickle.load(f)
          learning_rate = numpy.recfromtxt(model_folder+'/learning_rate.csv')

          # create the model          
          model = tensorflow.keras.Sequential.from_config(model_dict)

          # TODO: is this correct ???
          norm_layer = model.get_layer("normalization")
          norm_layer.adapt(inputs)

          # compile the model
          model.compile(
               optimizer=tensorflow.keras.optimizers.Adam(learning_rate=learning_rate),
               loss=tensorflow.keras.losses.MeanSquaredError(), 
               metrics=['mse']
          )

          #performing the training and the validation with the 80% of the dataset
          history = model.fit(x_train, y_train, verbose=0, epochs=200, batch_size=256)

          print("\nTraining done ...")

          # Learning evaluation --> performing test accuracy with the remaining 20%
          # x_test hold the dataset features,  y_test is holding the real value (labels) corresponding to x_test  and y is hoding the prediction
          y = model.predict(x_test)

          #In this step we compare the predicted values holding in  y to the real values of y_test
          #we compare only the one best values  
          best_idx_pred = numpy.argmax(y, axis=1) # GFLOPs as output
          best_idx_test = numpy.argmax(y_test, axis=1) # GFLOPs as output

          print("NeuralNet:", sum(best_idx_test==best_idx_pred), len(best_idx_test))

          Accuracy = (sum(best_idx_test==best_idx_pred)  * 100) /len(best_idx_test)
          print(Accuracy)
          crossValidation_pred_Best1 = numpy.append(crossValidation_pred_Best1, Accuracy)
          
     Sum_best1 = sum(crossValidation_pred_Best1)
     print(crossValidation_pred_Best1)
     scores_best1.append(crossValidation_pred_Best1)

if red_data_points == 0:
     # data for Figure 7 of the manuscript
     numpy.savetxt("cross_val_full_set.csv", scores_best1)

else:
     # data for Figure 8 of the manuscript
     numpy.savetxt("cross_val_"+str(red_data_points)+".csv", scores_best1)
