#
#   Author:    Badr-Eddine Bouhlal
#
import tensorflow 
import tensorflow.keras as keras
import numpy
import csv
import sys
import pandas
import matplotlib.pylab as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

import preprocess
import labels

#
# Experiment setup
k = 5                    # number of folds, i. e. how many partitions
repeats = range(1,11)    # number of repetition of the k-fold algorithm
if len(sys.argv) < 2:
    print("Expected argument: \n")
    print(" - dataset file (*.csv) either relative or absolute path.")
    print(" - number of data points for evaluation (optional, if not present this means all)")

# get the data
dataset = sys.argv[1]      # "../datasets/nvidia_K20m.csv", "../datasets/nvidia_RTX3080.csv"
dataset_name = sys.argv[1].split("/")[-1]
csv_labels = labels.get_csv_labels(dataset_name)
csv_labels_output = labels.get_csv_labels_output(dataset_name)

# preprocess the data
X_train, X_test, t_train, t_test, min_values, max_values, trainings_indices = preprocess.preprocess_regression(dataset, csv_labels, 0.8)
tensorflow.keras.backend.clear_session()

# The complete dataset is split into 80% training and 20% test data we need to merge the
# whole Dataset, because the split will be automaticaly performed during the cross validation
inputs = numpy.concatenate((X_train, X_test), axis=0)
targets = numpy.concatenate(( t_train, t_test), axis=0)

if len(sys.argv) == 3:
     red_data_points = int(sys.argv[2])
     print("Use a reduced set of", red_data_points, "data points")
     inputs = inputs[:red_data_points]
     targets = targets[:red_data_points]
else:
     print("Use all available ", inputs.shape(0), "data points")

scores_best1 =[]

#number of repeats of the cross validation 
for r in repeats:
     print("Start Repeat", r)
     
     crossValidation_pred_Best1 = []
     kfold = KFold(n_splits=k, random_state=None, shuffle=False)
     fold = 0
     print(f"############ Cross-validation using {k} folds")
     for train_index, test_index in kfold.split(inputs,targets): 

          best1 = 0   
               
          fold+=1
          
          Sum_best1 = 0

          #split the data within the crosss validation --> each round with 80% for the training and 20% for the testing 
          x_train = inputs[train_index]
          y_train = targets[train_index]
          x_test = inputs[test_index]
          y_test = targets[test_index]
          #performing the training and the validation with the 80% of the dataset
          model =  keras.Sequential([
               # input layer is created behind the scene
               keras.layers.Dense(64, activation=tensorflow.nn.relu, input_shape=[len(X_train.keys())]),
               keras.layers.Dense(64, activation=tensorflow.nn.relu),
               keras.layers.Dense(len(t_train.keys()))
          ])
          optimizer = keras.optimizers.RMSprop(0.001)
          model.compile(
                         loss='mse',
                         optimizer=optimizer,
                         metrics=['mae','mse'])
          history = model.fit(x_train,y_train,validation_split=.2,
          verbose=0,epochs=200)
          #number of epochs choosed fixesd at 200 --> choosed based on the plot of the validation

          print("\nTraining done ...")



          hist = pandas.DataFrame(history.history)
          hist['epoch'] = history.epoch
          print(hist.tail())
          #print(x_test)
          #print(y_test)

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

# Figure 6 of the manuscript
if red_data_points == 0:
     plt.rcParams['font.size'] = 8      # size in 8 pt
     cm = 1/2.54  # centimeters to inches
     fig, ax = plt.subplots(1,1,figsize=(17*cm,9*cm), dpi=300)
     plt.subplots_adjust(top=0.95, bottom=0.15, left=0.15, right=0.97) # configure distance to the outer edge

     ax.boxplot(scores_best1, labels=[1,2,3,4,5,6,7,8,9,10], showmeans=True)

     ax.set_xlabel('cross validation repetitions', fontweight="bold")
     ax.set_ylabel('optimal format selection [%]', fontweight="bold")

     fig.savefig("../figures/Fig6.png")
     fig.savefig("../figures/Fig6.svg")
     plt.show()
else:
     # data for Figure 7
     numpy.savetxt("cross_val_"+str(red_data_points)+".csv", scores_best1)
