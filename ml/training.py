#
#   Author: Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>, Badr-Eddine Bouhlal
#
import pandas
import numpy
import csv
import sys
import tensorflow 
import tensorflow.keras as keras
import matplotlib.pylab as plt

import preprocess
import labels

if len(sys.argv) < 2:
    print("Expected argument: dataset file (*.csv) either relative or absolute path.")

dataset = sys.argv[1]   # "../datasets/nvidia_K20m.csv", "../datasets/nvidia_RTX3080.csv"
dataset_name = sys.argv[1].split("/")[-1]
model_folder = 'model_'+dataset_name.replace(".csv","")
csv_labels = labels.get_csv_labels(dataset_name)

#
# The complete dataset is split into 80% training and 20% test data
X_train, X_test, t_train, t_test, min_values, max_values, trainings_indices = preprocess.preprocess_regression(dataset, csv_labels, 0.8)
tensorflow.keras.backend.clear_session()

model =  keras.Sequential([
    # input layer is created behind the scene
    keras.layers.Dense(64, activation=tensorflow.nn.relu, input_shape=[len(X_train.keys())]),
    keras.layers.Dense(64, activation=tensorflow.nn.relu),
    keras.layers.Dense(len(t_train.keys())) #, activation=tensorflow.nn.relu)
])

optimizer = keras.optimizers.RMSprop(0.001) # eventuell variieren

model.compile(
    loss='mse',
    optimizer=optimizer,
    metrics=['mae','mse'])

model.summary()

EPOCHS = 200

# validation_split seperates again 20% from the training set as
# validation set (should not overlap with test set)
history = model.fit(
  X_train, t_train,
  epochs=EPOCHS, validation_split=.2)
print("\nTraining done ...")

#
# Save the neural network for later use
# We need to store the values for the input normalization too
model.save(model_folder)
with open(model_folder+'/normalize.csv', mode='w') as Datafile:
    Exe_writer = csv.writer(Datafile, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    Exe_writer.writerow(min_values)
    Exe_writer.writerow(max_values)
    print(trainings_indices)
    Exe_writer.writerow(trainings_indices)
print("All network related data stored in '"+model_folder+"'")
#
# Learning evaluation
hist = pandas.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail())

def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    #plt.ylim([0, 0.08])
    plt.xlabel('Epoch')
    plt.ylabel('Error [Time]')
    plt.legend()
    plt.grid(True)

plot_loss(history)
#plt.savefig("val_loss.png")
plt.show()

loss, mae, mse = model.evaluate(X_test, t_test, verbose=0)
print("testing set Mean Squared Error : {:5.8f} Time\n".format(mse))

y = model.predict(X_test)
mse = numpy.mean((t_test - y)**2)
print("Mean Squared Error: \n", mse)
