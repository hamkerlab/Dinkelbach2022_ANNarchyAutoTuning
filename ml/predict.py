import tensorflow
import csv
import numpy
import pandas
import preprocess
import labels
import pickle
import matplotlib.pylab as plt

def analyse_regression(model_folder, dataset_file):
    """
    Load a model and the test set indices stored in *model_folder*.
    The corresponding dataset is stored in *dataset_file*
    """
    tensorflow.keras.backend.clear_session()

    model = tensorflow.keras.models.load_model(model_folder)
    model.summary()

    # training set indices used in training
    with open(model_folder+"/indices.csv", mode='r') as Datafile:
        csv_data = csv.reader(Datafile)
        
        indices = []
        # only one line
        for d in csv_data:
            indices = numpy.array(d).astype(int)

    X_test, y_test, auto_test = preprocess.get_evaluation_data(dataset_file, indices)

    y_pred = model.predict(X_test)

    return y_pred, y_test, auto_test

def analysis(model_folder, dataset_file):
    """
    Compare the NN and the auto-selection.    
    """
    y_pred, y_test, auto_test = analyse_regression(model_folder, dataset_file)

    # GFLOPs is max optimal
    best_data = numpy.argmax(y_test, axis=1)
    best_predicted = numpy.argmax(y_pred, axis=1)

    #
    # Accuracy
    print("Achieved results on", y_test.shape[0], "test set elements")
    csr_is_optimal = numpy.sum((best_data)==0) / y_test.shape[0]
    print("  CSR optimal:", round(csr_is_optimal*100.0, 2))
    accuracy_heuristic = numpy.sum(auto_test == best_data) / y_test.shape[0]
    print("  Accuracy (heuristic):", round(accuracy_heuristic*100.0,2))
    accuracy_neural_net = numpy.sum(best_predicted == best_data) / y_test.shape[0]
    print("  Accuracy (neural network):", round(accuracy_neural_net*100.0,2))

    #
    # Matrix Formats
    sizes = numpy.zeros((3,3))
    
    # original data
    csr_is_optimal = numpy.array(numpy.where(numpy.array(best_data)==0))
    ellr_is_optimal = numpy.array(numpy.where(numpy.array(best_data)==1))
    dense_is_optimal = numpy.array(numpy.where(numpy.array(best_data)==2))
    sizes[:,0] = [csr_is_optimal.shape[1], ellr_is_optimal.shape[1], dense_is_optimal.shape[1]]

    # heuristic
    csr_is_optimal = numpy.array(numpy.where(numpy.array(auto_test)==0))
    ellr_is_optimal = numpy.array(numpy.where(numpy.array(auto_test)==1))
    dense_is_optimal = numpy.array(numpy.where(numpy.array(auto_test)==2))
    sizes[:,1] = [csr_is_optimal.shape[1], ellr_is_optimal.shape[1], dense_is_optimal.shape[1]]

    # machine learning (regression)
    csr_is_optimal = numpy.array(numpy.where(numpy.array(best_predicted)==0))
    ellr_is_optimal = numpy.array(numpy.where(numpy.array(best_predicted)==1))
    dense_is_optimal = numpy.array(numpy.where(numpy.array(best_predicted)==2))
    sizes[:,2] = [csr_is_optimal.shape[1], ellr_is_optimal.shape[1], dense_is_optimal.shape[1]]

    return sizes, len(best_data)

#
# Plotting
#
title_list=["NVIDIA K20m", "NVIDIA RTX2060", "NVIDIA RTX3080"]
model_list=["model_nvidia_K20m", "model_nvidia_RTX2060", "model_nvidia_RTX3080"]
dataset_list=["../datasets/nvidia_K20m.csv", "../datasets/nvidia_RTX2060.csv", "../datasets/nvidia_RTX3080.csv"]

cm = 1/2.54  # centimeters to inches
plt.rcParams["font.size"] = 8
fig, axes = plt.subplots(1,3,figsize=(17*cm, 7*cm), dpi=300)
plt.subplots_adjust(top=0.90, bottom=0.25, left=0.085, right=0.98,wspace=0.3)

i = 0
for model_folder, dataset_file in zip(model_list, dataset_list):
    format_partition, num_points = analysis(model_folder, dataset_file)

    format_partition /= num_points
    format_partition *= 100.0
    axes[i].set_ylim([0,100])

    axes[i].bar([0,1.5,3], format_partition[0,:])
    axes[i].bar([0,1.5,3], format_partition[1,:], bottom=format_partition[0,:])
    axes[i].bar([0,1.5,3], format_partition[2,:], bottom=format_partition[0,:]+format_partition[1,:])
    axes[i].set_xticks([0,1.5,3])
    axes[i].set_xticklabels(["test set", "heuristic", "machine\nlearning"], fontweight="bold")
    axes[i].set_title(title_list[i])
    
    i += 1

axes[0].set_ylabel("data format is optimal [%]", fontweight="bold")
axes[0].legend(["Compressed Sparse Row", "ELLPACK-R", "Dense"], ncol=3, bbox_to_anchor=(2.7,-0.18))

axes[0].text(-1.9, 108, "A)", fontweight="bold", fontsize=11)
axes[1].text(-1.5, 108, "B)", fontweight="bold", fontsize=11)
axes[2].text(-1.5, 108, "C)", fontweight="bold", fontsize=11)

fig.savefig("../figures/Fig6.png")
fig.savefig("../figures/Fig6.svg")
