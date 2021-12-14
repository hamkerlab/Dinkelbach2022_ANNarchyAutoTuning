#
#   Author: Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>, Badr-Eddine Bouhlal
#
import tensorflow 
import tensorflow.keras as keras
import numpy
import csv
import pandas
import matplotlib.pylab as plt
cm = 1/2.54  # centimeters to inches
plt.rcParams['font.size'] = 8

import labels

def transform_input(X, min_values, max_values):
    
    for i in range(len(X)):
        X[i] = (X[i] - min_values[i]) / (max_values[i]-min_values[i])

    return X

def analysis(dataset, model_folder, use_gflops_as_target):
    """
    """
    tensorflow.keras.backend.clear_session()
    model = keras.models.load_model(model_folder)
    min_values=[]
    max_values=[]
    trainings_indices=[]

    with open(model_folder+'/normalize.csv', mode='r') as Datafile:
        data = csv.reader(Datafile)

        tmp = []
        for row in data:
            tmp.append(row)
        
        min_values = numpy.array(tmp[0]).astype(float)
        max_values = numpy.array(tmp[1]).astype(float)
        trainings_indices = numpy.array(tmp[2]).astype(int)

    model_data = []
    best_idx_data = []
    best_idx_train_data = []
    best_data = []
    auto_format = []
    raw_idx = []

    with open(dataset, mode='r') as Datafile:
        csv_data = csv.reader(Datafile)

        for idx, row in enumerate(csv_data):
            # row data [ignore last 2 columns]
            times = row[7:10]
            times = numpy.array(times).astype(float)

            features = numpy.array(row[:7]).astype(float)

            # we consider only testset indices for most evaluations
            # and especially for the NN prediction
            if idx in trainings_indices:
                best_idx_train_data.append(numpy.argmin(times))
            else:
                norm_features = transform_input(features, min_values, max_values)
                model_data.append(list(norm_features))

                raw_idx.append(idx)
                best_idx_data.append(numpy.argmin(times))
                best_data = times[numpy.argmin(times)]

                # position of the "auto (label)" field
                sel_format = row[10]

                # TODO: improve!!!!
                if sel_format == "csr":
                    auto_format.append(0)
                elif sel_format == "ellr":
                    auto_format.append(1)
                elif sel_format == "dense":
                    auto_format.append(2)
                else:
                    auto_format.append(-1)

    # Make the model prediction
    y = model.predict(model_data)
    # Attention!!! need to be adjusted dependent on your training !!!
    if use_gflops_as_target:
        best_idx_pred = numpy.argmax(y, axis=1) # GFLOPs as output
    else:
        best_idx_pred = numpy.argmin(y, axis=1) # computation time as output

    auto_format = numpy.array(auto_format)

    return best_idx_pred, best_idx_data, auto_format

fig, axes = plt.subplots(1,2,figsize=(17*cm,8*cm), dpi=300)
plt.subplots_adjust(top=0.93, bottom=0.20, left=0.10, right=0.95)


i = 0
for dataset_name in ["nvidia_K20m.csv", "nvidia_RTX3080.csv"]:
    dataset = "../datasets/"+dataset_name
    csv_labels = labels.get_csv_labels(dataset_name)
    print(dataset, csv_labels)

    # Load testset, perform neural network prediction
    csv_labels = ["Compressed Sparse Row", "ELLPACK-R", "Dense"]
    csv_labels_output = labels
    model_folder = 'model_'+dataset_name.replace(".csv","")

    best_idx_pred, best_idx_data, auto_format = analysis(dataset = dataset, model_folder=model_folder, use_gflops_as_target=True)

    # Analysis
    csr_is_optimal = numpy.array(numpy.where(numpy.array(best_idx_data)==0))
    print("CSR optimal:", csr_is_optimal.shape[1], len(best_idx_data), (csr_is_optimal.shape[1]/len(best_idx_data))*100.0)
    print("NeuralNet:", sum(best_idx_data==best_idx_pred), len(best_idx_data), (sum(best_idx_data==best_idx_pred)/len(best_idx_data))*100.0 )
    print("Heuristic:", sum(best_idx_data==auto_format), len(best_idx_data), (sum(best_idx_data==auto_format)/len(best_idx_data))*100.0 )
    #for i in range(len(best_idx_data)):
    #    print(raw_idx[i], best_idx_data[i], best_idx_pred[i], auto_format[i])


    sizes = numpy.zeros((3,3))

    # original data
    csr_is_optimal = numpy.array(numpy.where(numpy.array(best_idx_data)==0))
    ellr_is_optimal = numpy.array(numpy.where(numpy.array(best_idx_data)==1))
    dense_is_optimal = numpy.array(numpy.where(numpy.array(best_idx_data)==2))

    sizes[:,0] = [csr_is_optimal.shape[1], ellr_is_optimal.shape[1], dense_is_optimal.shape[1]]

    # heuristic
    csr_is_optimal = numpy.array(numpy.where(numpy.array(auto_format)==0))
    ellr_is_optimal = numpy.array(numpy.where(numpy.array(auto_format)==1))
    dense_is_optimal = numpy.array(numpy.where(numpy.array(auto_format)==2))

    sizes[:,1] = [csr_is_optimal.shape[1], ellr_is_optimal.shape[1], dense_is_optimal.shape[1]]

    # machine learning
    csr_is_optimal = numpy.array(numpy.where(numpy.array(best_idx_pred)==0))
    ellr_is_optimal = numpy.array(numpy.where(numpy.array(best_idx_pred)==1))
    dense_is_optimal = numpy.array(numpy.where(numpy.array(best_idx_pred)==2))

    sizes[:,2] = [csr_is_optimal.shape[1], ellr_is_optimal.shape[1], dense_is_optimal.shape[1]]

    #ax1.set_ylim([0,len(best_idx_data)])
    #ax1.set_ylabel("number of times being selected as optimal")

    sizes /= len(best_idx_data)
    sizes *= 100.0
    axes[i].set_ylim([0,100])

    axes[i].bar([0,1,2], sizes[0,:])
    axes[i].bar([0,1,2], sizes[1,:], bottom=sizes[0,:])
    axes[i].bar([0,1,2], sizes[2,:], bottom=sizes[0,:]+sizes[1,:])
    axes[i].set_xticks([0,1,2])
    axes[i].set_xticklabels(["test set", "heuristic", "machine learning"], fontweight="bold")

    if i==0:
        axes[i].set_ylabel("data format is optimal [%]", fontweight="bold")
        axes[i].legend(csv_labels, ncol=3, bbox_to_anchor=(1.8,-0.1))

    if i==0:
        axes[i].text(-1.1, 105, "A)", fontweight="bold")
    else:
        axes[i].text(-1.1, 105, "B)", fontweight="bold")

    i+=1

fig.savefig("../figures/Fig5.png")
fig.savefig("../figures/Fig5.svg")
plt.show()
