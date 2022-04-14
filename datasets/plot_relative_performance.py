import csv
import sys
import numpy
import matplotlib.pylab as plt

# we evaluate both training and test set (if True) otherwise only test set.
complete_data = True
model_folder= ""
if not complete_data:
    model_folder = sys.argv[1]

# which data files should be considered
datasets_to_analyze = ["nvidia_K20m.csv", "nvidia_RTX2060.csv", "nvidia_RTX3080.csv"]
title_list=["NVIDIA K20m", "NVIDIA RTX2060", "NVIDIA RTX3080"]

# figure labels
datasets_corner_title = ["A)", "B)", "C)"]

def eval_dataset(dataset, model_folder):
    """
    Evaluate a dataset stored in *dataset* stored as csv. If only
    the testset should be considered, the keras model folder must
    be provided (*model_folder*)
    """
    trainings_indices = []
    if not complete_data:
        # readout the trainings indices
        with open(model_folder+'/indices.csv', mode='r') as Datafile:
            data = csv.reader(Datafile)

            for row in data:
                trainings_indices = numpy.array(row).astype(int)
        print(trainings_indices.shape)

    times_list = []
    with open(dataset, mode='r') as Datafile:
        csv_data = csv.reader(Datafile)

        for idx, row in enumerate(csv_data):

            # row data [ignore last 2 columns]
            times = row[7:10]
            times = numpy.array(times).astype(float)

            features = numpy.array(row[:7]).astype(float)

            # we consider only testset indices for most evaluations
            # and especially for the NN prediction
            if idx in trainings_indices and not complete_data:
                continue
            
            times = numpy.array(row[7:10]).astype(float)
            times_list.append(times)

    time_in_seconds = numpy.array(times_list)
    del times_list

    return time_in_seconds

#
# Plotting
cm = 1/2.54  # centimeters to inches
plt.rcParams['font.size'] = 8

fig, axes = plt.subplots(1, 3, figsize=(17*cm,7*cm), dpi=300, sharey=True)
plt.subplots_adjust(top=0.90, bottom=0.15, left=0.08, right=0.98, wspace=0.15)

flierprops = dict(marker='o', markerfacecolor='none', markersize=3, linewidth=0.5, markeredgecolor='k')
meanprops = dict(marker='^', markerfacecolor='green', markersize=4, markeredgecolor='none')

for idx, ax in enumerate(axes):
    time_in_seconds = eval_dataset(datasets_to_analyze[idx], model_folder)
    csr_to_ellr = time_in_seconds[:,0]/time_in_seconds[:,1]
    csr_to_dense = time_in_seconds[:,0]/time_in_seconds[:,2]

    # TODO: mean-line better?
    ax.boxplot(csr_to_ellr, positions=[0], showmeans=True, meanprops=meanprops, flierprops=flierprops)
    ax.boxplot(csr_to_dense, positions=[1], showmeans=True, meanprops=meanprops, flierprops=flierprops)

    ax.set_title(title_list[idx])
    ax.set_xticks([0,1])
    ax.set_xticklabels(["ELLPACK-R", "dense"])
    ax.set_xlabel("matrix formats", fontweight="bold")
    ax.yaxis.grid(True)
    _, ymax = plt.ylim()
    
    if idx == 0:
        ax.text(-1.05, ymax+0.23, datasets_corner_title[idx], fontweight="bold", fontsize=11)
        ax.set_ylabel("ratio to CSR (>1 faster)", fontweight="bold")
    else:
        ax.text(-0.75, ymax+0.23, datasets_corner_title[idx], fontweight="bold", fontsize=11)
    
fig.savefig("../figures/Fig5.png")
fig.savefig("../figures/Fig5.svg")
