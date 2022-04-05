import numpy
import csv
import matplotlib.pylab as plt
default_colors=plt.rcParams['axes.prop_cycle'].by_key()['color']

def eval_dataset(dataset, model_folder="", complete_data=True):
    """
    Evaluate a dataset stored in *dataset* stored as csv. If only
    the testset should be considered, the keras model folder must
    be provided (*model_folder*)
    """
    trainings_indices = []
    if not complete_data:
        # readout the trainings indices
        with open(model_folder+'/normalize.csv', mode='r') as Datafile:
            data = csv.reader(Datafile)

            tmp = []
            for row in data:
                tmp.append(row)

            trainings_indices = numpy.array(tmp[2]).astype(int)

    times_list = []
    num_row_list = []
    num_col_list = []
    nnz_list = []
    density_list = []
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

            num_row_list.append(row[0])
            num_col_list.append(row[1])
            nnz_list.append(row[2])
            density_list.append(row[3])
            times = numpy.array(row[7:10]).astype(float)
            times_list.append(times)

    num_row = numpy.array(num_row_list).astype(float)
    num_col = numpy.array(num_col_list).astype(float)
    density = numpy.array(density_list).astype(float)
    nnz = numpy.array(nnz_list).astype(float)
    time_in_seconds = numpy.array(times_list).astype(float)
    nnz_per_row = nnz/num_row
    gflops = numpy.zeros(time_in_seconds.shape)
    for i in range(3):
        gflops[:,i] = ((2*nnz*1000)/time_in_seconds[:,i])/(10**9)

    del times_list

    return num_row, num_col, density, nnz_per_row, time_in_seconds, gflops

datasets = ["nvidia_K20m.csv", "nvidia_RTX2060.csv", "nvidia_RTX3080.csv"]
dataset_labels = ["K20m", "RTX2060", "RTX3080"]
cm = 1.0/2.54

f, axes = plt.subplots(len(datasets), 3, figsize=(20*cm,19*cm), dpi=300)
plt.subplots_adjust(left=0.09, bottom=0.13, right=0.98, top=0.98, hspace=0.15)

for idx_ds, dataset in enumerate(datasets):
    num_row, num_col, density, nnz_per_row, time_in_seconds, gflops = eval_dataset(dataset)

    #
    #   First decision stage >= 60% filling degree dense
    #
    axes[idx_ds, 0].scatter(density, gflops[:,0], color=default_colors[0], s=5)
    axes[idx_ds, 1].scatter(density, gflops[:,1], color=default_colors[1], s=5)
    axes[idx_ds, 2].scatter(density, gflops[:,2], color=default_colors[2], s=5)
    if (idx_ds < 2):
        axes[idx_ds, 0].set_ylabel("GFLOPs ("+dataset_labels[idx_ds]+")", fontweight="bold", labelpad=17)
    else:
        axes[idx_ds, 0].set_ylabel("GFLOPs ("+dataset_labels[idx_ds]+")", fontweight="bold", labelpad=10)

for i in range(len(datasets)):
    # row-wise y-axis share
    y_max = 0
    y_min = 0
    for j in range(3):
        ylim = axes[i,j].get_ylim()
        y_min = ylim[0] if ylim[0] < y_min else y_min
        y_max = ylim[1] if ylim[1] > y_max else y_max

    for j in range(3):
        axes[i,j].set_ylim([y_min, y_max])
        axes[i,j].set_xlim([-0.1,1.1])
        axes[i,j].yaxis.grid(True)

    # threshold for decision
    for j in range(3):
        axes[i,j].vlines(0.6, y_min, y_max, linestyle="--", color="k")
        axes[i,j].set_ylim(ylim)
        

for i in range(3):
    for j in range(3):
        axes[i,j].set_xlim([-0.1,1.1])
        axes[i,j].set_xticks([0.1*x for x in range(0,11,2)])
        
        if i == 2:
            axes[2,j].set_xticklabels([str(int(10*x)) for x in range(0,11,2)])
        else:
            axes[i,j].set_xticklabels([])

for ax in axes[2,:]:
    ax.set_xlabel("matrix density [%]", fontweight="bold")

# legend
axes[2,0].scatter(-1,-1,label="CSR", color=default_colors[0])
axes[2,0].scatter(-1,-1,label="ELLPACK-R", color=default_colors[1])
axes[2,0].scatter(-1,-1,label="Dense", color=default_colors[2])
axes[2,0].legend(ncol=3, bbox_to_anchor=(2.5,-0.3))

f.savefig("../figures/Suppl_Fig2.png")
f.savefig("../figures/Suppl_Fig2.svg")

f, axes = plt.subplots(1,3, figsize=(20*cm,9*cm), sharey=True, sharex=True, dpi=300)
plt.subplots_adjust(left=0.08, bottom=0.2, right=0.98, top=0.95, hspace=0.3)

def func(x, a, b, c):
    return a * numpy.exp(-b * x) + c

for idx_ds, dataset in enumerate(datasets):
    num_row, num_col, density, nnz_per_row, time_in_seconds, gflops = eval_dataset(dataset)

    # remove the values from the dataset
    num_row_red = numpy.delete(num_row, numpy.where(density>=0.6))
    num_col_red = numpy.delete(num_col, numpy.where(density>=0.6))
    nnz_per_row_red = numpy.delete(nnz_per_row, numpy.where(density>=0.6))
    gflops_red = numpy.delete(gflops, numpy.where(density>=0.6), axis=0)

    # compute the relative performance between ELLR and CSR
    ellr_to_csr = gflops_red[:,1] / gflops_red[:,0]

    # As we want to fit a function, we sort the arrays based on the x-Axis
    x_sort = numpy.array([x for x,y in sorted(zip(nnz_per_row_red, ellr_to_csr))])
    y_sort = numpy.array([y for x,y in sorted(zip(nnz_per_row_red, ellr_to_csr))])

    # plot the original data
    axes[idx_ds].scatter(x_sort, y_sort, s=3)

    # plot the fitted curve
    from scipy.optimize import curve_fit
    popt, pcov = curve_fit(func, x_sort, y_sort)
    axes[idx_ds].plot(x_sort, func(x_sort, *popt), color="r")

axes[0].set_ylabel("ratio ELLR to CSR", fontweight="bold")
for ax in axes:
    ax.set_xscale("log")
    ax.set_xlabel("avg. nonzero per row", fontweight="bold")
    ax.yaxis.grid(True)

ylim = ax.get_ylim()
for ax in axes:
    ax.vlines(128, ylim[0], ylim[1], color="k", linestyle = "--")
axes[0].set_ylim(ylim)

axes[0].text(1.0, ylim[1], "A)", fontweight="bold")
axes[1].text(2.5, ylim[1], "B)", fontweight="bold")
axes[2].text(2.5, ylim[1], "C)", fontweight="bold")

f.savefig("../figures/Suppl_Fig3.png")
f.savefig("../figures/Suppl_Fig3.svg")

