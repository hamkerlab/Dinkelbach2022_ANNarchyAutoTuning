#
#   Author: Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>
#
import matplotlib.pylab as plt
import numpy
import csv
import sys
import data


# Setup configuration taken from command-line
if len(sys.argv) < 3:
    print("Expected two arguments: number of rows, number of columns")
    exit(1)

N_post = int(sys.argv[1])
N_pre = int(sys.argv[2])
folder_list = ["k20m", "rtx2060", "rtx3080"]
folder_labels = ["NVIDIA K20m", "NVIDIA RTX2060", "NVIDIA RTX3080"]
corner_labels = ["A)", "B)", "C)"]

def load_data(folder):
    data_gpu_mean = {}
    data_gpu_std = {}

    for fmt in data.fmt_list:
        with open(folder+'/'+data.dataset+'_cuda_'+str(N_post)+'_'+str(N_pre)+'_'+fmt+'_gflops.csv', mode='r') as Datafile:
            csv_data = csv.reader(Datafile)

            raw_data_list = []
            for row in csv_data:
                raw_data_list.append(row)
            
            raw_data = numpy.array(raw_data_list).astype(float)
            if (raw_data.shape[0]==1):
                data_gpu_mean[fmt] = raw_data[0,:]
                data_gpu_std[fmt] = numpy.zeros(len(data.conf))
            else:
                data_gpu_mean[fmt] = numpy.mean(raw_data, axis=0)
                data_gpu_std[fmt] = numpy.std(raw_data, axis=0)

            del raw_data_list
            del raw_data
    
    return data_gpu_mean, data_gpu_std

cm = 1/2.54  # centimeters to inches
plt.rcParams['font.size'] = 8
fig, axes = plt.subplots(1,len(folder_list),figsize=(17*cm,7*cm), dpi=300)
plt.subplots_adjust(top=0.93, bottom=0.25, left=0.07, right=0.98, wspace=0.4)

x_pos = numpy.arange(len(data.conf))
for idx_f, folder in enumerate(folder_list):
    mean_data, std_data = load_data(folder)

    idx_ds = 0
    for k, v in mean_data.items():
        axes[idx_f].errorbar(x_pos, v, yerr=std_data[k], label=data.fmt_label[idx_ds])
        idx_ds += 1

for idx_f, label in enumerate(folder_labels):
    axes[idx_f].set_ylabel("GFLOPs ("+label+")", fontweight="bold")

axes[0].legend(ncol=3, bbox_to_anchor=(3.2,-0.2))

axes[0].text(-3.0, axes[0].get_ylim()[1], "A)", fontweight="bold", fontsize=11)
axes[1].text(-3.1, axes[1].get_ylim()[1], "B)", fontweight="bold", fontsize=11)
axes[2].text(-3.5, axes[2].get_ylim()[1], "C)", fontweight="bold", fontsize=11)

for ax in axes:
    ax.set_xlabel("matrix density [%]", fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(int(c*100.0)) for c in data.conf])
    ax.grid(True)

fig.savefig("../figures/Fig4.png")
fig.savefig("../figures/Fig4.svg")
