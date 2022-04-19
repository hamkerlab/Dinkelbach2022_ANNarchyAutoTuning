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
folder_no_fmath = "k20m_no_fmath"
folder_fmath = "k20m_fmath"

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
fig, axes = plt.subplots(1,3,figsize=(17*cm,7*cm), dpi=300)
plt.subplots_adjust(top=0.90, bottom=0.25, left=0.07, right=0.98, wspace=0.4)

i = 0
x_pos = numpy.arange(len(data.conf))
mean_data_no_fmath, std_data_no_fmath = load_data(folder_no_fmath)
for k, v in mean_data_no_fmath.items():
    axes[0].errorbar(x_pos, v, yerr=std_data_no_fmath[k], label=data.fmt_label[i])
    i+= 1

mean_data_fmath, std_data_fmath = load_data(folder_fmath)
for k, v in mean_data_fmath.items():
    axes[1].errorbar(x_pos, v, yerr=std_data_fmath[k])

i = 0
for k, v in mean_data_no_fmath.items():
    off = -0.3+i*0.25
    axes[2].bar(x_pos+off, v/mean_data_fmath[k], width=0.25)
    i+=1

# cut the lower side
ylim = axes[2].get_ylim()
axes[2].set_ylim([0.9, ylim[1]])

axes[0].set_ylabel("GFLOPs", fontweight="bold")
axes[1].set_ylabel("GFLOPs", fontweight="bold")
axes[2].set_ylabel("ratio", fontweight="bold")
axes[0].legend(ncol=3, bbox_to_anchor=(3.2,-0.2))

axes[0].text(-3.1, axes[0].get_ylim()[1]+1.5, "A)", fontweight="bold", fontsize=11)
axes[1].text(-3.1, axes[1].get_ylim()[1]+1.5, "B)", fontweight="bold", fontsize=11)
axes[2].text(-5.0, axes[2].get_ylim()[1]+0.01, "C)", fontweight="bold", fontsize=11)

axes[0].set_title("fast math disabled")
axes[1].set_title("fast math enabled")
axes[2].set_title("ratio disabled to enabled")

for idx, ax in enumerate(axes):
    ax.set_xlabel("matrix density [%]", fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(int(c*100.0)) for c in data.conf])
    ax.grid(True)

fig.savefig("../figures/Suppl_Fig4.png")
fig.savefig("../figures/Suppl_Fig4.svg")
