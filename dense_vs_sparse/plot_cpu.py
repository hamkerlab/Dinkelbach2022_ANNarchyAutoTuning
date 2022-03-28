#
#   Author: Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>
#
import matplotlib.pylab as plt
import numpy
import csv
import sys
import data
from pandas import wide_to_long


# Setup configuration taken from command-line
if len(sys.argv) < 4:
    print("Expected two arguments: number of rows, number of columns, folder containing the data")
    exit(1)

N_post = int(sys.argv[1])
N_pre = int(sys.argv[2])
folder = sys.argv[3]

data_cpu_mean = {}
data_cpu_std = {}

data_cpu_simd_mean = {}
data_cpu_simd_std = {}

#
#   Data Collection
#
file_suffix = "gflops" if data.store_as_gflops else "times"

for fmt in data.fmt_list:
    with open(folder+'/'+data.dataset+'_openmp_no_simd_'+str(N_post)+'_'+str(N_pre)+'_'+fmt+'_'+file_suffix+'.csv', mode='r') as Datafile:
        csv_data = csv.reader(Datafile)

        raw_data_list = []
        for row in csv_data:
            raw_data_list.append(row)
        
        raw_data = numpy.array(raw_data_list).astype(float)
        if (raw_data.shape[0]==1):
            data_cpu_mean[fmt] = raw_data[0,:]
            data_cpu_std[fmt] = numpy.zeros(len(data.conf))
        else:
            data_cpu_mean[fmt] = numpy.mean(raw_data, axis=0)
            data_cpu_std[fmt] = numpy.std(raw_data, axis=0)

        del raw_data_list
        del raw_data

    with open(folder+'/'+data.dataset+'_openmp_'+str(N_post)+'_'+str(N_pre)+'_'+fmt+'_'+file_suffix+'.csv', mode='r') as Datafile:
        csv_data = csv.reader(Datafile)

        raw_data_list = []
        for row in csv_data:
            raw_data_list.append(row)
        
        raw_data = numpy.array(raw_data_list).astype(float)
        if (raw_data.shape[0]==1):
            data_cpu_simd_mean[fmt] = raw_data[0,:]
            data_cpu_simd_std[fmt] = numpy.zeros(len(data.conf))
        else:
            data_cpu_simd_mean[fmt] = numpy.mean(raw_data, axis=0)
            data_cpu_simd_std[fmt] = numpy.std(raw_data, axis=0)

        del raw_data_list
        del raw_data

cm = 1/2.54  # centimeters to inches
plt.rcParams['font.size'] = 8
fig, [ax1, ax2] = plt.subplots(1,2,figsize=(17*cm,7*cm), dpi=300)
plt.subplots_adjust(top=0.935, bottom=0.25, left=0.07, right=0.97, wspace=0.25)

x_pos = numpy.arange(len(data.conf))
for k, v in data_cpu_mean.items():
    l = ax1.errorbar(x_pos, v, yerr=data_cpu_std[k])
    ax1.errorbar(x_pos, data_cpu_simd_mean[k], yerr=data_cpu_simd_std[k], color=l[0].get_color(), linestyle="--")

if data.store_as_gflops:
    ax1.set_ylabel("GFLOPs", fontweight="bold")
else:
    ax1.set_ylabel("computation time [s]", fontweight="bold")
ax1.set_xlabel("matrix density [%]", fontweight="bold")
ax1.set_xticks(x_pos)
ax1.set_xticklabels([str(int(c*100.0)) for c in data.conf])
ylim = ax1.get_ylim()
ax1.text(-1.8, ylim[1], "A)", fontweight="bold", fontsize=11)
ax1.grid(True)

i = 0
for k, v in data_cpu_mean.items():
    if data.store_as_gflops:
        vec_efficiency = data_cpu_simd_mean[k] / v # > 1 SIMD faster
    else:
        vec_efficiency = v / data_cpu_simd_mean[k] # > 1 SIMD faster
    off = -0.25+i*0.25
    ax2.bar(x_pos+off, vec_efficiency, width=0.25, label=data.fmt_label[i])

    i += 1

ylim = ax2.get_ylim()
ax2.set_ylim([1, ylim[1]])
ax2.set_ylabel("SIMD improvement", fontweight="bold")
ax2.set_xlabel("matrix density [%]", fontweight="bold")
ax2.set_xticks(x_pos)
ax2.text(-2.8, ylim[1], "B)", fontweight="bold", fontsize=11)
ax2.set_xticklabels([str(int(c*100.0)) for c in data.conf])

ax2.legend(ncol=3, bbox_to_anchor=(0.51,-0.2))

fig.savefig("../figures/Fig3.png")
fig.savefig("../figures/Fig3.svg")

