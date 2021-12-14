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

data_gpu_mean = {}
data_gpu_std = {}

data_cpu_mean = {}
data_cpu_std = {}

data_cpu_simd_mean = {}
data_cpu_simd_std = {}

#
#   Data Collection
#

for fmt in ["csr", "dense", "auto"]:
    with open(data.dataset+'_openmp_no_simd_'+str(N_post)+'_'+str(N_pre)+'_'+fmt+'_times.csv', mode='r') as Datafile:
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

    with open(data.dataset+'_openmp_'+str(N_post)+'_'+str(N_pre)+'_'+fmt+'_times.csv', mode='r') as Datafile:
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

for fmt in ["csr", "dense", "auto"]:
    with open(data.dataset+'_cuda_'+str(N_post)+'_'+str(N_pre)+'_'+fmt+'_times.csv', mode='r') as Datafile:
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

#
#   Plotting
#
cm = 1/2.54  # centimeters to inches
plt.rcParams['font.size'] = 8
fig, [ax1, ax2] = plt.subplots(1,2,figsize=(17*cm,9*cm), dpi=300)
plt.subplots_adjust(top=0.935, bottom=0.15, left=0.09, right=0.97)

x_pos = numpy.arange(len(data.conf))
for k, v in data_cpu_mean.items():
    l = ax1.errorbar(x_pos, v, yerr=data_cpu_std[k], label=k)
    ax1.errorbar(x_pos, data_cpu_simd_mean[k], yerr=data_cpu_simd_std[k], color=l[0].get_color(), linestyle="--")

ax1.legend()
ax1.set_ylabel("computation time [s]", fontweight="bold")
ax1.set_xlabel("matrix density [%]", fontweight="bold")
ax1.set_xticks(x_pos)
ax1.set_xticklabels([str(int(c*100.0)) for c in data.conf])
ylim = ax1.get_ylim()
ax1.text(-2.3, ylim[1], "A)", fontweight="bold")
ax1.grid(True)

x_pos = numpy.arange(len(data.conf))
for k, v in data_gpu_mean.items():
    ax2.errorbar(x_pos, v, yerr=data_gpu_std[k], label=k)

plt.legend()
#ax2.set_ylabel("computation time [s]")
ax2.set_xlabel("matrix density [%]", fontweight="bold")
ax2.set_xticks(x_pos)
ax2.set_xticklabels([str(int(c*100.0)) for c in data.conf])
ylim = ax2.get_ylim()
ax2.text(-2.3, ylim[1], "B)", fontweight="bold")
ax2.grid(True)

fig.savefig("../figures/Fig3.png")
fig.savefig("../figures/Fig3.svg")
plt.show()
