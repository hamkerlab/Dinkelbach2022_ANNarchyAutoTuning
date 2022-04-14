#
#   Author:    Badr-Eddine Bouhlal
#
import matplotlib.pylab as plt
import numpy

ds_sizes = [100, 500, 1000, 1500, 2000, 2500, 3000]

mean_per_ds_size = numpy.zeros(len(ds_sizes))
std_per_ds_size = numpy.zeros(len(ds_sizes))

for idx, ds in enumerate(ds_sizes):
    raw = numpy.recfromtxt("cross_val_"+str(ds)+".csv")

    mean_per_ds_size[idx] = numpy.mean(raw)
    std_per_ds_size[idx] = numpy.std(raw)

plt.rcParams['font.size'] = 8      # size in 8 pt
cm = 1/2.54  # centimeters to inches
fig, ax = plt.subplots(1,1,figsize=(17*cm,9*cm), dpi=300)
plt.subplots_adjust(top=0.95, bottom=0.15, left=0.15, right=0.97) # configure distance to the outer edge

ax.errorbar(ds_sizes, mean_per_ds_size, yerr=std_per_ds_size, linestyle="", marker="." )
plt.xticks(ds_sizes, ds_sizes)
plt.xlabel('dataset size', fontweight="bold")
plt.ylabel('optimal format selection [%]', fontweight="bold")
ax.yaxis.grid(True)

print(ds_sizes)
print(mean_per_ds_size)
fig.savefig("../figures/Fig8.png")
fig.savefig("../figures/Fig8.svg")

plt.show()
