#
#   Author:    Helge Uelo Dinkelbach, Badr-Eddine Bouhlal
#
import matplotlib.pylab as plt
import numpy

scores_best1 = raw = numpy.recfromtxt("cross_val_full_set.csv")

plt.rcParams['font.size'] = 8      # size in 8 pt
cm = 1/2.54  # centimeters to inches
fig, ax = plt.subplots(1,1,figsize=(17*cm,9*cm), dpi=300)
plt.subplots_adjust(top=0.95, bottom=0.15, left=0.09, right=0.97) # configure distance to the outer edge

ax.boxplot(scores_best1.T, labels=[1,2,3,4,5,6,7,8,9,10], showmeans=True)

ax.set_xlabel('cross validation repetitions', fontweight="bold")
ax.set_ylabel('optimal format selection [%]', fontweight="bold")
ax.yaxis.grid(True)

fig.savefig("../figures/Fig7.png")
fig.savefig("../figures/Fig7.svg")
plt.show()
