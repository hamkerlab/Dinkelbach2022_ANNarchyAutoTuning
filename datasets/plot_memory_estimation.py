import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
default_colors=plt.rcParams['axes.prop_cycle'].by_key()['color']

csv_labels = [
     # used as features
     "number rows", "number columns", "overall number nonzeros", "density",
     "mean nnz per row", "min nnz per row", "max nnz per row",
     # output (time)
     "csr (gpu)", "ellr (gpu)", "dense (gpu)",
     # heuristic
     "auto (label)", "auto (times)"
]

data = pd.read_csv('nvidia_K20m.csv', names=csv_labels, header=None, index_col=False)

M = data['number rows'].to_numpy().astype(int)
N = data['number columns'].to_numpy().astype(int)
nnz = data['overall number nonzeros'].to_numpy().astype(int)
max_nnz = data['max nnz per row'].to_numpy().astype(int)
density = data['density'].to_numpy().astype(float)

cm = 1/2.54
plt.rcParams['font.size'] = 8
f, axes = plt.subplots(1,3,figsize=(17*cm,7*cm), dpi=300, sharey=True)
plt.subplots_adjust(top=0.93, bottom=0.15, left=0.07, right=0.98, wspace=0.15)    

# Double precision
idx_size = 4
val_size = 8


memory_csr = M * idx_size     # row_ptr
memory_csr += nnz * idx_size  # col_idx
memory_csr += nnz * val_size  # values
bytes_per_nnz_csr = memory_csr / nnz
axes[0].scatter(density, bytes_per_nnz_csr, color=default_colors[0], s=4)

memory_ellr = M * max_nnz * idx_size   # col_idx
memory_ellr += M * max_nnz * val_size  # values
memory_ellr += M * idx_size            # rl
bytes_per_nnz_ellr = memory_ellr / nnz
axes[1].scatter(density, bytes_per_nnz_ellr, color=default_colors[1], s=4)

memory_dense = M * N * val_size    # values
bytes_per_nnz_dense = memory_dense / nnz
axes[2].scatter(density, bytes_per_nnz_dense, color=default_colors[2], s=4)

plt.yscale("log")

axes[0].set_ylabel("bytes per nonzero", fontweight="bold")
for ax in axes:
     ax.set_xlabel("matrix density [%]", fontweight="bold")
     ax.set_xticks([0.1*x for x in range(0,11,2)])
     ax.set_xticklabels([str(int(10*x)) for x in range(0,11,2)])
     ax.yaxis.grid(True)

f.savefig("../figures/Suppl_Fig1.png")
f.savefig("../figures/Suppl_Fig1.svg")
