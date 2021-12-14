#
#   Author: Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com
#
from ANNarchy import *
import csv

# part of the setup which is not taken from command-line
import data

# Setup configuration taken from command-line
N_post = int(sys.argv[1])
N_pre = int(sys.argv[2])
fmt = sys.argv[3]
paradigm = sys.argv[4]
no_simd = bool(int(sys.argv[5]))

# a simple accumulator
simple_neuron = Neuron(
    equations="r = sum(exc)+1"
)

# results, one value per configuration
result_times = np.zeros(len(data.conf))

for idx_c, c in enumerate(data.conf):
    # clear also resets global configurations
    clear()

    # some global optimization flags
    setup(paradigm=paradigm, disable_SIMD_SpMV=no_simd)

    # Network: 2 populations, 1 projection (FP-pattern)
    in_pop = Population(N_pre, simple_neuron)
    out_pop = Population(N_post, simple_neuron)
    proj = Projection(in_pop, out_pop, "exc")
    proj.connect_fixed_probability(c, Uniform(0,1), storage_format=fmt)

    # compile and initialize
    compile()

    # perform the simulation
    t1 = time.time()
    simulate(1000)
    t2 = time.time()

    # just to check progress on the command line
    print("Configuration p =", c, "done")
    result_times[idx_c] = t2-t1

# setup the configurations and store the results afterwards in the file
simd_str = "_no_simd" if no_simd else ""
with open(data.dataset+'_'+paradigm+simd_str+'_'+str(N_post)+'_'+str(N_pre)+'_'+fmt+'_times.csv', mode='a') as Datafile:
    Exe_writer = csv.writer(Datafile, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    Exe_writer.writerow(result_times)
