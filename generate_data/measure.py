#
#   Generate the trainings data based for a configuration
#   provided as command line arguments.
#
from ANNarchy import *
import csv
import time

simple_accumulator = Neuron(
    equations = " r = sum(exc)+1 "
)

def measure(num_rows, num_columns, pattern, pattern_arg, return_stats=True, storage_format="lil", on_cpu=True, num_threads=1, annarchy_folder="", base_seed=56789):
    clear()

    # use a fixed seed to ensure that all parallel configurations
    # use the same matrices
    if on_cpu:
        setup(paradigm="openmp", num_threads=num_threads, seed=base_seed, disable_SIMD_SpMV=False)
    else:
        setup(paradigm="cuda", num_threads=num_threads, seed=base_seed)

    # create the network
    pre = Population( num_columns, neuron=simple_accumulator )
    post = Population( num_rows, neuron=simple_accumulator )

    bsr_size = -1
    if "bsr" in storage_format:
        bsr_size = int(storage_format.split("_")[1])
        storage_format = "bsr"

    tmp = Projection( pre, post, "exc" )
    if pattern == "fp":
        tmp.connect_fixed_probability( pattern_arg, weights=Uniform(0,1), storage_format=storage_format)
    elif pattern == "fnp":
        tmp.connect_fixed_number_pre( pattern_arg, weights=Uniform(0,1), storage_format=storage_format)
    else:
        print("Invalid pattern name ...")

    if bsr_size>-1:
        tmp._bsr_size = bsr_size

    # compile
    net = Network(everything=True)
    net.compile(directory=annarchy_folder, annarchy_json="./annarchy.json")

    t1 = time.time()
    net.simulate(1000)
    t2 = time.time()

    if return_stats:

        nnz_per_row_list = []

        for d in net.get(tmp).dendrites:
            nnz_per_row_list.append(d.size)

        nnz_per_row = np.array(nnz_per_row_list)

        result = [
            # Features as suggested by Hou et al. 2017
            # features (basic information)
            net.get(tmp).post.size,
            net.get(tmp).pre.size,
            net.get(tmp).nb_synapses,
            # features (non-zero distribution information)
            net.get(tmp).nb_synapses/(net.get(tmp).post.size*net.get(tmp).pre.size),
            np.mean(nnz_per_row),
            np.min(nnz_per_row),
            np.max(nnz_per_row),
            # time
            t2-t1
        ]
    else:
        if storage_format=="auto":
            return [tmp._storage_format, t2-t1]
        else:
            return [t2-t1]

    del net
    return result

if __name__ == "__main__":
    base_seed = int(time.time())

    num_rows = int(sys.argv[1])
    num_columns = int(sys.argv[2])
    pattern = sys.argv[3]
    pattern_arg = float(sys.argv[4])
    target_folder = sys.argv[5]
    target_name = sys.argv[6]

    fmt_gpu = ["csr", "ellr", "dense", "auto"]
    results = []

    for idx, fmt in enumerate(fmt_gpu):
        print("Measure GPU, format =", fmt)
        x = True if fmt=="csr" else False
        gpu_result = measure(num_rows, num_columns, pattern, pattern_arg, x, fmt, False, 1, "annarchy_"+fmt+"_gpu", base_seed)
        results.extend(gpu_result)

    with open(target_folder+'/'+target_name, mode='a') as Datafile:
        Exe_writer = csv.writer(Datafile, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        Exe_writer.writerow(results)
