#
#   Generate a list of configurations for the trainings data.
#
import numpy.random as random
import sys
import csv

num_conf = int(sys.argv[1])

# available neuron sizes
N = [1000,2000,4000,8000,16000, 20000]
# probabilities for fixed probability pattern
FP_conf = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 0.75, 1.0]
# number of pre-neurons for fixed number pre pattern
FNP_conf = [32,64,128,256,512,1024,2048,4096]

print("Generate", num_conf, "configurations.")
with open('configurations.csv', mode='w') as Datafile:
    Exe_writer = csv.writer(Datafile, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)

    for i in range(num_conf):

        m_idx = random.randint(0, len(N)-1)
        n_idx = random.randint(0, len(N)-1)
        neur_config = (N[m_idx], N[n_idx])

        #
        # Generate the configuration. We limit the number of nonzero
        # per row to a realistic range.
        p = random.random()
        if p < 0.5:
            param = FP_conf[random.randint(0, len(FP_conf)-1)]
            while( (param * neur_config[1]) > 6000):
                # the configuration would be invalid, so we redraw
                param = FP_conf[random.randint(0, len(FP_conf)-1)]

            conn_config = ('fp', param)
        else:
            param = FNP_conf[random.randint(0, len(FNP_conf)-1)]
            while(param > neur_config[1]):
                # the configuration would be invalid, so we redraw
                param = FNP_conf[random.randint(0, len(FNP_conf)-1)]

            conn_config = ('fnp', param,)

        Exe_writer.writerow(neur_config+conn_config)
