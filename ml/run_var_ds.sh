for num_ds in 100 500 1000 1500 2000 2500 3000
do
    python cross_validation.py ../datasets/nvidia_K20m.csv $num_ds
done
