for num_ds in 2500 #100 500 1000 1500 2000 2500 3000
do
    # configure network
    python optimize_network.py ../datasets/nvidia_K20m_new_ds.csv model_nvidia_K20m_red_$num_ds $num_ds
    # train best obtained configuration
    python training.py ../datasets/nvidia_K20m_new_ds.csv model_nvidia_K20m_red_$num_ds $num_ds
    # perform CV
    python cross_validation.py ../datasets/nvidia_K20m_new_ds.csv model_nvidia_K20m_red_$num_ds $num_ds
done
