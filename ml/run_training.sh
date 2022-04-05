# determine an optimal network configuration
python optimize_network.py ../datasets/nvidia_K20m.csv model_nvidia_K20m
python optimize_network.py ../datasets/nvidia_RTX2060.csv model_nvidia_RTX2060
python optimize_network.py ../datasets/nvidia_RTX3080.csv model_nvidia_RTX3080

# train a model with the determined configuration
python training.py ../datasets/nvidia_K20m.csv model_nvidia_K20m
python training.py ../datasets/nvidia_RTX2060.csv model_nvidia_RTX2060
python training.py ../datasets/nvidia_RTX3080.csv model_nvidia_RTX3080
