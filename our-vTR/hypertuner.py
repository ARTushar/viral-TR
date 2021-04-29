from train import train


parameters = {
    "datasets": [
        ['dataset1', 'SRR3101734_seq.fa', 'SRR3101734_out.dat'],
        # ['dataset2', 'SRR5241432_seq.fa', 'SRR5241432_out.dat']
    ],
    "convolution_type": ["custom"],
    "kernel_size": [14],
    "kernel_count": [128, 256, 512],
    "alpha-beta": [(10, 1/10)],
    # "alpha": [10],
    # "beta": [1/10],
    "distribution": [0.3, 0.2, 0.2, 0.3],
    "linear_layer_shapes": [[32], [64, 16]],
    "l1_lambda": 1e-3,
    "l2_lambda": 0,
    "batch_size": [64],
    "epochs": [20],
    "learning_rate": 1e-3
}

for dataset in parameters["datasets"]:
    for conv_type in parameters["convolution_type"]:
        for kernel_size in parameters['kernel_size']:
            for kernel_count in parameters['kernel_count']:
                for alpha, beta in parameters['alpha-beta']:
                    for linear_layer_shapes in parameters['linear_layer_shapes']:
                        for batch_size in parameters['batch_size']:
                            for epoch in parameters['epochs']:
                                params = {
                                    "data_dir": dataset[0],
                                    "sequence_file": dataset[1],
                                    "label_file": dataset[2],
                                    "convolution_type": conv_type,
                                    "kernel_size": kernel_size,
                                    "kernel_count": kernel_count,
                                    "alpha": alpha,
                                    "beta": beta,
                                    "distribution": parameters['distribution'],
                                    "linear_layer_shapes": linear_layer_shapes,
                                    "l1_lambda": parameters['l1_lambda'],
                                    "l2_lambda": parameters['l2_lambda'],
                                    "batch_size": batch_size,
                                    "epochs": epoch,
                                    "learning_rate": parameters['learning_rate']
                                }
                                train(params)