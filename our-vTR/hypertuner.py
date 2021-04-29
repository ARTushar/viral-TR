from train import train


parameters = {
    "datasets": [
        ['dataset1', 'SRR3101734_seq.fa', 'SRR3101734_out.data'],
        ['dataset2', 'SRR5241432_seq.fa', 'SRR5241432_out.dat']
    ],
    "convolution_type": "custom",
    "kernel_size": [8, 12, 16],
    "kernel_count": [16, 32, 64, 128, 256, 512],
    "alpha": [10, 100, 1000],
    "beta": [10, 100, 1000],
    "distribution": [0.3, 0.2, 0.2, 0.3],
    "linear_layer_shapes": [[8], [16], [32]],
    "l1_lambda": 1e-3,
    "l2_lambda": 0,
    "batch_size": [64, 128, 256, 512],
    "epochs": [20, 30, 40, 50],
    "learning_rate": 1e-3
}

for dataset in parameters["datasets"]:
    for conv_type in parameters["convulution_type"]:
        for kernel_size in parameters['kernel_size']:
            for kernel_count in parameters['kernel_count']:
                for alpha in parameters['alpha']:
                    for beta in parameters['beta']:
                        for linear_layer_shapes in parameters['linear_layer_shapes']:
                            for batch_size in parameters['batch_size']:
                                for epoch in parameters['epochs']:
                                    params = {
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