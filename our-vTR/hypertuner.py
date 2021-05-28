from CV import run_cv
from train import train


parameters = {
    'datasets': [
        # ['dataset1', 'SRR3101734_seq.fa', 'SRR3101734_out.dat'],
        # ['dataset2', 'SRR5241432_seq.fa', 'SRR5241432_out.dat'],
        ['dataset3', 'SRR5241430_seq.fa', 'SRR5241430_out.dat']
    ],
    'convolution_type': 'custom',
    'kernel_size': [12],
    'kernel_count': [128],
    'alpha': 10,
    'beta': 1/10,
    'distribution': [0.3, 0.2, 0.2, 0.3],
    'pool_type': 'max',
    'linear_layer_shapes': [[32]],
    'l1_lambda': 1e-3,
    'l2_lambda': [0],
    'dropout_p': None,
    'batch_size': 64,
    'epochs': 1,
    'learning_rate': 1e-3,
    'stratify': True,
    'n_splits': 10
}

for dataset in parameters['datasets']:
    for kernel_size in parameters['kernel_size']:
        for kernel_count in parameters['kernel_count']:
            for linear_layer_shapes in parameters['linear_layer_shapes']:
                for lambda2 in parameters['l2_lambda']:
                    params = {
                        'data_dir': dataset[0],
                        'sequence_file': dataset[1],
                        'label_file': dataset[2],
                        'distribution': parameters['distribution'],
                        'batch_size': parameters['batch_size'],
                        'learning_rate': parameters['learning_rate'],
                        'alpha': parameters['alpha'],
                        'beta': parameters['beta'],
                        'convolution_type': parameters['convolution_type'],
                        'kernel_size': kernel_size,
                        'kernel_count': kernel_count,
                        'pool_type': parameters['pool_type'],
                        'linear_layer_shapes': linear_layer_shapes,
                        'l1_lambda': parameters['l1_lambda'],
                        'l2_lambda': lambda2,
                        'dropout_p': parameters['dropout_p'],
                        'epochs': parameters['epochs'],
                        # 'n_splits': parameters['n_splits'],
                        # 'stratify': parameters['stratify']
                    }
                    train(params)
                    # run_cv(params)
