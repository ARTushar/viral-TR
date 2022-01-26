from pprint import pprint


kernel_size = 14
kernel_count = 1
version = 6


def read_kernels(input_file):
    kernels = []
    with open(input_file) as f:
        for _ in range(8):
            f.readline()

        for i in range(kernel_count):
            for _ in range(3):
                f.readline()
            kernels.append([list(map(float, f.readline().split())) for _ in range(kernel_size)])
    
    return kernels



kernels = read_kernels(f'../globals/logos/{version}/motif.meme')
pprint(kernels)