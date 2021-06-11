from model import SimpleModel
from utils.motif import make_motif

version = 0
other_model = SimpleModel.load_from_checkpoint(f'saved_models/version_{version}.ckpt')
make_motif('logos/test/', other_model.get_kernels(), [.3, .2, .2, .3])