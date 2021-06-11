import os
from model import SimpleModel
from utils.motif import make_motif

version = 124
other_model = SimpleModel.load_from_checkpoint(f'saved_models/version_{version}.ckpt')

test_dir = 'logos/test'
if not os.path.isdir(test_dir):
	os.makedirs(test_dir)
make_motif(test_dir, other_model.get_probabilities(), [.3, .2, .2, .3])