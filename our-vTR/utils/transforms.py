from sklearn.preprocessing import LabelEncoder
from torch.functional import Tensor
import torch

mapping = {
    'A': 'T',
    'C': 'G',
    'G': 'C',
    'T': 'A'
}

def transform_sequence(sequence: str) -> Tensor:
    sequence += 'ACGT'
    integer_encoder = LabelEncoder()

    integer_encoded = integer_encoder.fit_transform(list(sequence))
    one_hot_encoded = torch.nn.functional.one_hot(torch.tensor(integer_encoded))
    one_hot_encoded = one_hot_encoded[:-4].type(torch.float)

    return one_hot_encoded

def transform_input(sequence: str) -> Tensor:
    reverse = ''.join(mapping[c] for c in sequence)
    return transform_sequence(sequence=sequence), transform_sequence(sequence=reverse)

def transform_label(label: str) -> Tensor:
    return torch.tensor([1, 0], dtype=torch.float) if label == '0' else torch.tensor([0, 1], dtype=torch.float)


# print(transform_input('A')[0].dtype)
# print(transform_label('0').dtype)