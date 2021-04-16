from typing import Tuple

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import torch
from torch import Tensor
import torch.nn.functional as F

mapping = {
    'A': 'T',
    'C': 'G',
    'G': 'C',
    'T': 'A'
}

def reverse_compliment(sequence: str) -> str:
    return ''.join(mapping[c] for c in sequence)


def transform_sequence(sequence: str) -> Tensor:
    sequence += 'ACGT'
    integer_encoder = LabelEncoder()

    integer_encoded = integer_encoder.fit_transform(list(sequence))
    one_hot_encoded = F.one_hot(torch.tensor(integer_encoded))
    one_hot_encoded = one_hot_encoded[:-4].type(torch.float)

    return one_hot_encoded.T

def transform_input(sequence: str) -> Tuple[Tensor, Tensor]:
    reverse = reverse_compliment(sequence)
    return transform_sequence(sequence=sequence), transform_sequence(sequence=reverse)

def transform_label(label: str) -> Tensor:
    return torch.tensor([1, 0], dtype=torch.float) if label == '0' else torch.tensor([0, 1], dtype=torch.float)


def transform_all_sequences(sequences: list) -> Tuple[Tensor, Tensor]:
    input_sequences = [transform_sequence(sequence) for sequence in sequences]
    reverse_sequences = [
        transform_sequence(reverse_compliment(sequence)) for sequence in sequences
    ]
    return torch.stack(input_sequences), torch.stack(reverse_sequences)

def transform_all_labels(labels: list) -> Tensor:
    labels = list(map(int, labels))
    tensor_labels = torch.tensor(labels).reshape(-1, 1)
    one_hot_encoded = F.one_hot(tensor_labels)

    return one_hot_encoded.squeeze().type(torch.float)



# print(transform_input('A')[0].dtype)
# print(transform_label('0').dtype)

# X1, X2 = transform_all_sequences(['ACG', 'CGT'])
# print(X1.shape)
# print(X1.dtype)
# print(X1)

# print(X2.shape)
# print(X2.dtype)
# print(X2)
# Y = transform_all_labels(['0', '1'])
# print(Y.shape)
# print(Y.dtype)
# print(Y)
