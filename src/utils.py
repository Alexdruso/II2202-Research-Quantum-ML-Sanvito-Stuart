import numpy as np


def build_precision_two_complement_fixed_point(
        num_bits: int = 32,
        point_position: float = 0.5
) -> np.array:
    positive_bits = num_bits - 1
    integer_bits = int(positive_bits * point_position)
    decimal_bits = positive_bits - integer_bits

    return np.array([-2 ** integer_bits]
                    + [2 ** -i for i in reversed(range(1, decimal_bits + 1))]
                    + [2 ** i for i in range(integer_bits)])


def build_precision_matrix(
        num_weights: np.array,
        precision: np.array
) -> np.array:
    return np.kron(np.identity(num_weights), precision.T)


def build_augmented_feature_matrix(
        feature_matrix: np.array
) -> np.array:
    bias = np.array([np.ones(feature_matrix.shape[0])]).T
    return np.concatenate([bias, feature_matrix], axis=1)


def get_weights(
        precision_matrix: np.array,
        binary_weights: np.array
) -> np.array:
    return precision_matrix.dot(binary_weights)
