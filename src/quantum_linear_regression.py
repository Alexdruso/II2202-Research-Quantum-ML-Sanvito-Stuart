from sklearn.linear_model import LinearRegression
import dimod
from dwave.system.samplers import LeapHybridSampler, DWaveSampler

from utils import *


def build_q_linear_regression(
        precision_matrix: np.array,
        feature_matrix: np.array,
        target_matrix: np.array,
        fit_intercept: bool = False
) -> np.array:
    augmented_feature_matrix = build_augmented_feature_matrix(feature_matrix) if fit_intercept else feature_matrix
    bias = -2 * precision_matrix.T.dot(augmented_feature_matrix.T).dot(target_matrix)
    coupler = precision_matrix.T.dot(augmented_feature_matrix.T).dot(augmented_feature_matrix).dot(precision_matrix)

    return coupler + np.diag(bias)


samplers = {
    'quantum': DWaveSampler(),
    'hybrid': LeapHybridSampler(),
    'simulated': dimod.SimulatedAnnealingSampler()
}


class QuantumLinearRegression(LinearRegression):

    def __init__(
            self,
            fit_intercept: bool = True,
            positive: bool = True,
            num_bits: int = 8,
            point_position: float = 0.5,
            sampler: str = 'quantum'
    ):
        self.positive = positive
        self.num_bits = num_bits
        self.point_position = point_position
        self.sampler = sampler
        super().__init__(fit_intercept=fit_intercept, normalize=False, copy_X=True, n_jobs=None)

    def fit(self, X, y, sample_weight=None):
        X, y = self._validate_data(
            X, y, y_numeric=True, multi_output=True
        )

        precision_matrix = build_precision_matrix(
            num_weights=X.shape[1] + 1 if self.fit_intercept else X.shape[1],
            precision=build_precision_two_complement_fixed_point(
                num_bits=self.num_bits,
                point_position=self.point_position
            )
        )

        q_matrix = build_q_linear_regression(
            precision_matrix=precision_matrix,
            feature_matrix=X,
            target_matrix=y,
            fit_intercept=self.fit_intercept
        )

        array_bqm = dimod.AdjArrayBQM(q_matrix, 'BINARY')

        sample_set = samplers[self.sampler].sample(array_bqm)

        w_binary = np.asarray(list(sample_set.first.sample.values()))

        w_approx = precision_matrix.dot(w_binary)

        if self.fit_intercept:
            self.intercept_ = w_approx[0]
            self.coef_ = w_approx[1:]
        else:
            self.intercept_ = 0
            self.coef_ = w_approx

        return self
