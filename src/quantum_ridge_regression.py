from sklearn.linear_model import Ridge
import dimod
from dwave.system.samplers import LeapHybridSampler, DWaveSampler
from dimod import sampleset

from .utils import *


def build_q_ridge_regression(
        precision_matrix: np.array,
        feature_matrix: np.array,
        target_matrix: np.array,
        fit_intercept: bool = False,
        alpha: float = 1.0
) -> np.array:
    augmented_feature_matrix = build_augmented_feature_matrix(feature_matrix) if fit_intercept else feature_matrix
    bias = -2 * precision_matrix.T @ augmented_feature_matrix.T @ target_matrix
    coupler = precision_matrix.T @ augmented_feature_matrix.T @ augmented_feature_matrix @ precision_matrix
    penalization = precision_matrix.T @ precision_matrix

    return coupler + np.diag(bias) + alpha * penalization


samplers = {
    'quantum': DWaveSampler,
    'hybrid': LeapHybridSampler,
    'simulated': dimod.SimulatedAnnealingSampler
}


class QuantumRidge(Ridge):
    def __init__(
            self,
            alpha: float = 1.0,
            fit_intercept: bool = True,
            positive: bool = True,
            num_bits: int = 32,
            point_position: float = 0.5,
            sampler: str = 'simulated'
    ):
        self.sample_set: sampleset = None
        self.positive = positive
        self.num_bits = num_bits
        self.point_position = point_position
        self.sampler = sampler
        super().__init__(alpha=alpha, fit_intercept=fit_intercept)

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

        q_matrix = build_q_ridge_regression(
            precision_matrix=precision_matrix,
            feature_matrix=X,
            target_matrix=y,
            fit_intercept=self.fit_intercept,
            alpha=self.alpha
        )

        array_bqm = dimod.AdjArrayBQM(q_matrix, 'BINARY')

        sample_set = samplers[self.sampler]().sample(array_bqm)

        self.sample_set = list(
            map(
                lambda solution: precision_matrix @ solution,
                map(
                    lambda record: record[0],
                    sample_set.record
                )
            )
        ) if self.sampler in {'quantum, simulated'} else list(
            map(
                lambda solution: precision_matrix @ solution,
                map(
                    lambda record: record[0],
                    [samplers[self.sampler]().sample(array_bqm).record[0] for _ in range(3)]
                )
            )
        )

        w_binary = np.asarray(list(sample_set.first.sample.values()))

        w_approx = precision_matrix.dot(w_binary)

        if self.fit_intercept:
            self.intercept_ = w_approx[0]
            self.coef_ = w_approx[1:]
        else:
            self.intercept_ = 0
            self.coef_ = w_approx

        return self


if __name__ == '__main__':
    print(
        QuantumRidge(sampler='simulated', alpha=0.0, fit_intercept=False).fit(
            np.asarray([[1], [1]]),
            np.asarray([1, 1])
        ).coef_
    )
