from itertools import permutations
from typing import List, Dict, Tuple
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV
from .quantum_ridge_regression import QuantumRidge
import os
import pickle
import numpy as np
from numpy.linalg import norm
from scipy.stats import ttest_1samp, uniform, zscore

import pandas as pd


def generate_features_targets_from_dataframe(dataframe: pd.DataFrame) -> List[Dict]:
    """
    This function takes a dataframe as input and returns a list of dictionaries with features and targets
    (plus the names of the columns).

    :param dataframe: the dataframe from which to generate the combinations of features and targets
    :return: a list of dictionaries containing the names of the columns, the feature column and the target column
    """
    return [
        {
            'columns': (X, y),
            'feature': dataframe[[X]],
            'target': dataframe[y]
        }
        for X, y in permutations(dataframe.columns, 2)
    ]


def run_algorithms(
        columns: Tuple[str, str],
        X: pd.Series,
        y: pd.Series,
        alpha: float = None,
        data_path: str = None
) -> Dict:
    """
    This function runs the classical algorithm to get the closed form solution and then runs the quantum algorithm
    multiple times to draw samples.

    :param columns: the name of the columns involved in training the algorithms
    :param X: the pandas series used as input
    :param y: the output ground truth
    :param alpha: the weight of the penalization term
    :param data_path: the path to where to save the results (if None, no result is saved)
    :return: a dictionary with the name of the columns, the weights of the closed form solution and the samples
    """

    X = zscore(X)
    y = zscore(y)

    if alpha is None:
        alpha = RandomizedSearchCV(
            estimator=Ridge(),
            param_distributions={
                'alpha': uniform(loc=0, scale=10)
            },
            random_state=1
        ).fit(X, y).best_params_['alpha']

        print("The best alpha here is {}.".format(alpha))

    if data_path is not None:

        file_name: str = columns[0] + '_' + columns[1] + '_' + str(alpha) + '.pickle'
        path = data_path + file_name

        if os.path.isfile(path):
            with open(path, 'rb') as fp:
                return pickle.load(fp)

    closed_form = Ridge(alpha=alpha, solver='cholesky')
    closed_form.fit(X, y)

    result = {
        'columns': columns,
        'closed_form': np.concatenate(
            (np.asarray([closed_form.intercept_]), closed_form.coef_)
        ),
        'quantum': QuantumRidge(
            sampler='hybrid',
            alpha=alpha
        ).fit(X, y).sample_set
    }

    if data_path is not None:
        file_name: str = columns[0] + '_' + columns[1] + '_' + str(alpha) + '.pickle'
        path = data_path + file_name

        with open(path, 'wb') as fp:
            pickle.dump(result, fp)

    return result


def convert_samples_to_relative_error(
        closed_form_solution: np.ndarray,
        quantum_samples: List[np.ndarray]
) -> List[float]:
    """
    This function takes as input the closed form weights found with the Cholesky solution and compares them with the
    ones obtained through the quantum function.

    :param closed_form_solution: the fixed weights obtained through the closed form solution
    :param quantum_samples: the weights sampled from the quantum annealer
    :return: a list of the relative error between the closed form weights and
    """
    return [
        norm(closed_form_solution - sample, ord=2) / norm(closed_form_solution, ord=2)
        for sample in quantum_samples
    ]


def test_mean_error(errors: List[float]) -> bool:
    _, p_value = ttest_1samp(
        a=errors,
        popmean=0,
        alternative='greater'
    )

    return p_value > 0.05


if __name__ == '__main__':
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    dataset = pd.read_csv(url, names=names)

    features_targets = generate_features_targets_from_dataframe(
        dataframe=dataset
    )

    result = run_algorithms(
        columns=features_targets[0]['columns'],
        X=features_targets[0]['feature'],
        y=features_targets[0]['target'],
        alpha=0.0
    )

    errors = convert_samples_to_relative_error(
        closed_form_solution=result['closed_form'],
        quantum_samples=result['quantum']
    )

    print(
        test_mean_error(
            errors=errors
        )
    )
