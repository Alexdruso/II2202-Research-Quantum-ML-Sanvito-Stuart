from src.experiment_utils import *
from numpy.random import default_rng
from scipy.stats import binom_test

datasets = {
    'Iris': pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
        names=['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    ).drop('class', axis=1),
    'Boston': pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data",
        names=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'],
        delim_whitespace=True
    ),
    'Diabetes': pd.read_csv(
        "https://www4.stat.ncsu.edu/~boos/var.select/diabetes.tab.txt",
        delim_whitespace=True
    )
}

boolean_samples = []

for name, dataset in datasets.items():
    print('Running the experiments on {} dataset...'.format(name))

    features_targets = generate_features_targets_from_dataframe(
        dataframe=dataset
    )

    for feature_target in features_targets:
        print('Considering columns {}...'.format(feature_target['columns']))

        result = run_algorithms(
            columns=feature_target['columns'],
            X=feature_target['feature'],
            y=feature_target['target'],
            data_path='data/'
        )

        print('Computing the relative error in the weights...')

        errors = convert_samples_to_relative_error(
            closed_form_solution=result['closed_form'],
            quantum_samples=result['quantum']
        )

        print('The average error is {}%.'.format(np.mean(errors)*100))

        boolean_sample = test_mean_error(
            errors=errors
        )

        if boolean_sample:
            print('There is statistical evidence that the error is 0.')
        else:
            print('The quantum algorithm sucks here.')

        boolean_samples.append(boolean_sample)

p_value = binom_test(
    x=boolean_samples.count(True),
    n=len(boolean_samples),
    p=1,
    alternative='less'
)

print(boolean_samples.count(True)/len(boolean_samples))
print(len(boolean_samples))

print(
    'Overall, the evidence indicates that the quantum algorithm {} the same results of the classical one.'.format(
        'does not reach' if p_value > 0.05 else 'reaches'
    )
)
