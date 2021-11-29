from src.experiment_utils import *
from numpy.random import default_rng

datasets = {
    'Iris': pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
        names=['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    ).drop('class', axis=1),
}

rng = default_rng(seed=1)

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
            alpha=10 * rng.random(),
            data_path='data/'
        )

        print('Computing the relative error in the weights...')

        errors = convert_samples_to_relative_error(
            closed_form_solution=result['closed_form'],
            quantum_samples=result['quantum']
        )

        print('The average error is {}%.'.format(np.mean(errors)))

        boolean_sample = test_mean_error(
            errors=errors
        )

        if boolean_sample:
            print('There is statistical evidence that the error is 0.')
        else:
            print('The quantum algorithm sucks here.')

        boolean_samples.append(test_mean_error)
