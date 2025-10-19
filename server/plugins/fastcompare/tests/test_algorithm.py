import time

import numpy as np
from plugins.fastcompare.algo.ease import EASE
from plugins.fastcompare.algo.wrappers.data_loadering import MLDataLoaderWrapper

import pytest

from plugins.fastcompare import filter_params

tested_algorithm_combinations = [
    (EASE, {"displayed_name": "Something", "positive_threshold": 3.0, "l2": 0.01}),
    # Add more parameter combinations if needed (e.g. add your algorithm implementation)
]

@pytest.fixture(scope="session")
def loader():
    print(f"Loading data, takes some time")
    loader = MLDataLoaderWrapper()
    loader.load_data()
    return loader

@pytest.fixture(scope="session")
def fitted_algorithm(request, loader):
    algorithm_factory, algorithm_parameters = request.param
    print(f"Fitted_algorithm fixture with: {algorithm_factory}")
    assert hasattr(algorithm_factory, "name") and callable(getattr(algorithm_factory, "name")), "Algorithm must provide name() method"
    algorithm_name = algorithm_factory.name()


    # Construct the algorithm with parameters from config
    # And construct the algorithm
    algorithm = algorithm_factory(loader, **filter_params(algorithm_parameters, algorithm_factory))
    algorithm_displayed_name = algorithm_parameters["displayed_name"]
    print(f"Training algorithm: {algorithm_name}, {algorithm_displayed_name}")
    algorithm.fit()
    print(f"Done training algorithm: {algorithm_displayed_name}")
    return algorithm

# Verify prediction filtering works as expected
@pytest.mark.parametrize(
    'fitted_algorithm',
    tested_algorithm_combinations,
    indirect=True
)
def test_predict_filtering(fitted_algorithm, loader):
    all_items = loader.ratings_df.item.unique()
    np.random.seed(42)
    for k in range(1, 101):
        # Filter out all but K items
        filter_out_items = np.random.choice(all_items, all_items.size - k)
        res = fitted_algorithm.predict([], filter_out_items, k)
        intersection = set(res).intersection(filter_out_items)
        assert len(intersection) == 0, f"Intersection should be empty: {intersection}"


# Verify prediction shape is as expected
@pytest.mark.parametrize(
    'fitted_algorithm',
    tested_algorithm_combinations,
    indirect=True
)
def test_predict_shape(fitted_algorithm):
    for k in range(1, 101):
        res = fitted_algorithm.predict([], [], k)
        assert len(res) == k, f"{len(res)} != {k}"


# Verify "rough prediction speed" is in reasonable range
@pytest.mark.parametrize(
    'fitted_algorithm',
    tested_algorithm_combinations,
    indirect=True
)
def test_predict_speed(fitted_algorithm):
    n_iters = 10
    for k in range(1, 101):
        start_time = time.perf_counter()
        for i in range(n_iters):
            _ = fitted_algorithm.predict([], [], k)
        tot_time = time.perf_counter() - start_time
        assert (tot_time / n_iters) < 1.0, f"k={k} has average time per iteration: {tot_time / n_iters} >= 1.0"