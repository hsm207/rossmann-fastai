import collections

import numpy as np
import pandas as pd
import pytest

import custom_transform


@pytest.fixture
def sample_data():
    SampleData = collections.namedtuple('SampleData',
                                        ['train_df', 'valid_df', 'cat_names', 'cont_names'])

    train_df = pd.DataFrame({'A': np.random.randint(1, 100, 1000),
                             'B': np.random.exponential(5, 1000)})

    valid_df = pd.DataFrame({'A': np.random.randint(200, 300, 1000),
                             'B': np.random.exponential(5, 1000)})

    sample_data = SampleData(train_df=train_df,
                             valid_df=valid_df,
                             cont_names=['A', 'B'],
                             cat_names=[])

    return sample_data


def test_quantile_normalize(sample_data):
    train_df, valid_df, cat_names, cont_names = sample_data
    transformer = custom_transform.QuantileNormalize(cat_names, cont_names)

    transformer.apply_train(train_df)
    transformer.apply_test(valid_df)

    assert np.isclose(train_df.mean(), 0.500, atol=1 / 1000).all()
    assert np.isclose(train_df.std(), 0.289, atol=1 / 1000).all()

    assert np.isclose(valid_df.mean(), [1., 0.50], atol=2 / 100).all()
    assert np.isclose(valid_df.std(), [0., 0.28], atol=2 / 100).all()


def test_power(sample_data):
    train_df, valid_df, cat_names, cont_names = sample_data
    transformer = custom_transform.Power(cat_names, cont_names)

    transformer.apply_train(train_df)
    transformer.apply_test(valid_df)

    assert train_df.columns.values.tolist() == cont_names
    assert np.equal(train_df['A_sqrt'], train_df['A'] ** 0.5).all()
    assert np.equal(valid_df['B_square'], valid_df['B'] ** 2).all()
