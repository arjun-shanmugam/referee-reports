import pytest
from referee_reports.models import Regression, RegularizedRegression
import pandas as pd
from sklearn.datasets import load_breast_cancer

@pytest.fixture
def regularized_regression_breast_cancer():
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    return RegularizedRegression(model_name='regularized regression',
                                 y_data=y,
                                 X_data=X,
                                 add_constant=True,
                                 logistic=False,
                                 log_transform=False,
                                 standardize=True)
def test_preprocess_inputs():
    # Create a test dataset containing both numeric and categorical columns.
    test_X = pd.concat([pd.DataFrame([[0, 2], [0, 2], [1, 4], [1, 4]], columns=['Column 1', 'Column 2'], dtype='int64'),
                        pd.Series(['Dog', 'Cat', 'Mouse', 'Cat'], name='Column 3', dtype='category')],
                       axis=1)
    regression = Regression(model_name='regression',
                            y_data=None,
                            X_data=test_X,
                            add_constant=True,
                            logistic=False,
                            log_transform=False,
                            standardize=True)
    regression._preprocess_inputs()
    expected_X = pd.DataFrame([[-1., -1., 1, 0, 1], [-1., -1., 0, 0, 1], [1., 1., 0, 1, 1], [1., 1., 0, 0, 1]], columns=['Column 1',
                                                                                                                         'Column 2',
                                                                                                                         'Column 3: Dog',
                                                                                                                         'Column 3: Mouse',
                                                                                                                         'Constant'])
    actual_X = regression._X_data
    pd.testing.assert_frame_equal(expected_X, actual_X, check_like=True)  # Do not check for exact order of columns or index.

