

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler

from referee_reports.models import Regression, RegularizedRegression
import pandas as pd
from sklearn.datasets import load_breast_cancer


@pytest.fixture
def get_breast_cancer_data():
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    return pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns), y


@pytest.fixture
def regularized_regression_breast_cancer():
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    return RegularizedRegression(model_name='regularized regression',
                                 y_data=y,
                                 X_data=X,
                                 add_constant=True,
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


def test_cross_validated_logistic_LASSO(get_breast_cancer_data):
    X, y = get_breast_cancer_data
    alpha_range = np.linspace(0.001, 1, 1000)
    # Run cross validated logistic Lasso using GridSearch.
    regularized_regression = RegularizedRegression(model_name='regularized regression',
                                                   y_data=y,
                                                   X_data=X,
                                                   add_constant=False,
                                                   log_transform=False,
                                                   standardize=False)
    regularized_regression.fit(penalty='l1',
                               logistic=True,
                               stratify=True,
                               cv_folds=5,
                               seed=7,
                               alphas=alpha_range,
                               adjust_alpha=False)

    # Run cross-validated logistic Lasso using LogisticRegressionCV.
    expected_results = LogisticRegressionCV(Cs=1 / (len(X) * alpha_range),
                                            fit_intercept=False,
                                            cv=5,
                                            random_state=7,
                                            solver='liblinear',
                                            penalty='l1',
                                            scoring='neg_log_loss').fit(X, y)

    assert expected_results.C_ == 1 / (regularized_regression._results_table[2]["$\\alpha^{*}$: "] * len(X))
    pd.testing.assert_series_equal(regularized_regression._results_table[0],
                                   pd.Series(expected_results.coef_.tolist()[0], index=X.columns).loc[regularized_regression._results_table[0].index])

def test_cross_validated_logistic_ridge(get_breast_cancer_data):
    X, y = get_breast_cancer_data
    alpha_range = np.linspace(0.001, 1, 1000)
    # Run cross validated logistic Lasso using GridSearch.
    regularized_regression = RegularizedRegression(model_name='regularized regression',
                                                   y_data=y,
                                                   X_data=X,
                                                   add_constant=False,
                                                   log_transform=False,
                                                   standardize=False)
    regularized_regression.fit(penalty='l2',
                               logistic=True,
                               stratify=True,
                               cv_folds=5,
                               seed=7,
                               alphas=alpha_range,
                               adjust_alpha=False)

    # Run cross-validated logistic Lasso using LogisticRegressionCV.
    expected_results = LogisticRegressionCV(Cs=1 / alpha_range,
                                            fit_intercept=False,
                                            cv=5,
                                            random_state=7,
                                            solver='liblinear',
                                            penalty='l2',
                                            scoring='neg_log_loss').fit(X, y)

    assert expected_results.C_ == 1 / regularized_regression._results_table[2]["$\\alpha^{*}$: "]


    pd.testing.assert_series_equal(regularized_regression._results_table[0],
                                   pd.Series(expected_results.coef_.tolist()[0], index=X.columns).loc[regularized_regression._results_table[0].index])

# TODO: Figure out why SKLearn results are slightly different from mine...
"""def test_cross_validated_logistic_elasticnet(get_breast_cancer_data):
    X, y = get_breast_cancer_data
    alpha_range = np.linspace(0.001, 30, 100)
    l1_wt_range = np.linspace(0.1, 1, 100)
    # Run cross validated logistic Lasso using GridSearch.
    regularized_regression = RegularizedRegression(model_name='regularized regression',
                                                   y_data=y,
                                                   X_data=X,
                                                   add_constant=False,
                                                   log_transform=False,
                                                   standardize=False)
    regularized_regression.fit(penalty='elasticnet',
                               logistic=True,
                               stratify=True,
                               cv_folds=5,
                               seed=7,
                               alphas=alpha_range,
                               l1_ratios=l1_wt_range,
                               adjust_alpha=False)

    # Run cross-validated logistic Lasso using LogisticRegressionCV.
    expected_results = LogisticRegressionCV(Cs=1 / alpha_range,
                                            fit_intercept=False,
                                            cv=5,
                                            l1_ratios=l1_wt_range,
                                            random_state=7,
                                            solver='saga',
                                            penalty='elasticnet',
                                            scoring='neg_log_loss').fit(X, y)

    assert expected_results.C_ == 1 / regularized_regression._results_table[2]["$\\alpha^{*}$: "]
    assert expected_results.l1_ratio_ == regularized_regression._results_table[2]["$w^{L1}^*$"]
    pd.testing.assert_series_equal(regularized_regression._results_table[0],
                                   pd.Series(expected_results.coef_.tolist()[0], index=X.columns).sort_values())
"""