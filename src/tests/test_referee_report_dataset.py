"""Tests RefereeReportDataset functionality.

    Author: Arjun Shanmugam
"""

import pandas as pd
import pandas.api.types as ptypes
import pytest

from referee_reports.referee_report_dataset import RefereeReportDataset


@pytest.fixture
def referee_report_dataset():
    return RefereeReportDataset("../../data/intermediate/reports.txt.pkl",
                                "../../data/intermediate/papers.txt.pkl",
                                "../../output/",
                                7)

def test__format_non_vocabulary_columns(referee_report_dataset):
    referee_report_dataset._format_non_vocabulary_columns()
    assert (pd.Series(referee_report_dataset._reports_df.columns).str[0] == "_").all()
    assert (pd.Series(referee_report_dataset._reports_df.columns).str[-1] == "_").all()
    assert (pd.Series(referee_report_dataset._papers_df.columns).str[0] == "_").all()
    assert (pd.Series(referee_report_dataset._papers_df.columns).str[-1] == "_").all()

def test__build_dtm(referee_report_dataset):
    referee_report_dataset._format_non_vocabulary_columns()
    # Test that R is constructed properly.
    referee_report_dataset._reports_df['_cleaned_text_'] = ['The quick brown fox jumped over the lazy dog',
                                                            'The quick brown fox hopped over the lazy dog',
                                                            'The quick brown fox jumped over the lazy dog',
                                                            'The quick brown fox hopped over the lazy dog']

    referee_report_dataset._papers_df['_cleaned_text_'] = ['The quick brown fox leaped over the lazy cat']
    referee_report_dataset._build_dtm(text_representation='R', ngrams=1)
    actual_R = referee_report_dataset._df.drop(columns=referee_report_dataset._reports_df.columns)
    expected_R = pd.DataFrame(data=[[1, 1, 1, 0, 1, 1, 1, 1, 2], [1, 1, 1, 1, 0, 1, 1, 1, 2], [1, 1, 1, 0, 1, 1, 1, 1, 2], [1, 1, 1, 1, 0, 1, 1, 1, 2]],
                              columns=['brown', 'dog', 'fox', 'hopped', 'jumped', 'lazy', 'over', 'quick', 'the'],
                              index=pd.MultiIndex.from_tuples([('99-99999', 1), ('99-99999', 2), ('99-99999', 3), ('99-99999', 4)]))
    expected_R.index = expected_R.index.rename(["paper", "refnum"])
    pd.testing.assert_frame_equal(actual_R, expected_R)

    # Test that NR is constructed properly
    referee_report_dataset._reports_df['_cleaned_text_'] = ['The quick brown fox jumped over the lazy dog',
                                                            'The quick brown fox hopped over the lazy dog',
                                                            'The quick brown fox jumped over the lazy dog',
                                                            'The quick brown fox hopped over the lazy dog']
    referee_report_dataset._papers_df['_cleaned_text_'] = ['The quick brown fox leaped over the lazy cat']
    referee_report_dataset._build_dtm(text_representation='NR', ngrams=1)
    actual_NR = referee_report_dataset._df.drop(columns=referee_report_dataset._reports_df.columns)
    # Test.
    expected_NR = pd.DataFrame(data=[[1 / 9, 1 / 9, 1 / 9, 0, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 2 / 9], [1 / 9, 1 / 9, 1 / 9, 1 / 9, 0, 1 / 9, 1 / 9, 1 / 9, 2 / 9],
                                     [1 / 9, 1 / 9, 1 / 9, 0, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 2 / 9], [1 / 9, 1 / 9, 1 / 9, 1 / 9, 0, 1 / 9, 1 / 9, 1 / 9, 2 / 9]],
                               columns=['brown', 'dog', 'fox', 'hopped', 'jumped', 'lazy', 'over', 'quick', 'the'],
                               index=pd.MultiIndex.from_tuples([('99-99999', 1), ('99-99999', 2), ('99-99999', 3), ('99-99999', 4)]))
    expected_NR.index = expected_R.index.rename(["paper", "refnum"])
    pd.testing.assert_frame_equal(actual_NR, expected_NR)


def test_build_df(referee_report_dataset):
    referee_report_dataset
    # referee_report_dataset.build_df(normalize_reports_with_papers=True)
    # expected_columns = ['_paper_', '_refnum_', '_recommendation_', '_decision_', '_female_']
    # actual_columns = referee_report_dataset.df.columns.tolist()
    #
    # # Test that the DataFrame contains the expected columns.
    # assert set(expected_columns).issubset(set(actual_columns))
    #
    #
    # # Test that there are no null values in the DataFrame.
    # contains_any_null_values = referee_report_dataset.df.isnull().any(axis=None)
    # assert not contains_any_null_values


def test_build_tf_matrices(referee_report_dataset):
    # Create test data with which to produce a T.F.V.
    referee_report_dataset.df = pd.DataFrame()
    referee_report_dataset.df['_cleaned_text_report_'] = ['The quick brown fox jumped over the lazy dog',
                                                          'The quick brown fox hopped over the lazy dog']
    referee_report_dataset.df['_cleaned_text_paper_'] = ['The quick brown fox leaped over the lazy cat',
                                                         'The slow mouse leaped over the lazy dog.']

    # Build report DTM.
    referee_report_dataset._build_tf_matrices(normalize_reports_with_papers=True, ngrams=1, min_df=1, max_df=1.0)

    # Test.

    expected_paper_df = pd.DataFrame(data=[[1, 0, 1, 0, 0, 1, 1, 1, 2], [0, 1, 0, 0, 0, 1, 1, 0, 2]],
                                     columns=['brown', 'dog', 'fox', 'hopped', 'jumped', 'lazy', 'over', 'quick', 'the'])
    expected_adjusted_report_df = pd.DataFrame(data=[[0, 1, 0, 0, 1, 0, 0, 0, 0], [1, 0, 1, 1, 0, 0, 0, 1, 0]],
                                               columns=['brown', 'dog', 'fox', 'hopped', 'jumped', 'lazy', 'over', 'quick', 'the'])
    pd.testing.assert_frame_equal(referee_report_dataset.tf_reports, expected_report_df)
    pd.testing.assert_frame_equal(referee_report_dataset.tf_papers, expected_paper_df)
    pd.testing.assert_frame_equal(referee_report_dataset.tf_reports_adjusted, expected_adjusted_report_df)


def test_regularized_regress(referee_report_dataset):
    referee_report_dataset.build_df(normalize_reports_with_papers=True)

    # Test for use of dependent variable which is not in the dataset.
    with pytest.raises(ValueError, match="The specified dependent variable is not a variable in the dataset."):
        referee_report_dataset.regularized_regress(y='nonexistent_dependent_variable',
                                                   X=['_refnum_', '_paper_'],
                                                   logistic=True,
                                                   normalize=True,
                                                   alphas=[0],
                                                   cv_folds=5,
                                                   stratify=True,
                                                   add_constant=True,
                                                   model_name="model1",
                                                   method='LASSO')
    # Test for use of independent variables which are not in the dataset.
    with pytest.raises(ValueError, match="One or more of the specified independent variables is not a variable in the dataset."):
        referee_report_dataset.regularized_regress(y='_female_', X=['nonexistent_independent_var_1', 'nonexistent_independent_var_2'], logistic=True,
                                                   normalize=True,
                                                   alphas=[0],
                                                   cv_folds=5,
                                                   stratify=True,
                                                   add_constant=True,
                                                   model_name="model1",
                                                   method='LASSO')
    # Test for the specification of 0 independent variables.
    with pytest.raises(ValueError, match="You must specify at least one independent variable."):
        referee_report_dataset.regularized_regress(y='_female_', X=[], logistic=True,
                                                   normalize=True,
                                                   alphas=[0],
                                                   cv_folds=5,
                                                   stratify=True,
                                                   add_constant=True,
                                                   model_name="model1",
                                                   method='LASSO')
    # Test for use of nonexistent regression algorithm.
    with pytest.raises(ValueError, match="Please specify either \'LASSO\', or \'ridge\'."):
        referee_report_dataset.regularized_regress('_female_', ['_refnum_', '_paper_'], "model", method='fake_regression_algorithm', logistic=True,
                                                   normalize=True,
                                                   alphas=[0],
                                                   cv_folds=5,
                                                   stratify=True,
                                                   add_constant=True, )


def test_add_column(referee_report_dataset):
    referee_report_dataset.build_df(normalize_reports_with_papers=True)

    # Test that adding a column which is not a Series throws an error.
    with pytest.raises(ValueError, match="The passed column is not a pandas Series."):
        referee_report_dataset.add_column(['not', 'a', 'Series'])
    # Test that adding a Series of different length throws an error.
    with pytest.raises(ValueError, match="The specified column must contain the same number of rows as the existing dataset."):
        referee_report_dataset.add_column(pd.Series(1, index=range(len(referee_report_dataset.df) - 3)))
    # Test that adding a Series with a different index throws an error.
    with pytest.raises(ValueError, match="The specified column must have an idential pandas Index compared to the existing dataset."):
        referee_report_dataset.add_column(pd.Series(1, index=range(0 + 5, len(referee_report_dataset.df) + 5)))
    # Test that adding a Series with a non-unique name throws an error.
    with pytest.raises(ValueError, match="The specified column's name must not be identical to an existing column in the dataset."):
        referee_report_dataset.add_column(pd.Series(1, index=range(len(referee_report_dataset.df)), name='gender'))

    test_column = pd.Series(list(range(len(referee_report_dataset.df))), index=referee_report_dataset.df.index, name='New Column')
    referee_report_dataset.add_column(test_column)
    pd.testing.assert_series_equal(referee_report_dataset.df['New Column'], test_column)


def test_column_datatypes(referee_report_dataset):
    referee_report_dataset.build_df(normalize_reports_with_papers=True)

    # Check that all columns are either numeric type or category type.
    for column in referee_report_dataset.df.columns:
        if column in ['_paper_', '_refnum_', '_recommendation_', '_decision_']:
            assert ptypes.is_categorical_dtype(referee_report_dataset.df[column])
        else:
            assert ptypes.is_numeric_dtype(referee_report_dataset.df[column])


def test_balance_sample_by_categorical_column(referee_report_dataset):
    referee_report_dataset.build_df(normalize_reports_with_papers=True)

    referee_report_dataset.balance_sample_by_categorical_column('_female_')
    assert referee_report_dataset.df['_female_'].value_counts()[0] == referee_report_dataset.df['_female_'].value_counts()[1]

    referee_report_dataset.df = pd.DataFrame(pd.Series([1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4], name='Test Column'))
    referee_report_dataset.balance_sample_by_categorical_column('Test Column')
    expected = {1: 3, 2: 3, 3: 3, 4: 3}
    actual = referee_report_dataset.df['Test Column'].value_counts().to_dict()
    assert expected == actual
