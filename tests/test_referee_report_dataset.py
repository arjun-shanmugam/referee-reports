"""Tests RefereeReportDataset functionality.

    Author: Arjun Shanmugam
"""

import pandas as pd
import pytest

from referee_reports.referee_report_dataset import RefereeReportDataset


@pytest.fixture
def referee_report_dataset():
    return RefereeReportDataset("../test_data/intermediate/reports.txt.pkl",
                                "../../data/intermediate/papers.txt.pkl",
                                "../../output/",
                                7)


def test__format_non_vocabulary_columns(referee_report_dataset):
    referee_report_dataset._format_non_vocabulary_columns()
    assert (pd.Series(referee_report_dataset._reports_df.columns).str[0] == "_").all()
    assert (pd.Series(referee_report_dataset._reports_df.columns).str[-1] == "_").all()
    assert (pd.Series(referee_report_dataset._papers_df.columns).str[0] == "_").all()
    assert (pd.Series(referee_report_dataset._papers_df.columns).str[-1] == "_").all()


def test__build_dtm_R(referee_report_dataset):
    referee_report_dataset._format_non_vocabulary_columns()
    # Test that R is constructed properly.
    referee_report_dataset._reports_df['_cleaned_text_'] = ['The quick brown fox jumped over the lazy dog',
                                                            'The quick brown fox hopped over the lazy dog',
                                                            'The quick brown fox jumped over the lazy dog',
                                                            'The quick brown fox hopped over the lazy dog']
    referee_report_dataset._build_dtm(text_representation='R', ngrams=1)
    actual_R = referee_report_dataset._df
    expected_R = pd.DataFrame(data=[[1, 1, 1, 0, 1, 1, 1, 1, 2], [1, 1, 1, 1, 0, 1, 1, 1, 2], [1, 1, 1, 0, 1, 1, 1, 1, 2], [1, 1, 1, 1, 0, 1, 1, 1, 2]],
                              columns=['brown', 'dog', 'fox', 'hopped', 'jumped', 'lazy', 'over', 'quick', 'the'],
                              index=pd.MultiIndex.from_tuples([('99-99999', 1), ('99-99999', 2), ('99-99999', 3), ('99-99999', 4)]))
    expected_R.index = expected_R.index.rename(["paper", "refnum"])
    pd.testing.assert_frame_equal(actual_R, expected_R)


def test__build_dtm_NR(referee_report_dataset):
    # Test that NR is constructed properly
    referee_report_dataset._reports_df['_cleaned_text_'] = ['The quick brown fox jumped over the lazy dog',
                                                            'The quick brown fox hopped over the lazy dog',
                                                            'The quick brown fox jumped over the lazy dog',
                                                            'The quick brown fox hopped over the lazy dog']
    referee_report_dataset._build_dtm(text_representation='NR', ngrams=1)
    actual_NR = referee_report_dataset._df
    # Test.
    expected_NR = pd.DataFrame(data=[[1 / 9, 1 / 9, 1 / 9, 0, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 2 / 9], [1 / 9, 1 / 9, 1 / 9, 1 / 9, 0, 1 / 9, 1 / 9, 1 / 9, 2 / 9],
                                     [1 / 9, 1 / 9, 1 / 9, 0, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 2 / 9], [1 / 9, 1 / 9, 1 / 9, 1 / 9, 0, 1 / 9, 1 / 9, 1 / 9, 2 / 9]],
                               columns=['brown', 'dog', 'fox', 'hopped', 'jumped', 'lazy', 'over', 'quick', 'the'],
                               index=pd.MultiIndex.from_tuples([('99-99999', 1), ('99-99999', 2), ('99-99999', 3), ('99-99999', 4)]))
    expected_NR.index = expected_NR.index.rename(['paper', 'refnum'])
    pd.testing.assert_frame_equal(actual_NR, expected_NR)


def test__merge_with_referee_characteristics(referee_report_dataset):
    referee_report_dataset._format_non_vocabulary_columns()
    referee_report_dataset._merge_with_referee_characteristics()
    actual_df = referee_report_dataset._df
    expected_df = pd.DataFrame([["Reject", "Reject", 1], ["Major Revision", "Revise", 1], ["Accept", "Reject", 1], ["Reject", "Reject", 1]],
                               columns=["_recommendation_", "_decision_", "_female_"],
                               index=pd.MultiIndex.from_tuples([('99-99999', 1), ('99-99999', 2), ('99-99999', 3), ('99-99999', 4)]))
    expected_df.index = expected_df.index.rename(['paper', 'refnum'])
    pd.testing.assert_frame_equal(actual_df, expected_df)


def test__restrict_to_papers_with_mixed_gender_referees(referee_report_dataset):
    # Test that papers without mixed-gender refereeship are dropped.
    referee_report_dataset._df = pd.DataFrame([1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, ],
                                              columns=['_female_'],
                                              index=pd.MultiIndex.from_tuples([('99-99997', 1), ('99-99997', 2), ('99-99997', 3), ('99-99997', 4),
                                                                               ('99-99998', 1), ('99-99998', 2), ('99-99998', 3), ('99-99998', 4),
                                                                               ('99-99999', 1), ('99-99999', 2), ('99-99999', 3), ('99-99999', 4)]))
    referee_report_dataset._df.index = referee_report_dataset._df.index.rename(['paper', 'refnum'])
    referee_report_dataset._restrict_to_papers_with_mixed_gender_referees()
    actual_df = referee_report_dataset._df
    expected_df = pd.DataFrame([0, 0, 0, 1],
                               columns=['_female_'],
                               index=pd.MultiIndex.from_tuples([('99-99998', 1), ('99-99998', 2), ('99-99998', 3), ('99-99998', 4)]))
    expected_df.index = expected_df.index.rename(['paper', 'refnum'])
    pd.testing.assert_frame_equal(actual_df, expected_df)

def test__balance_sample_by_gender(referee_report_dataset):
    # Test that the sample is balanced correctly by gender.
    referee_report_dataset._df = pd.DataFrame([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, ],
                                              columns=['_female_'],
                                              index=pd.MultiIndex.from_tuples([('99-99996', 1), ('99-99996', 2), ('99-99996', 3), ('99-99996', 4),
                                                                               ('99-99997', 1), ('99-99997', 2), ('99-99997', 3), ('99-99997', 4),
                                                                               ('99-99998', 1), ('99-99998', 2), ('99-99998', 3), ('99-99998', 4),
                                                                               ('99-99999', 1), ('99-99999', 2), ('99-99999', 3), ('99-99999', 4)]))
    referee_report_dataset._df.index = referee_report_dataset._df.index.rename(['paper', 'refnum'])
    referee_report_dataset._balance_sample_by_gender()
    # Assert that reports dropped are associated with paper 99-99996 and 99-99997 (it has only male referees) and not the other papers.
    assert len(referee_report_dataset._df.loc['99-99996', :]) == 3
    assert len(referee_report_dataset._df.loc['99-99997', :]) == 3
    assert len(referee_report_dataset._df.loc['99-99998', :]) == 4
    assert len(referee_report_dataset._df.loc['99-99999', :]) == 4

