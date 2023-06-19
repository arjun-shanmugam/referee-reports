import numpy as np
import pandas as pd

from referee_reports.document_readers import ReportReader, _drop_rows_with_duplicate_indices
import pytest


@pytest.fixture()
def report_reader_right_before_merge():
    report_reader = ReportReader("test_data/raw/reports-pkl/", "", "test_data/raw/referee_gender_nonames.csv")
    report_reader._validate_raw_data()
    report_reader._filter_duplicate_documents()
    report_reader._format_index()
    report_reader._decode_text()
    return report_reader


@pytest.fixture()
def report_reader_with_incomplete_referee_characteristics():
    report_reader = ReportReader("test_data/raw/reports-pkl/", "", "test_data/misc_test_assets/incomplete_referee_gender_nonames.csv")
    report_reader._validate_raw_data()
    report_reader._filter_duplicate_documents()
    report_reader._format_index()
    report_reader._decode_text()
    return report_reader


@pytest.fixture()
def report_reader_with_duplicate_authors():
    report_reader = ReportReader("test_data/raw/reports-pkl/", "", "test_data/misc_test_assets/duplicate_authors_referee_gender_nonames.csv")
    report_reader._validate_raw_data()
    report_reader._filter_duplicate_documents()
    report_reader._format_index()
    report_reader._decode_text()
    return report_reader


def test__drop_rows_with_duplicate_indices():
    # Case 1: Single-level index without a name.
    df_with_duplicate_index = pd.DataFrame([['row 1a', 'row 1b'],
                                            ['row 2a', 'row 2b'],
                                            ['row 3a', 'row 3b'],
                                            ['row 4a', 'row 4b'],
                                            ['row 5a', 'row 5b'],
                                            ['row 6a', 'row 6b'],
                                            ['row 7a', 'row 7b'],
                                            ['row 8a', 'row 8b'],
                                            ['row 9a', 'row 9b'],
                                            ['row 10a', 'row 10b']], columns=['column 1', 'column 2'], index=[1, 2, 3, 4, 4, 2, 2, 5, 6, 7])
    with pytest.warns(RuntimeWarning):
        actual_df = _drop_rows_with_duplicate_indices(df_with_duplicate_index, "")
    expected_df = pd.DataFrame([['row 1a', 'row 1b'],
                                ['row 2a', 'row 2b'],
                                ['row 3a', 'row 3b'],
                                ['row 4a', 'row 4b'],
                                ['row 8a', 'row 8b'],
                                ['row 9a', 'row 9b'],
                                ['row 10a', 'row 10b']], columns=['column 1', 'column 2'], index=[1, 2, 3, 4, 5, 6, 7])
    pd.testing.assert_frame_equal(actual_df, expected_df, check_names=False)

    # Case 2: Single-level index with a name.
    df_with_duplicate_index = pd.DataFrame([['row 1a', 'row 1b'],
                                            ['row 2a', 'row 2b'],
                                            ['row 3a', 'row 3b'],
                                            ['row 4a', 'row 4b'],
                                            ['row 5a', 'row 5b'],
                                            ['row 6a', 'row 6b'],
                                            ['row 7a', 'row 7b'],
                                            ['row 8a', 'row 8b'],
                                            ['row 9a', 'row 9b'],
                                            ['row 10a', 'row 10b']], columns=['column 1', 'column 2'], index=[1, 2, 3, 4, 4, 2, 2, 5, 6, 7])
    df_with_duplicate_index.index = df_with_duplicate_index.index.rename("my_index")
    with pytest.warns(RuntimeWarning):
        actual_df = _drop_rows_with_duplicate_indices(df_with_duplicate_index, "")
    expected_df = pd.DataFrame([['row 1a', 'row 1b'],
                                ['row 2a', 'row 2b'],
                                ['row 3a', 'row 3b'],
                                ['row 4a', 'row 4b'],
                                ['row 8a', 'row 8b'],
                                ['row 9a', 'row 9b'],
                                ['row 10a', 'row 10b']], columns=['column 1', 'column 2'], index=[1, 2, 3, 4, 5, 6, 7])
    expected_df.index = expected_df.index.rename('my_index')
    pd.testing.assert_frame_equal(actual_df, expected_df)

    # Case 3: Multiindex without a name.
    df_with_duplicate_index = pd.DataFrame([['row 1a', 'row 1b'],
                                            ['row 2a', 'row 2b'],
                                            ['row 3a', 'row 3b'],
                                            ['row 4a', 'row 4b'],
                                            ['row 5a', 'row 5b'],
                                            ['row 6a', 'row 6b'],
                                            ['row 7a', 'row 7b'],
                                            ['row 8a', 'row 8b'],
                                            ['row 9a', 'row 9b'],
                                            ['row 10a', 'row 10b']], columns=['column 1', 'column 2'],
                                           index=pd.MultiIndex.from_tuples([(1, 'A'), (2, 'B'), (3, 'C'),
                                                                            (4, 'D'), (4, 'D'), (2, 'B'),
                                                                            (2, 'B'), (5, 'E'), (6, 'F'),
                                                                            (7, 'G')]))
    with pytest.warns(RuntimeWarning):
        actual_df = _drop_rows_with_duplicate_indices(df_with_duplicate_index, "")
    expected_df = pd.DataFrame([['row 1a', 'row 1b'],
                                ['row 2a', 'row 2b'],
                                ['row 3a', 'row 3b'],
                                ['row 4a', 'row 4b'],
                                ['row 8a', 'row 8b'],
                                ['row 9a', 'row 9b'],
                                ['row 10a', 'row 10b']], columns=['column 1', 'column 2'], index=pd.MultiIndex.from_tuples([(1, 'A'), (2, 'B'), (3, 'C'),
                                                                                                                            (4, 'D'), (5, 'E'), (6, 'F'),
                                                                                                                            (7, 'G')]))
    pd.testing.assert_frame_equal(actual_df, expected_df)

    # Case 4: Multiindex with a name.
    df_with_duplicate_index = pd.DataFrame([['row 1a', 'row 1b'],
                                            ['row 2a', 'row 2b'],
                                            ['row 3a', 'row 3b'],
                                            ['row 4a', 'row 4b'],
                                            ['row 5a', 'row 5b'],
                                            ['row 6a', 'row 6b'],
                                            ['row 7a', 'row 7b'],
                                            ['row 8a', 'row 8b'],
                                            ['row 9a', 'row 9b'],
                                            ['row 10a', 'row 10b']], columns=['column 1', 'column 2'],
                                           index=pd.MultiIndex.from_tuples([(1, 'A'), (2, 'B'), (3, 'C'),
                                                                            (4, 'D'), (4, 'D'), (2, 'B'),
                                                                            (2, 'B'), (5, 'E'), (6, 'F'),
                                                                            (7, 'G')]))
    df_with_duplicate_index.index.names = ['index_level_0', 'index_level_1']
    with pytest.warns(RuntimeWarning):
        actual_df = _drop_rows_with_duplicate_indices(df_with_duplicate_index, "")
    expected_df = pd.DataFrame([['row 1a', 'row 1b'],
                                ['row 2a', 'row 2b'],
                                ['row 3a', 'row 3b'],
                                ['row 4a', 'row 4b'],
                                ['row 8a', 'row 8b'],
                                ['row 9a', 'row 9b'],
                                ['row 10a', 'row 10b']], columns=['column 1', 'column 2'], index=pd.MultiIndex.from_tuples([(1, 'A'), (2, 'B'), (3, 'C'),
                                                                                                                            (4, 'D'), (5, 'E'), (6, 'F'),
                                                                                                                            (7, 'G')]))
    expected_df.index.names = ['index_level_0', 'index_level_1']
    pd.testing.assert_frame_equal(actual_df, expected_df)


def test__drop_duplicate_authors(report_reader_with_duplicate_authors):
    # Test that we get the same referee reports DataFrame despite beginning with author number-paper duplicates in
    # referee characteristics file.
    with pytest.warns(RuntimeWarning):
        report_reader_with_duplicate_authors._merge_referee_characteristics()
    actual_df = report_reader_with_duplicate_authors._df.drop(columns=['full_filename', 'raw_text'])

    expected_df = pd.DataFrame([["Reject", "Reject", 1, 1, 0, 1, 0],
                                ["Major Revision", "Revise", 1, 1, 0, 1, 0],
                                ["Accept", "Reject", 1, 1, 0, 1, 0],
                                ["Reject", "Reject", 1, 1, 0, 1, 0],
                                ["Reject", "Reject", 1, 0, 0, 1, np.NaN],
                                ["Major Revision", "Revise", 1, 0, 0, 1, np.NaN],
                                ["Accept", "Reject", 1, 0, 0, 1, np.NaN]],
                               columns=["recommendation", "decision", "female", "author_1_female", "author_2_female", "author_3_female", "author_4_female"],
                               index=pd.MultiIndex.from_tuples([('99-99999', 1), ('99-99999', 2), ('99-99999', 3), ('99-99999', 4),
                                                                ('99-99998', 1), ('99-99998', 2), ('99-99998', 3)]))
    expected_df.index = expected_df.index.rename(['paper', 'num'])
    pd.testing.assert_frame_equal(actual_df, expected_df, check_like=True, check_dtype=False)


def test__merge_referee_characteristics(report_reader_right_before_merge, report_reader_with_incomplete_referee_characteristics):
    # Test that referee characteristics are merged correctly.
    report_reader_right_before_merge._merge_referee_characteristics()
    actual_df = report_reader_right_before_merge._df.drop(columns=['full_filename', 'raw_text'])

    expected_df = pd.DataFrame([["Reject", "Reject", 1, 1, 0, 1, 0],
                                ["Major Revision", "Revise", 1, 1, 0, 1, 0],
                                ["Accept", "Reject", 1, 1, 0, 1, 0],
                                ["Reject", "Reject", 1, 1, 0, 1, 0],
                                ["Reject", "Reject", 1, 0, 0, 1, np.NaN],
                                ["Major Revision", "Revise", 1, 0, 0, 1, np.NaN],
                                ["Accept", "Reject", 1, 0, 0, 1, np.NaN]],
                               columns=["recommendation", "decision", "female", "author_1_female", "author_2_female", "author_3_female", "author_4_female"],
                               index=pd.MultiIndex.from_tuples([('99-99999', 1), ('99-99999', 2), ('99-99999', 3), ('99-99999', 4),
                                                                ('99-99998', 1), ('99-99998', 2), ('99-99998', 3)]))
    expected_df.index = expected_df.index.rename(['paper', 'num'])
    pd.testing.assert_frame_equal(actual_df, expected_df, check_like=True, check_dtype=False)

    # Test that a warning is raised when the referee characteristics dataset is incomplete.
    # That is, we are missing referee characteristics for any reports.
    with pytest.warns(ImportWarning):
        report_reader_with_incomplete_referee_characteristics._merge_referee_characteristics()
