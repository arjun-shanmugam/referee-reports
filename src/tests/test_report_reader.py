import pandas as pd

from referee_reports.document_readers import ReportReader
import pytest

@pytest.fixture()
def report_reader_right_before_merge():
    report_reader = ReportReader("../../data/raw/reports-pkl/", "", "../../data/raw/referee_gender_nonames.csv")
    report_reader._validate_raw_data()
    report_reader._filter_duplicate_documents()
    report_reader._format_index()
    report_reader._decode_text()
    return report_reader
@pytest.fixture()
def report_reader_with_incomplete_referee_characteristics():
    report_reader = ReportReader("../../data/raw/reports-pkl/", "", "../../data/test_assets/incomplete_referee_gender_nonames.csv")
    report_reader._validate_raw_data()
    report_reader._filter_duplicate_documents()
    report_reader._format_index()
    report_reader._decode_text()
    return report_reader

def test__merge_referee_characteristics(report_reader_right_before_merge, report_reader_with_incomplete_referee_characteristics):
    # Test that referee characteristics are merged correctly.
    report_reader_right_before_merge._merge_referee_characteristics()
    actual_characteristics = report_reader_right_before_merge._df[['recommendation', 'decision', 'female']]
    expected_df = pd.DataFrame([["Reject", "Reject", 1], ["Major Revision", "Revise", 1], ["Accept", "Reject", 1], ["Reject", "Reject", 1]],
                               columns=["recommendation", "decision", "female"],
                               index=pd.MultiIndex.from_tuples([('99-99999', 1), ('99-99999', 2), ('99-99999', 3), ('99-99999', 4)]))
    expected_df.index = expected_df.index.rename(['paper', 'refnum'])
    pd.testing.assert_frame_equal(actual_characteristics, expected_df)

    # Test that an error is thrown when the referee characteristics dataset is incomplete...
    # That is, we are missing referee characteristics for any reports, make sure we throw an error.
    with pytest.raises(FileNotFoundError):
        report_reader_with_incomplete_referee_characteristics._merge_referee_characteristics()