from referee_reports.document_readers import ReportReader
import pytest

@pytest.fixture()
def report_reader_right_before_merge():
    report_reader = ReportReader("../../data/raw/reports-pkl/", "", "../../data/test_assets/incomplete_referee_gender_nonames.csv")
    report_reader._validate_raw_data()
    report_reader._filter_duplicate_documents()
    report_reader._format_index()
    report_reader._decode_text()
    return report_reader

def test__merge_referee_characteristics(report_reader_right_before_merge):
    # Test that an error is thrown when the referee characteristics dataset is incomplete...
    # That is, we are missing referee characteristics for any reports, make sure we throw an error.
    with pytest.raises(FileNotFoundError):
        report_reader_right_before_merge._merge_referee_characteristics()