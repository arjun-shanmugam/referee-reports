from referee_reports.document_readers import JournalDocumentReader, MalformedDocumentError
import os
from io import StringIO
import pandas as pd
import pkldir
import pytest


@pytest.fixture
def journal_document_reader():
    return JournalDocumentReader("test_data/raw/papers-pkl/", "test_data/intermediate/")


def test__validate_raw_data():
    journal_document_reader = JournalDocumentReader("test_data/misc_test_assets/papers_with_directory-pkl/",
                                                    "")
    with pytest.raises(IsADirectoryError):
        journal_document_reader._validate_raw_data()


def test__filter_duplicate_documents(journal_document_reader):
    # Test on simple DataFrame containing four identical fake files saved in different formats.
    journal_document_reader._df = pd.DataFrame.from_dict({'full_filename': ['doc_1.docx.pkl', 'doc_1.pdf.pkl', 'doc_1.txt.pkl', 'doc_1.md.pkl',
                                                                            'doc_2.docx.pkl', 'doc_2.pdf.pkl', 'doc_2.txt.pkl', 'doc_2.md.pkl']})
    journal_document_reader._filter_duplicate_documents()
    actual = journal_document_reader._df
    expected = pd.DataFrame.from_dict({'filename_without_extension': ['doc_1', 'doc_2'],
                                       'full_filename': ['doc_1.pdf.pkl', 'doc_2.pdf.pkl']}).set_index('filename_without_extension')
    pd.testing.assert_frame_equal(actual, expected)

    # Test on a DataFrame containing three identical fake files saved in different formats, none of which are given as arguments in the method call.
    journal_document_reader._df = pd.DataFrame.from_dict({'full_filename': ['doc_1.csv.pkl', 'doc_1.py.pkl', 'doc_1.pages.pkl']})
    with pytest.raises(FileNotFoundError):
        journal_document_reader._filter_duplicate_documents()


def test__decode_text(journal_document_reader):
    # Check that every row of the 'text' column contains a string.
    journal_document_reader._validate_raw_data()
    journal_document_reader._filter_duplicate_documents()
    journal_document_reader._decode_text()
    expected_output = pd.Series(str, index=journal_document_reader._df.index)
    pd.testing.assert_series_equal(journal_document_reader._df['raw_text'].apply(lambda row: type(row)), expected_output, check_names=False)

    # Check that an error is thrown when little to no text is read from one or more documents.
    journal_document_reader = JournalDocumentReader("test_data/misc_test_assets/empty_documents-pkl/", "")
    journal_document_reader._validate_raw_data()
    journal_document_reader._filter_duplicate_documents()
    with pytest.raises(MalformedDocumentError):
        journal_document_reader._decode_text()


def test__tokenize_text(journal_document_reader):
    # Test the removal of stopwords, non-alphanumeric characters, and escaped characters.
    journal_document_reader._df.loc[:, 'raw_text'] = ["The**value of \\beta is 3.4 and \\gamma equals 2.",
                                                      "The value of \\\\beta `.34.is 3.4 and \\gamma equals 2.",
                                                      "The\nvalue of \\\\\\beta is[]{3.4 and \\\\gamma \\equals 2 .\\",
                                                      "The va[\23.4lue of \\\\\\beta is 3.4 and \\gamma \\equals 2 .\\"]
    journal_document_reader._tokenize_text()
    actual = journal_document_reader._df['cleaned_text']
    expected = pd.Series(["value equals", "value equals", "value", ""])
    pd.testing.assert_series_equal(actual, expected, check_names=False)


def test__pickle_df(journal_document_reader):
    # Create test test_data and pickle it, then check that the pickled test_data is identical to the test_data in memory and that no unpickled file exists.
    journal_document_reader = JournalDocumentReader("test_data/raw/papers-pkl/",
                                                    "test_data/misc_test_assets/pickled_cleaned_test_data/")
    journal_document_reader._validate_raw_data()
    journal_document_reader._filter_duplicate_documents()
    journal_document_reader._decode_text()
    journal_document_reader._tokenize_text()
    journal_document_reader._pickle_df()
    actual = pd.read_csv(StringIO(pkldir.decode("test_data/misc_test_assets/pickled_cleaned_test_data/journal_documents.txt.pkl").decode('utf-8')),
                         index_col='filename_without_extension')
    expected = journal_document_reader._df
    pd.testing.assert_frame_equal(actual, expected)
    assert not os.path.isfile("test_data/misc_test_assets/pickled_cleaned_test_data/journal_documents.txt")
