from referee_reports.document_readers import JournalDocumentReader, MalformedDocumentError
import os
from io import StringIO
import pandas as pd
import pkldir
import pytest


@pytest.fixture
def journal_document_reader():
    return JournalDocumentReader("../../test_data/raw/papers-pkl/", "../../test_data/intermediate/")


def test__validate_raw_data():
    journal_document_reader = JournalDocumentReader("../../test_data/misc_test_assets/papers_with_directory-pkl/",
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
    journal_document_reader = JournalDocumentReader("../../test_data/misc_test_assets/empty_documents-pkl/", "")
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
    journal_document_reader = JournalDocumentReader("../../test_data/raw/papers-pkl/",
                                                    "../../test_data/misc_test_assets/pickled_cleaned_test_data/")
    journal_document_reader._validate_raw_data()
    journal_document_reader._filter_duplicate_documents()
    journal_document_reader._decode_text()
    journal_document_reader._tokenize_text()
    journal_document_reader._pickle_df()
    actual = pd.read_csv(StringIO(pkldir.decode("../test_data/misc_test_assets/pickled_cleaned_test_data/journal_documents.txt.pkl").decode('utf-8')),
                         index_col='filename_without_extension')
    expected = journal_document_reader._df
    pd.testing.assert_frame_equal(actual, expected)
    assert not os.path.isfile("../../test_data/misc_test_assets/pickled_cleaned_test_data/journal_documents.txt")

# TODO: MOve below code to paper reader testing file
# def test__get_string_with_most_occurrences(paper_reader_with_nathans_paper_only, paper_reader_with_test_papers_only):
#     paper_reader_with_nathans_paper_only._decode_text()
#     paper_reader_with_nathans_paper_only._restrict_to_intro()
#
#     with open('/test_data/home/ashanmu1/Desktop/refereebias/code/misc_test_assets/JPUBE-D-99-99999_thank_yous.txt', 'r') as f:
#         expected = f.read()
#         keywords = ['thanks',
#                     'thank',
#                     'manuscript',
#                     'indebted',
#                     'comments',
#                     'discussion',
#                     'NBER',
#                     'excellent',
#                     'research assistance',
#                     'helpful',
#                     'expressed in this paper',
#                     'errors',
#                     'disclaimer',
#                     'grant',
#                     '@']
#     actual = (paper_reader_with_nathans_paper_only._df['text']
#     .str.split(pat="\n\n")
#     .apply(lambda strings: paper_reader_with_nathans_paper_only._get_string_with_most_occurrences(strings=strings,
#                                                                                                   keywords=keywords))
#     .iloc[0]
#     )
#
#     assert actual == expected, "The strings do not match!"
#
#     # Print thank you sections from each test paper to make sure that the algorithm is working as intended.
#     paper_reader_with_test_papers_only.build_df()
#
#     with open("/test_data/home/ashanmu1/Desktop/refereebias/code/cleaned_introductions.txt", 'w') as f:
#         for text, filename in zip(paper_reader_with_test_papers_only._df['cleaned_text'], paper_reader_with_test_papers_only._df['filename']):
#             f.write(filename + "\n" + text + "\n\n\n\n")
#
#     paper_reader_with_test_papers_only._decode_text()
#     with open("/test_data/home/ashanmu1/Desktop/refereebias/code/firm_choices.txt", 'w') as f:
#         f.write(paper_reader_with_test_papers_only._df.loc[paper_reader_with_test_papers_only._df['filename'] == 'FirmChoices-RES.pdf.pkl']['text'].iloc[0])
#
#
# def test__remove_thank_yous(paper_reader_with_nathans_paper_only):
#     # Define dummy keywords.
#     keywords = ['keyword 1', 'keyword 2']
#
#     # Define dummy test_data.
#     paper_reader_with_nathans_paper_only._df['text'] = pd.Series(["Section 1 of Text\n\nSection 2 of Text keyword 1 keyword 2\n\nSection 3 of Text",
#                                                                  "Section 1 of Text\n\nSection 2 of Text keyword 1\n\nSection 3 of Text\n\nSection 4 of Text keyword 1 keyword 1\n\nSection 5 of Text"],
#                                                                 index=range(2))
#
#     paper_reader_with_nathans_paper_only._remove_thank_yous(keywords=keywords)
#     expected = pd.Series(["Section 1 of Text\n\n\n\nSection 3 of Text",
#                           "Section 1 of Text\n\nSection 2 of Text keyword 1\n\nSection 3 of Text\n\n\n\nSection 5 of Text"])
#     actual = paper_reader_with_nathans_paper_only._df['text']
#     pd.testing.assert_series_equal(actual, expected, check_names=False)


# def test__get_count_per_group_of_sentences(journal_document_reader):
#     # Define list to test on.
#     ungrouped_sentences = ['Sentence 1', 'Sentence 2 WORD', 'Sentence 3 woRd',
#                            'Sentence 4', 'Sentence 5 word', 'Sentence 6',
#                            'sentence 7', 'Sentence 8', 'Sentence 9 word',
#                            'Sentence 10']
#     expected_output = pd.Series([2, 2, 2, 1, 1, 0, 1, 1])
#     pd.testing.assert_series_equal(journal_document_reader._get_count_per_group_of_sentences(ungrouped_sentences, 3, minimum_count=2, word="word")[0],
#                                    expected_output,
#                                    check_names=False)
#     assert journal_document_reader._get_count_per_group_of_sentences(ungrouped_sentences, 3, minimum_count=2, word="word")[1] == 3
#
#
# def test__restrict_to_intro(journal_document_reader):
#     journal_document_reader._decode_text()
#     journal_document_reader._restrict_to_intro()
#
#     # Check that all introductions are of a reasonable length.
#     for i in range(len(journal_document_reader._df['introduction_length'])):
#         if not (35 < journal_document_reader._df['introduction_length'].iloc[i] < 200):
#             print("Paper number " + str(i) + "'s introduction is " + str(
#                 journal_document_reader._df['introduction_length'].iloc[i]) + " sentences long. Note that this may not be an error.")
