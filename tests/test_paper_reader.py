import re
import pandas as pd
from referee_reports.document_readers import PaperReader
import pytest

@pytest.fixture
def paper_reader():
    paper_reader = PaperReader("test_data/raw/papers-pkl/", "test_data/intermediate/")
    paper_reader._validate_raw_data()
    paper_reader._filter_duplicate_documents()
    paper_reader._format_index()
    paper_reader._decode_text()
    return paper_reader

def test__remove_jpube_cover_pages(paper_reader):
    print("\n\n\n")
    print(paper_reader._df)
    paper_reader._remove_jpube_cover_pages()
    assert not paper_reader._df['raw_text'].str.contains("Elsevier Editorial System").any()  # Test that text does not contain "Elsevier Editorial System"

    # Test that the word "abstract" only appears once in each document
    actual = paper_reader._df['raw_text'].str.count("abstract", flags=re.IGNORECASE)
    expected = pd.Series(1, index=paper_reader._df.index)
    pd.testing.assert_series_equal(actual, expected, check_names=False)

    # Check that beginning of paper is as expected.
    assert paper_reader._df['raw_text'].iloc[0][:80] == "\n\nEﬃcient Welfare Weights\n\nNathaniel Hendren∗\n\nAugust, 2019\n\nAbstract\n\nHow shoul"

def test__restrict_to_intro(paper_reader):
    # Test that we are properly restricting paper to introduction.
    paper_reader._remove_jpube_cover_pages()
    paper_reader._restrict_to_intro()
    # Test on Nathan's sample paper.
    assert paper_reader._df['raw_text'].iloc[0][-95:] == "Section 3 illustrates how the weights implement the modiﬁed Kaldor-\nHicks eﬃciency experiments."

    # Test on fictional paper text.
    paper_reader = PaperReader("", "")
    index = ['paper_1', 'paper_2']
    full_filename = ['filename_1', 'filename_2']
    paper_1_text = """This is the first sentence of paper 1. This is the second sentence of paper 1. This is the third sentence of paper 1. This is the
    fourth sentence of paper 1. This is the fifth sentence of paper 1. This sentence contains the word section. This sentence does not contain the word. 
    This sentence does not contain the word. This sentence contains the word section. This sentence contains the word section. 
    This sentence contains the word section. This sentence contains the word section. This is the first sentence of paper 1 which comes after the 
    introduction. This is the second sentence of paper 1 which comes after the introduction.""".replace("\n", "")
    paper_2_text = """This is the first sentence of paper 2. This is the second sentence of paper 2. This is the third sentence of paper 2. This is the
    fourth sentence of paper 2. This is the fifth sentence of paper 2. This sentence contains the word section. This sentence does not contain the word.
    This sentence does not contain the word. This sentence does not contain the word. This is the first sentence of paper 2 which comes after the 
    introduction. This is the second sentence of paper 2 which comes after the introduction.""".replace("\n", "")
    raw_text = [" ".join(paper_1_text.split()), " ".join(paper_2_text.split())]
    paper_reader._df = pd.DataFrame.from_dict({'full_filename': full_filename, 'raw_text': raw_text})
    paper_reader._df.index = index
    paper_reader._restrict_to_intro()
    actual = paper_reader._df
    index = ['paper_1', 'paper_2']
    full_filename = ['filename_1', 'filename_2']
    paper_1_text = """This is the first sentence of paper 1. This is the second sentence of paper 1. This is the third sentence of paper 1. This is the
    fourth sentence of paper 1. This is the fifth sentence of paper 1. This sentence contains the word section. This sentence does not contain the word. 
    This sentence does not contain the word. This sentence contains the word section. This sentence contains the word section.""".replace("\n", "")
    paper_2_text = """This is the first sentence of paper 2. This is the second sentence of paper 2. This is the third sentence of paper 2. This is the
    fourth sentence of paper 2. This is the fifth sentence of paper 2. This sentence contains the word section. This sentence does not contain the word.
    This sentence does not contain the word. This sentence does not contain the word. This is the first sentence of paper 2 which comes after the 
    introduction. This is the second sentence of paper 2 which comes after the introduction.""".replace("\n", "")
    raw_text = [" ".join(paper_1_text.split()), " ".join(paper_2_text.split())]
    cutoff_found = [True, False]
    expected = pd.DataFrame.from_dict({'full_filename': full_filename, 'raw_text': raw_text, 'cutoff_found': cutoff_found})
    expected.index = index
    pd.testing.assert_frame_equal(actual, expected)