"""
document_readers.py

Defines useful classes for reading and processing raw documents.
"""
import os
import pickle
import re
from typing import List
import pandas as pd
import pkldir
import referee_reports.document_readers
from referee_reports.constants import NLPConstants
from nltk.tokenize import sent_tokenize, word_tokenize


class JournalDocumentReader:
    """
    TODO
    """
    _raw_pickled_documents_directory: str
    _df: pd.DataFrame
    _cleaned_pickled_output_directory: str

    def __init__(self, raw_pickled_documents_directory: str, cleaned_pickled_output_directory: str):
        self._raw_pickled_documents_directory = raw_pickled_documents_directory
        self._df = pd.DataFrame()
        self._cleaned_pickled_output_directory = cleaned_pickled_output_directory

    def _validate_raw_data(self):
        files = os.listdir(self._raw_pickled_documents_directory)

        # Raise error if raw pickled documents directory contains any subdirectories.
        if any(os.path.isdir(os.path.join(self._raw_pickled_documents_directory, file)) for file in files):
            raise IsADirectoryError(f"{self._raw_pickled_documents_directory} should contain only pickled documents, but it contains a sub-directory as well.")

        self._df['full_filename'] = files

    def _filter_duplicate_documents(self):
        def _choose_optimal_format(df):
            ordered_file_type_preferences = ["pdf", "docx", "txt", "md"]
            for file_type in ordered_file_type_preferences:
                # For each possible file type in order of preference, check if a row with that file type exists.
                if df['file_type'].str.contains(file_type).any():
                    # If it does, return it. Otherwise, we check for the next best file type.
                    aggregated = df.loc[df['file_type'] == file_type, :].iloc[0].copy()
                    aggregated.name = None
                    return aggregated
            # If none of the preferred file types exist in the 'file_type' column, raise an error.
            raise FileNotFoundError(f"Document {df.loc[0, 'full_filename'].split('.')[0]} not found in .pdf, .docx, .txt, or .md formats."
                                    "Check that all document formats are valid.")

        # Select optimal format for reports which appear more than once.
        filename_split = self._df['full_filename'].str.split(pat='.', regex=False)
        self._df['filename_without_extension'] = filename_split.str[0]
        self._df['file_type'] = filename_split.str[1]
        self._df = self._df.groupby(['filename_without_extension']).apply(lambda x: _choose_optimal_format(x))['full_filename']
        self._df = pd.DataFrame(self._df)

    def _decode_text(self, text_encoding='UTF-8'):
        # Extract text.
        filepaths = pd.Series(self._raw_pickled_documents_directory, index=self._df.index).str.cat(self._df['full_filename'])
        bytes_ = filepaths.apply(lambda x: pkldir.decode(x))
        self._df['raw_text'] = bytes_.apply(lambda x: x.decode(text_encoding))

        # Check if any of the text strings are empty.
        self._df['raw_text'] = self._df['raw_text'].fillna("")
        if (self._df['raw_text'].str.len() <= 100).any():
            empty_documents = self._df.loc[self._df['raw_text'] == "", :].index.tolist()
            message = ",".join(empty_documents)
            raise MalformedDocumentError(
                f"No text or almost no text was extracted for the following documents: {message}. Check raw files for irregular formatting, etc.")

    def _tokenize_text(self):
        def _mwe_retokenize(tokens, token_of_interest):
            retokenized_list = []
            i = 0
            tokens_enum = enumerate(tokens)
            for i, current_token in tokens_enum:
                if current_token == token_of_interest and i < len(tokens) - 1:
                    index = i
                    next_token = tokens[i + 1]
                    current_token_retokenized = current_token
                    while next_token == token_of_interest and index < len(tokens):
                        next(tokens_enum)
                    if index < len(tokens) - 1:
                        current_token_retokenized = current_token_retokenized + next_token
                        current_token = next(tokens_enum)
                    retokenized_list.append(current_token_retokenized)
                else:
                    retokenized_list.append(current_token)
            return retokenized_list

        def _restrict_tokens(tokens: List[str]):  # TODO
            restricted_tokens = []
            for token in tokens:
                if token.isalpha() and len(token) > 2 and token.lower() not in NLPConstants.NLTK_STOPWORDS_ENGLISH:
                    restricted_tokens.append(token)
            return restricted_tokens

        self._df['cleaned_text'] = (self._df['raw_text']
                                    .str.replace("\newline|\n", " ", regex=True)  # Remove all newlines.
                                    .apply(lambda text: word_tokenize(text))  # Tokenize text using NLTK's recommended word tokenizer.
                                    # Re-combine backslashes in case backslashes are split by the algorithm as separate tokens.
                                    .apply(lambda tokens: _mwe_retokenize(tokens, "\\"))
                                    .apply(lambda tokens: _restrict_tokens(tokens))  # Remove tokens containing non-alphabetic characters and stopwords.
                                    .str.join(" ").str.lower()  # Join into one string of space-delimited tokens.
                                    )

    def _pickle_df(self):
        if type(self) == referee_reports.document_readers.PaperReader:
            filename = "papers.txt"
        elif type(self) == referee_reports.document_readers.ReportReader:
            filename = "reports.txt"
        elif type(self) == referee_reports.document_readers.JournalDocumentReader:  # For testing purposes only.
            filename = "journal_documents.txt"
        else:
            raise NotImplementedError("The method _pickle_df was called by an object of an unrecognized class, and cannot automatically name the cleaned,"
                                      "pickled output file. See the method definition in referee_reports.document_readers.JournalDocumentReader.")

        # At this point, we no longer have use for full_filename.
        self._df = self._df.drop(columns=['full_filename'])

        # Momentarily save as an unpickled CSV.
        unpickled_path = os.path.join(self._cleaned_pickled_output_directory, filename)
        self._df.to_csv(unpickled_path)

        # Pickle CSV and save it.
        pickled_path = os.path.join(self._cleaned_pickled_output_directory, filename + '.pkl')
        with open(pickled_path, "wb") as file:
            pickle.dump(pkldir.encode(unpickled_path), file)

        # Delete unpickled CSV file.
        os.remove(unpickled_path)


class MalformedDocumentError(Exception):
    """TODO"""


class PaperReader(JournalDocumentReader):
    """
    TODO
    """

    def build_df(self):
        """Builds a pandas DataFrame containing the text and ID of each paper."""
        self._validate_raw_data()
        self._filter_duplicate_documents()
        self._format_index()
        self._decode_text()
        self._remove_jpube_cover_pages()
        self._restrict_to_intro()
        self._tokenize_text()
        self._pickle_df()
        # TODO: Drop paper 19-00063, which is a revision and should not have been included in the sample.

    def _format_index(self):
        self._df.index = ['-'.join(paper.split('-')[-2:]) for paper in self._df.index]
        self._df.index = self._df.index.rename("paper")

    def _remove_jpube_cover_pages(self):
        def _remove_jpube_cover_page_from_single_paper(row, split_on):
            if split_on.lower() in row['raw_text'].lower():
                # Use regex module to split so that we can do so case-insensitively (we will lowercase the string during tokenization, not here).
                split = re.split(split_on, row['raw_text'], flags=re.IGNORECASE)
                raw_text_without_cover_page = "".join(split[1:])
                return pd.Series([row['full_filename'], raw_text_without_cover_page])
            else:
                raise MalformedDocumentError(f"Could not separate JPUBE cover page from paper {row.name}. Check that the JPUBE cover page is "
                                             f"separated from the text of this manuscript by the phrase \"Click here to view linked References\" "
                                             f"(any combination of uppercase or lowercase).")

        # Case-insensitively split text on the phrase "click here to view linked references" (taken from 97-hendren.pdf)
        text_to_split_on = "click here to view linked references"
        self._df = self._df.apply(lambda row: _remove_jpube_cover_page_from_single_paper(row, text_to_split_on),
                                  axis=1,
                                  result_type='broadcast')

    def _restrict_to_intro(self):
        """Restrict the papers in the sample solely to their introductions.
        """
        # Tokenize into sentences.
        sentence_tokenized_papers = self._df['raw_text'].apply(lambda text: sent_tokenize(text))

        def _estimate_intro_boundary(ungrouped_sentences: List[str], group_size: int, minimum_count: int, word: str):
            """
           TODO: Rewrite

            Case-insensitively calculates the number of occurences of 'word' in every unique consecutive group of 'group size' sentences.

                Returns a tuple containing:
                    -a Series giving the number of occurences in each sentence group
                    -the index of the last sentence in the first sentence group which has the specified number of occurrences.
            """
            # Store number of sentences in paper.
            num_sentences = len(ungrouped_sentences)

            # List to store number of occurences in each sentence group.
            num_occurences_per_group = []

            # To store the index of the last sentence in the first sentence group which contains enough uses of the word of interest.
            cutoff_sentence = float("-inf")

            # Loop through sentences, calculating occurrences in each sentence group.
            for i in range(num_sentences):
                count = 0
                # Continue to next iteration of loop if there are fewer remaining sentences than the group size.
                if num_sentences - i < group_size:
                    continue
                # Check the current sentence and the next <group_size> sentences for occurrences of the word of interest.
                for sentence in ungrouped_sentences[i: i + group_size]:
                    if word.lower() in sentence.lower():
                        # count = count + sentence.lower().count(word.lower())
                        if word.lower() in sentence.lower():
                            count += 1
                num_occurences_per_group.append(count)
                # Store the index of the last sentence in the first sentence group which contains enough uses of the word of interest.
                if count >= minimum_count and cutoff_sentence == float("-inf"):
                    cutoff_sentence = i + group_size
            return cutoff_sentence

        # Count occurrences of the word "section" in each consecutive group of three sentences.
        cutoffs = sentence_tokenized_papers.apply(lambda list_of_sentences: _estimate_intro_boundary(list_of_sentences,
                                                                                                     group_size=3,
                                                                                                     minimum_count=2,
                                                                                                     word="section"))
        # This Series equals -inf for papers where no cutoff could be found. Replace those cutoffs with the total number of sentences in the paper.
        cutoffs = cutoffs.where(cutoffs != float("-inf"), sentence_tokenized_papers.str.len())
        # Dummy indicating whether a cutoff could be found.
        self._df['cutoff_found'] = (cutoffs != sentence_tokenized_papers.str.len())  # If cutoff is different from the length of the paper, a cutoff was found.

        # Restrict each paper to only sentences preceding the cutoff sentence.
        restricted_sentences = []
        for sentences, cutoff in zip(sentence_tokenized_papers, cutoffs):
            restricted_sentences.append(sentences[:int(cutoff)])
        self._df['raw_text'] = pd.Series(restricted_sentences, index=self._df.index).str.join(" ")


class ReportReader(JournalDocumentReader):
    """
    TODO
    """
    _referee_characteristics_file: str

    def __init__(self, raw_pickled_documents_directory: str, cleaned_pickled_output_directory: str, referee_characteristics_file: str):
        JournalDocumentReader.__init__(self, raw_pickled_documents_directory, cleaned_pickled_output_directory)
        self._referee_characteristics_file = referee_characteristics_file

    def build_df(self):
        """Builds a pandas DataFrame containing text of each report."""
        self._validate_raw_data()
        self._filter_duplicate_documents()
        self._format_index()
        self._decode_text()
        self._merge_referee_characteristics()
        self._tokenize_text()
        self._pickle_df()

    def _format_index(self):
        self._df = self._df.reset_index()
        unformatted_index = self._df['filename_without_extension'].str.split(" ")
        self._df = self._df.drop(columns=['filename_without_extension'])
        self._df.loc[:, 'paper'] = unformatted_index.str[0]
        self._df.loc[:, 'refnum'] = unformatted_index.str[2].astype(int)
        self._df = self._df.set_index(['paper', 'refnum'])

    def _merge_referee_characteristics(self):
        referee_characteristics_df = pd.read_csv(self._referee_characteristics_file, index_col=['paper', 'refnum'])
        number_of_reports = len(self._df)
        self._df = pd.merge(self._df, referee_characteristics_df, right_index=True, left_index=True,
                            validate='1:1', how='inner')
        number_of_reports_merged_with_referee_characteristics = len(self._df)
        if number_of_reports != number_of_reports_merged_with_referee_characteristics:
            raise FileNotFoundError(f"A total of "
                                    f"{number_of_reports - number_of_reports_merged_with_referee_characteristics}"
                                    f"referee reports could not be merged with referee characteristics. Check that"
                                    f"all referee reports have a corresponding row in referee_gender_nonames.csv")
