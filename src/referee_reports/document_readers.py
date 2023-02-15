"""
document_readers.py

Defines useful classes for reading and processing raw documents.
"""
import os
import pickle
from typing import List
import matplotlib.pyplot as plt
import pandas as pd
import pkldir
from referee_reports.constants import NLPConstants
from nltk.tokenize import word_tokenize


class JournalDocumentReader:
    _raw_pickled_documents: str
    _df: pd.DataFrame
    _cleaned_pickled_output: str

    def __init__(self, raw_pickled_documents: str, cleaned_pickled_output: str):
        self._raw_pickled_documents = raw_pickled_documents
        self._df = pd.DataFrame()
        self._cleaned_pickled_output = cleaned_pickled_output

    def _validate_raw_data(self):
        files = os.listdir(self._raw_pickled_documents)

        # Raise error if raw pickled documents directory contains any subdirectories.
        if any(os.path.isdir(os.path.join(self._raw_pickled_documents, file)) for file in files):
            raise IsADirectoryError(f"{self._raw_pickled_documents} should contain only pickled documents, but it contains a sub-directory as well.")

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
            print(
                f"Document {df.loc[0, 'full_filename'].split('.')[0]} not found in .pdf, .docx, .txt, or .md formats. Check that all document formats are valid.")
            raise FileNotFoundError(f"Document {df.loc[0, 'full_filename'].split('.')[0]} not found in .pdf, .docx, .txt, or .md formats.")

        # Select optimal format for reports which appear more than once.
        self._df['filename_without_extension'] = self._df['full_filename'].str.split(pat='.', regex=False).str[0]
        self._df['file_type'] = self._df['full_filename'].str.split(pat='.', regex=False).str[1]
        self._df = self._df.groupby(['filename_without_extension']).apply(lambda x: _choose_optimal_format(x))['full_filename']
        self._df.index = self._df.index.rename("paper")
        self._df = pd.DataFrame(self._df)

    def _decode_text(self, text_encoding='UTF-8'):
        # Extract text.
        filepaths = pd.Series(self._raw_pickled_documents, index=self._df.index).str.cat(self._df['full_filename'])
        bytes_ = filepaths.apply(lambda x: pkldir.decode(x))
        self._df['raw_text'] = bytes_.apply(lambda x: x.decode(text_encoding))

        # Check if any of the text strings are empty.
        self._df['raw_text'] = self._df['text'].fillna("")
        if (self._df['raw_text'].str.len() <= 100).any():
            empty_documents = self._df.loc[self._df['text'] == "", :].index.tolist()
            message = ",".join(empty_documents)
            raise UnicodeError(
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

    def _pickle_df(self, filename):
        print(self._df.columns)
        # Momentarily save as an unpickled CSV.
        unpickled_path = os.path.join(self._cleaned_pickled_output, filename)
        self._df.to_csv(unpickled_path)

        # Pickle CSV and save it.
        pickled_path = os.path.join(self._cleaned_pickled_output, filename + ".pkl")
        pickle.dump(pkldir.encode(unpickled_path), open(pickled_path, "wb"))

        # Delete unpickled CSV file.
        os.remove(unpickled_path)


class PaperReader(JournalDocumentReader):
    """The PaperReader class implements the functionality needed to extract features from the sample papers.

    Args:
        JournalDocReader (JournalDocumentReader): The PaperReader inherits functionality from the JournalDocReader in cases where
                                             the same functionality needs to be applied to reports in addition to papers.
    """

    def build_df(self):  # TODO: FINISH THIS METHOD
        """Builds a pandas DataFrame containing the text and ID of each paper."""

        self._validate_raw_data()
        self._filter_duplicate_documents()
        self._decode_text()

        # TODO: Below

        self._remove_jpub_e_cover_page()

        self._restrict_to_intro()

        self._remove_thank_yous()

        self._tokenize_text(remove_stopwords=True, report=False)

        self.df = self._df.drop(columns=['folder'])

        # Extract paper number from filename as a separate column.
        split_filename_without_extension = self.df['filename'].str.split(pat='.').str[0].str.split(pat='-')
        self.df['paper'] = split_filename_without_extension.str[2].str.cat(split_filename_without_extension.str[3],
                                                                           sep='-')
        self.df['file_type'] = self.df['filename'].str.split(pat='.').str[1]

        # Drop unneeded columns.
        self.df = self.df.drop(columns=['introduction_length', 'introduction_sentences', 'sentence_tokenized_text', 'cutoff', 'bytes', 'filepath',
                                        'text', 'tokenized_text'])  # TODO: Re-add filename to this list

        # Drop reports about correspond to paper 19-00063, which is a revision and should not have been included in the sample.
        print("Reports associated with paper 19-00163 have been dropped from the sample. This paper is a revision and should not have been included")
        self.df = self.df.drop(index=self.df.loc[self.df['paper'] == '19-00163'].index).reset_index(drop=True)

        # Save all columns to a pickled CSV.
        self._pickle_df('papers.txt')

    def _remove_jpub_e_cover_page_helper(self, text_of_paper: str):
        """Remove the Journal of Public Economics cover page from each paper.

        This method is meant to be passed to a pd.Series.apply() to remove
        the JPubE cover pages from each paper. Cover pages are removed
        by splitting documents on the string "1\n2\n...\n65".

        Args:
            text_of_paper (str): The text of the paper from which to remove the cover page.

        Returns:
            str: The text of the paper after removing its cover page or the original text of
                 the paper if the cover page could not be removed.
        """
        # Split paper's text on the first appearace of line numbers.
        text_to_split_on = "\n".join([str(num) for num in range(1, 66)])
        if text_to_split_on in text_of_paper:
            split = text_of_paper.split(text_to_split_on)
            # Remove text before the first appearance of line numbers; rejoin
            return "".join(split[1:])
        else:
            print("Could not separate Journal of Public Economics cover page from rest of paper.")
            return text_of_paper

    def _remove_jpub_e_cover_page(self):
        """Applies the _remove_jpub_e_cover_page_helper() function to the text of each paper.
        """
        # Case-sensitively split text on the word 'Introduction'
        self.df['text'] = self.df['text'].apply(lambda text_of_paper: self._remove_jpub_e_cover_page_helper(text_of_paper))

    def _remove_thank_yous(self, keywords: List[str] = ['thanks',
                                                        'thank',
                                                        'manuscript',
                                                        'indebted',
                                                        'comments',
                                                        'discussion',
                                                        'NBER',
                                                        'excellent',
                                                        'research assistance',
                                                        'helpful',
                                                        'expressed in this paper',
                                                        'errors',
                                                        'disclaimer',
                                                        'grant',
                                                        '@']):
        """Attempt to remove thank you sections from each paper.

        Args:
            keywords (List[str], optional): _description_. Defaults to a list of keywords which commonly appear in paper thank you sections.
        """
        # Split string by "\n\n".
        split_by_double_newline = self.df['text'].str.split(pat="\n\n")

        # Get the string with the most occurrences of keywords in the splitted list. This is the thank you's section of intro.
        thank_yous = split_by_double_newline.apply(lambda strings: self._get_string_with_most_occurrences(strings=strings,
                                                                                                          keywords=keywords))
        num_failures = len(thank_yous.loc[thank_yous == "This string is not present in the paper. Returning it so that no text is erroneously deleted."])
        print("Could not algorithmically remove thank yous and author contact information from " + str(num_failures) + " papers.")
        introductions_without_thank_yous = [text.replace(thank_you, "") for text, thank_you in zip(self.df['text'], thank_yous)]

        self.df['text'] = introductions_without_thank_yous

    def _get_string_with_most_occurrences(self, strings: List[str], keywords: List[str]):
        """A helper function which returns the string in a list of strings which contains the most occurrences of specified keywords.

        Args:
            strings (List[str]): The list of strings from which to choose the string with most occurrences.
            keywords (List[str]): The list of keywords whose appearances will be tallied.

        Returns:
            str: The string in strings which contains the most occurrences of keywords, unless:
                    -strings has length less than 3
                    -the string in strings with the maximum number of occurrences has less than 2 occurrences of keywords.
                    -the returned string contains "Introduction"
        """
        # Keep track of the number of keywords appearing in each string.
        occurences_in_each_string = []

        # If we could not split string on "\n\n", return a string which is certainly not present in the paper so we do not replace anything.
        if len(strings) < 3:
            return "This string is not present in the paper. Returning it so that no text is erroneously deleted."

        for string in strings:
            # Keep track of number of keywords appearing in current string
            occurences_in_current_string = 0
            # Count occurrences of each keyword.
            for keyword in keywords:
                occurences_in_current_string += string.lower().count(keyword.lower())
            occurences_in_each_string.append(occurences_in_current_string)

        # Only remove the string with the most occurrences if it has at least 3 occurrences; otherwise, remove nothing.
        if np.max(occurences_in_each_string) < 2:
            return "This string is not present in the paper. Returning it so that no text is erroneously deleted."

        index_of_string_with_most_occurrences = np.argmax(occurences_in_each_string)

        # Checks to ensure we are not erroneously removing paper content.
        if "Introduction" in strings[index_of_string_with_most_occurrences]:
            return "This string is not present in the paper. Returning it so that no text is erroneously deleted."

        return strings[index_of_string_with_most_occurrences]

    def _restrict_to_intro(self):
        """Restrict the papers in the sample solely to their introductions.
        """
        # Tokenize into sentences.
        self.df['sentence_tokenized_text'] = self.df['text'].apply(lambda text: sent_tokenize(text))

        # Count occurences of the word "section" in each consecutive group of three sentences.
        counts_and_cutoff = self.df['sentence_tokenized_text'].apply(lambda sentences: self._get_count_per_group_of_sentences_helper(sentences))
        uses_over_groups = pd.concat(counts_and_cutoff.str[0].values, axis=1).fillna(0).mean(axis=1)  # Average across papers.

        plot_line(x=uses_over_groups.index[:500],
                  y=uses_over_groups.values[:500],
                  filepath=os.path.join(self.path_to_output, 'uses_of_section_over_sentence_groups.png'),
                  title="Mean Number Of Occurences of \"Section\" Across Papers",
                  xlabel="Sentence Group Number",
                  ylabel="Mean Number of Occurrences")

        # Store cutoff sentences for introduction.
        self.df['cutoff'] = counts_and_cutoff.str[1]

        # Calculate each cutoff as portion of paper.
        indices_where_a_cutoff_found_algorithmically_ = self.df['cutoff'].loc[self.df['cutoff'] != float("-inf")].index
        cutoffs_as_portion_of_paper = (self.df['cutoff'].loc[indices_where_a_cutoff_found_algorithmically_] /
                                       self.df['sentence_tokenized_text'].loc[indices_where_a_cutoff_found_algorithmically_].str.len()
                                       )

        # When introduction length as portion of total length is an outlier, replace with median introduction length as portion of total length.
        median_cutoff_as_portion_of_paper = cutoffs_as_portion_of_paper.median()
        q25 = cutoffs_as_portion_of_paper.quantile(0.25)
        q75 = cutoffs_as_portion_of_paper.quantile(0.75)
        iqr = q75 - q25  # Since paper lengths are right-skewed, use IQR to find outliers.
        outlier_indices = cutoffs_as_portion_of_paper.loc[
            (cutoffs_as_portion_of_paper <= (q25 - 1.5 * iqr)) | ((q75 + 1.5 * iqr) <= cutoffs_as_portion_of_paper)].index
        outlier_total_paper_lengths = self.df['sentence_tokenized_text'].loc[outlier_indices].str.len()
        self.df.loc[outlier_indices, 'cutoff'] = median_cutoff_as_portion_of_paper * outlier_total_paper_lengths
        print("Algorithmically derived introduction cutoffs are outliers for " + str(
            outlier_indices.size) + " papers. Setting their introduction cutoffs to median of paper length-normalized cutoffs.")

        # When we cannot algorithmically derive a cutoff, set cutoff to the median cutoff among papers for which we can.
        cutoff_not_found_indices = self.df['cutoff'].loc[self.df['cutoff'] == float("-inf")].index
        cutoff_not_found_total_paper_lengths = self.df['sentence_tokenized_text'].loc[cutoff_not_found_indices].str.len()
        self.df.loc[cutoff_not_found_indices, 'cutoff'] = median_cutoff_as_portion_of_paper * cutoff_not_found_total_paper_lengths
        print("Could not algorithmically separate the introductions of " + str(
            cutoff_not_found_indices.size) + " papers. Setting their introduction cutoffs to median of paper length-normalized cutoffs.")

        # Restrict each paper to only sentences preceeding the cutoff sentence.
        restricted_sentences = []
        for sentences, cutoff in zip(self.df['sentence_tokenized_text'], self.df['cutoff']):
            restricted_sentences.append(sentences[:int(cutoff)])

        # Store sentences comprising introduction as a column.
        self.df['introduction_sentences'] = restricted_sentences

        # Plot histogram of introduction lengths.
        self.df['introduction_length'] = self.df['introduction_sentences'].str.len()
        xlabel = '''Length (Sentences)

            Note: This figure is a histogram of the number of sentences in the sample papers after removing the cover pages and restricting
              to the introductions. Non-alphabetic tokens, tokens less than three characters in length, and stopwords have yet to be
              removed at this point.
                    '''
        title = "Paper Length Immediately After Restriction"
        filepath = os.path.join(self.path_to_output, 'hist_sentences_per_paper_after_restriction_before_tokenization.png')
        plot_histogram(x=self.df['introduction_length'], filepath=filepath, title=title, xlabel=xlabel)

        # Plot histogram of introduction lengths divided by total lengths.
        introduction_lengths_as_portion_of_paper_lengths = self.df['introduction_length'] / self.df['sentence_tokenized_text'].str.len()
        xlabel = '''Length (Sentences, As a Portion of Total Sentences In Paper)

        Note: This figure is a histogram of the number of sentences in the sample papers after removing the cover pages and restricting
              to the introductions. Non-alphabetic tokens, tokens less than three characters in length, and stopwords have yet to be
              removed at this point. All sentence counts were divided by the total number of sentences in the original paper. 
                    '''
        title = "Paper Length Immediately After Restriction as a Portion of Original Paper Length"
        filepath = os.path.join(self.path_to_output, 'hist_sentences_per_paper_normalized_after_restriction_before_tokenization.png')
        plot_histogram(x=introduction_lengths_as_portion_of_paper_lengths, filepath=filepath, title=title, xlabel=xlabel, decimal_places=3)

        # Reset the 'text' column to contain only text from paper introductions.
        self.df['text'] = self.df['introduction_sentences'].str.join(" ")

    def _get_count_per_group_of_sentences_helper(self, ungrouped_sentences, group_size=3, minimum_count=2, word="section"):
        """Case-insensitively calculates the number of occurences of 'word' in every unique consecutive group of 'group size' sentences.

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

        # Loop through sentences, calculating occurences in each sentence group.
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
        return pd.Series(num_occurences_per_group), cutoff_sentence


class ReportReader(JournalDocumentReader):

    def __init__(self, raw_pickled_documents: str, cleaned_pickled_output: str, path_to_output: str, path_to_referee_characteristics: str):
        JournalDocumentReader.__init__(self, raw_pickled_documents, cleaned_pickled_output, path_to_output)
        self.path_to_referee_characteristics = path_to_referee_characteristics

    def build_df(self):
        """Builds a pandas DataFrame containing text of each report."""

        self._decode_text()

        self._tokenize_text(remove_stopwords=True, report=True)

        self.df = self._df.drop(columns=['folder'])

        # Extract referee number and paper number from filename as a separate column.
        paperid_refid_separated = self.df['filename'].str.split(pat='.').str[0].str.split(pat=' ref ')
        self.df['paper'] = paperid_refid_separated.str[0]
        self.df['refnum'] = paperid_refid_separated.str[1].astype('int64')

        self.df['file_type'] = self.df['filename'].str.split(pat='.').str[1]

        # Many-to-one merge with referee characteristics.
        characteristics_df = pd.read_csv(self.path_to_referee_characteristics)
        self.df = self.df.merge(characteristics_df,
                                how='inner',
                                left_on=['paper', 'refnum'],
                                right_on=['paper', 'refnum'],
                                suffixes=('_left', '_right'),
                                validate='1:1')

        # Save all columns to pickled CSV.
        self.df = self.df.drop(columns=['tokenized_text', 'filename', 'filepath', 'bytes', 'text'])
        self._pickle_df('reports.txt')
