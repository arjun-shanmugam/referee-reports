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
from constants import NLPConstants
from nltk.tokenize import word_tokenize


class JournalDocReader:
    """Defines text analysis functionality that is used by the PaperReader and ReportReader classes.

    Defines functionality including repeated document analysis, optimal file format selection, depickling
    and decoding of text, tokenizing and cleaning of text, and re-pickling of text.
    """
    path_to_pkl_documents: str
    df: pd.DataFrame
    path_to_intermediate_data: str
    path_to_output: str

    def __init__(self, path_to_pkl_documents: str, path_to_intermediate_data: str, path_to_output: str):
        self.path_to_pkl_documents = path_to_pkl_documents
        self.df = pd.DataFrame()
        self.path_to_intermediate_data = path_to_intermediate_data
        self.path_to_output = path_to_output

    def analyze_repeated_documents(self, reports: bool):
        """Produces pie charts giving summary statistics on the different file formats of documents in the sample.

        Args:
            reports (bool): Boolean indicating whether reports or papers are being analyzed.
        """
        files = os.listdir(self.path_to_pkl_documents)

        # Warn user if directory contains directories as well as files.
        nonfile_counter = 0
        for file in files:
            if not os.path.isfile(os.path.join(self.path_to_pkl_documents, file)):
                nonfile_counter += 1
        if nonfile_counter > 0:
            print("WARNING: Data directory contains " + str(nonfile_counter) + " directories.")

        # Separate filename from extension.
        temp_df = pd.DataFrame()
        temp_df['filename'] = files
        temp_df['filename_without_extension'] = temp_df['filename'].str.split(pat='.').str[0]
        temp_df['file_type'] = temp_df['filename'].str.split(pat='.').str[1]

        fig, (ax1, ax2) = plt.subplots(1, 2)

        # Produce pie chart of counts of each document
        paper_counts = pd.Series(temp_df.groupby('filename_without_extension').agg('count').groupby('file_type').agg('count')['filename'])
        if reports:
            title = "Number of Different \n File Types in Which \n Each Report Appears"
        else:
            title = "Number of Different \n File Types in Which \n Each Paper Appears"
        paper_counts.plot(kind='pie',
                          ax=ax1,
                          title=title,
                          ylabel="",
                          autopct='%1.1f%%',
                          explode=pd.Series(0.05, index=paper_counts.index))

        # Produce pie chart of format distribution
        if reports:
            title = "Distribution of \n Best-Choice \n File Formats (Reports)"
            filename = "filetype_analysis_reports.png"
        else:
            title = "Distribution of \n Best-Choice \n File Formats (Papers)"
            filename = "filetype_analysis_papers.png"
        file_type_counts = pd.Series(self.df['file_type'].value_counts())
        file_type_counts.plot(kind='pie',
                              ax=ax2,
                              title="Distribution of \n Best-Choice \n File Formats",
                              ylabel="",
                              autopct='%1.1f%%',
                              explode=pd.Series(0.05, index=file_type_counts.index))

        plt.savefig(os.path.join(self.path_to_output, filename), bbox_inches='tight')

    def _decode_text(self, text_encoding='UTF-8', preferred_formats=["pdf", "docx", "txt", "md"], paper=False):
        files = os.listdir(self.path_to_pkl_documents)
        files_to_decode = []

        # Warn user if directory contains directories as well as files.
        nonfile_counter = 0
        for file in files:
            if not os.path.isfile(os.path.join(self.path_to_pkl_documents, file)):
                nonfile_counter += 1
        if nonfile_counter > 0:
            print("WARNING: Data directory contains " + str(nonfile_counter) + " directories.")

        # Separate filename from extension.
        temp_df = pd.DataFrame()
        temp_df['filename'] = files
        temp_df['filename_without_extension'] = temp_df['filename'].str.split(pat='.').str[0]
        temp_df['file_type'] = temp_df['filename'].str.split(pat='.').str[1]

        # Select optimal format for reports which appear more than once.
        self.df['filename'] = temp_df.groupby(['filename_without_extension']).apply(lambda x: self._choose_optimal_format(x, preferred_formats)).reset_index(
            drop=True)

        # Append full file path to each filename and extract text.
        self.df['folder'] = self.path_to_pkl_documents
        self.df['filepath'] = self.df['folder'].str.cat(self.df['filename'])
        self.df['bytes'] = self.df['filepath'].apply(lambda x: pkldir.decode(x))
        self.df['text'] = self.df['bytes'].apply(lambda x: x.decode(text_encoding))

        # Check if any of the text strings are empty.
        self.df['text'] = self.df['text'].replace({"": "THIS DOCUMENT IS EMPTY"})

        # Drop empty documents
        index_of_empty_documents = self.df.loc[self.df['text'] == "THIS DOCUMENT IS EMPTY"].index
        self.df = self.df.drop(index=index_of_empty_documents)
        print(str(len(index_of_empty_documents)) + " empty documents were found and dropped from the sample.")

    def _restrict_tokens(self, tokens: List[str], remove_stopwords):
        restricted_tokens = []
        for token in tokens:
            if remove_stopwords:
                if token.isalpha() and len(token) > 2 and token.lower() not in NLPConstants.NLTK_STOPWORDS_ENGLISH:
                    restricted_tokens.append(token)
            else:
                if token.isalpha() and len(token) > 2:
                    restricted_tokens.append(token)
        return restricted_tokens

    def _tokenize_text(self, report: bool, remove_stopwords=False):
        # Remove all newlines.
        self.df['text'] = self.df['text'].str.replace("\newline|\n", " ", regex=True)

        # Tokenize text using NLTK's recommended word tokenizer.
        self.df['tokenized_text'] = self.df['text'].apply(lambda text: word_tokenize(text))

        # Re-combine backslashes with the words they preceed in case backslashes are split by the algorithm as separate tokens.
        self.df['tokenized_text'] = self.df['tokenized_text'].apply(lambda tokens: self._mwe_retokenize(tokens, "\\"))

        # Remove tokens containing non-alphabetic characters.
        if remove_stopwords:
            self.df['cleaned_text'] = self.df['tokenized_text'].apply(lambda tokens: self._restrict_tokens(tokens, remove_stopwords=True))

        else:
            self.df['cleaned_text'] = self.df['tokenized_text'].apply(lambda tokens: self._restrict_tokens(tokens, remove_stopwords=False))

        # Plot histogram of document lengths.
        self.df['cleaned_text_length'] = self.df['cleaned_text'].str.len()
        if report:
            title = "Report Lengths After All Cleaning"
            filepath = os.path.join(self.path_to_output, 'hist_words_report_lengths_after_all_cleaning.png')
            xlabel = '''Length of Report (Words)

            Note: This figure is a histogram of the number of tokens in the sample reports after removing all non-alphabetic tokens,
            tokens less than three characters in length, and stopwords, just before the document term matrix for reports is created.
            Low frequency tokens have not yet been removed. 
            '''
        else:
            title = "Paper Lengths After All Cleaning"
            filepath = os.path.join(self.path_to_output, 'hist_words_paper_length_after_all_cleaning.png')
            xlabel = '''Length of Paper (Words)

            Note: This figure is a histogram of the number of tokens in the sample papers after the removal of the Journal of Public
            Economics cover pages, restriction to paper introductions, removal of thank yous, removal of non-alphabetic tokens,
            removal of tokens less than 3 characters in length, and removal of stopwords. Equivalently, this figure is built using
            paper lengths just before the document term matrix for papers is created. Low frequency tokens have not yet been removed.

            '''

        plot_histogram(x=self.df['cleaned_text_length'],
                       filepath=filepath,
                       title=title,
                       xlabel=xlabel)

        # Join into one string of space-delimited tokens.
        self.df['cleaned_text'] = self.df['cleaned_text'].str.join(" ").str.lower()

        # We no longer need the column giving length of cleaned text.
        self.df = self.df.drop(columns='cleaned_text_length')

    def _mwe_retokenize(self, tokens, token_of_interest):
        """Retokenizes a list of tokens by combining every token equal to 'token_of_interest' with the following token.
        """

        retokenized_list = []
        i = 0
        tokens_enum = enumerate(tokens)
        for i, current_token in tokens_enum:
            # print("CURRENT TOKEN: " + current_token + "| INDEX="+str(i))

            if current_token == token_of_interest and i < len(tokens) - 1:

                index = i
                next_token = tokens[i + 1]
                current_token_retokenized = current_token
                # print("\tCHECKING FOR TOKENS OF INTEREST IN THE IMMEDIATELY FOLLOWING TOKENS")
                while next_token == token_of_interest and index < len(tokens):
                    next(tokens_enum)

                if index < len(tokens) - 1:
                    # print("\tCONCATENATING WITH NEXT TOKEN, " + next_token + " WHICH IS NOT A TOKEN OF INTEREST")
                    current_token_retokenized = current_token_retokenized + next_token
                    current_token = next(tokens_enum)
                retokenized_list.append(current_token_retokenized)
            else:
                retokenized_list.append(current_token)

            # print("------------------------------------------------")

        return retokenized_list

    def _choose_optimal_format(self, df, file_type_ranking):
        """When applied to a groupby operation on a pandas DataFrame, choose the optimal file format
        as specified in the parameter "ranking."
        """

        for file_type in file_type_ranking:
            # For each possible file type in order of preference, check if a row with that file type exists.
            if len(df.loc[df['file_type'] == file_type]) > 0:
                # If it does, return it. Otherwise, we check for the next best file type.
                return df.loc[df['file_type'] == file_type]['filename'].iloc[0]
        # If none of the preferred file types exist in the 'file_type' column, return this message.
        return "This file does not exist in any of the specified file types."

    def _pickle_df(self, filename):

        # Momentarily save as an unpickled CSV.
        unpickled_path = os.path.join(self.path_to_intermediate_data, filename)
        self.df.to_csv(unpickled_path)

        # Pickle CSV and save it.
        pickled_path = os.path.join(self.path_to_intermediate_data, filename + ".pkl")
        pickle.dump(pkldir.encode(unpickled_path), open(pickled_path, "wb"))

        # Delete unpickled CSV file.
        os.remove(unpickled_path)


class PaperReader(JournalDocReader):
    """The PaperReader class implements the functionality needed to extract features from the sample papers.

    Args:
        JournalDocReader (JournalDocReader): The PaperReader inherits functionality from the JournalDocReader in cases where
                                             the same functionality needs to be applied to reports in addition to papers.
    """

    def build_df(self):
        """Builds a pandas DataFrame containing the text and ID of each paper."""

        self._decode_text()

        self._remove_jpub_e_cover_page()

        self._restrict_to_intro()

        self._remove_thank_yous()

        self._tokenize_text(remove_stopwords=True, report=False)

        self.df = self.df.drop(columns=['folder'])

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


class ReportReader(JournalDocReader):

    def __init__(self, path_to_pkl_documents: str, path_to_intermediate_data: str, path_to_output: str, path_to_referee_characteristics: str):
        JournalDocReader.__init__(self, path_to_pkl_documents, path_to_intermediate_data, path_to_output)
        self.path_to_referee_characteristics = path_to_referee_characteristics

    def build_df(self):
        """Builds a pandas DataFrame containing text of each report."""

        self._decode_text()

        self._tokenize_text(remove_stopwords=True, report=True)

        self.df = self.df.drop(columns=['folder'])

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

