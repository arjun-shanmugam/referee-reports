"""Defines the RefereeReportDataset class, which produces a report-level dataset.

    @author: Arjun Shanmugam
"""
import io
import os
import random
import warnings
from itertools import cycle
from typing import List

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import pkldir
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold


# from stargazer.stargazer import Stargazer
#
# from constants import OI_constants, OutputTableConstants
# from figure_utils import (plot_hist, plot_histogram, plot_pie, plot_pie_by,
#                           plot_scatter)
# from likelihood_ratio_model import LikelihoodRatioModel
# from ols_regression import OLSRegression
# from regularized_regression import RegularizedRegression


class RefereeReportDataset:
    """Builds a DataFrame at the report-level containing report,
    referee, and paper data. Allows the user to run statistical models for text analysis.
    """
    _df: pd.DataFrame
    _reports_df: pd.DataFrame
    _papers_df: pd.DataFrame
    _output_directory: str
    _seed: int
    _ngrams: int

    def __init__(self, cleaned_pickled_reports_file: str, cleaned_pickled_papers_file: str, output_directory: str, seed: int):
        """Instantiates a RefereeReportDataset object.

        Args:
            cleaned_pickled_reports_file (str): _description_
            cleaned_pickled_papers_file (str): _description_
            output_directory (str): _description_
            seed (int): _description_
        """
        # noinspection PyTypeChecker
        self._reports_df = pd.read_csv(io.StringIO(pkldir.decode(cleaned_pickled_reports_file).decode('utf-8')),
                                       index_col=['paper', 'refnum'])
        # noinspection PyTypeChecker
        self._papers_df = pd.read_csv(io.StringIO(pkldir.decode(cleaned_pickled_papers_file).decode('utf-8')),
                                      index_col='paper')
        self._output_directory = output_directory
        self._seed = seed
        self._df = pd.DataFrame(index=self._reports_df.index)
        np.random.seed(self._seed)  # Bad practice, but must set internal _seed to ensure reproducible output from sklearn.

        # self.report_dtm = pd.DataFrame()
        # self.report_vocabulary = []
        # self.tf_reports = pd.DataFrame()
        # self.tf_papers = pd.DataFrame()
        # self.tf_reports_adjusted = pd.DataFrame()
        # self.models = {}
        # self.ngrams = None

    def build_df(self, text_representation: str, ngrams: int, restrict_to_papers_with_mixed_gender_referees: bool, balance_sample_by_gender: bool):

        self._format_non_vocabulary_columns()
        # TODO: Restrict sample to mix-gendered referee groups if desired.
        if restrict_to_papers_with_mixed_gender_referees:
            self._restrict_to_mixed_gender_referees()
        self._build_dtm(text_representation, ngrams)
        self._merge_with_referee_characteristics()
        if balance_sample_by_gender:
            self._balance_sample_by_gender()

    def _format_non_vocabulary_columns(self):
        self._reports_df.columns = [f"_{column}_" for column in self._reports_df.columns]
        self._papers_df.columns = [f"_{column}_" for column in self._papers_df.columns]

    def _restrict_to_papers_with_mixed_gender_referees(self):  # TODO TODO TODO
        # Get the portion of each paper's referees who are female.
        refereeship_gender_breakdown = self._df.groupby(level=0)['_female_'].mean()

        # Restrict series to papers with at least one male referee and at least one female referee.
        refereeship_gender_breakdown = refereeship_gender_breakdown.loc[(refereeship_gender_breakdown != 0) & (refereeship_gender_breakdown != 1)]

        # Get the papers which meet these criteria.
        mixed_gender_papers = refereeship_gender_breakdown.index

        # Restrict the DataFrame to only those papers.
        self._df = self._df.loc[mixed_gender_papers, :]

    def _build_dtm(self, text_representation: str, ngrams: int):
        # Fit a CountVectorizer to the reports.
        vectorizer = CountVectorizer(ngram_range=(ngrams, ngrams))
        vectorizer = vectorizer.fit(self._reports_df['_cleaned_text_'])

        # Build the D.T.M. appropriately.
        if text_representation == 'R':
            dtm = vectorizer.transform(self._reports_df['_cleaned_text_']).toarray()
        elif text_representation == 'NR':
            dtm = vectorizer.transform(self._reports_df['_cleaned_text_']).toarray()
            report_lengths = np.sum(dtm, axis=1)
            dtm = dtm / report_lengths[:, np.newaxis]  # Divide each element in row i of dtm (2-D) by row i of report_lengths (1-D)

        elif text_representation == 'R-tilde':
            raise NotImplementedError("TODO: Implement R-tilde.")
        elif text_representation == 'NR/NP':
            raise NotImplementedError("TODO: Implement NR/NP.")
        else:
            raise ValueError("The argument text_representation to referee_report_dataset.build_df() must be either \"R\", \"NR\", \"R-tilde\", \"NR/NP\", ")

        # Concatenate DTM with the rest of the data.
        dtm = pd.DataFrame(dtm, self._reports_df.index, columns=vectorizer.get_feature_names_out())
        self._df = pd.concat([self._df, dtm], axis=1)

    def _merge_with_referee_characteristics(self):
        referee_characteristics = self._reports_df.drop(columns=['_raw_text_', '_cleaned_text_'])
        self._df = pd.concat([self._df, referee_characteristics], axis=1)

    def _balance_sample_by_gender(self):
        oversampled_gender = self._df['_female_'].value_counts().idxmax()  # Is 1 if females are oversampled, 0 if males are oversampled.
        imbalance = self._df['_female_'].value_counts().loc[oversampled_gender] - self._df['_female_'].value_counts().loc[1 - oversampled_gender]
        reports_of_oversampled_gender = self._df.loc[self._df['_female_'] == oversampled_gender, :].copy()
        papers_refereed_by_oversampled_gender = pd.Series(reports_of_oversampled_gender.index.get_level_values(0).value_counts().index)
        papers_refereed_by_oversampled_gender = papers_refereed_by_oversampled_gender.sample(frac=1, random_state=self._seed)  # Shuffle papers.

        num_papers_dropped = 0
        for paper in papers_refereed_by_oversampled_gender:
            reports_associated_with_current_paper = reports_of_oversampled_gender.loc[paper, :]
            # Get the number of reports associated with the current paper that were written by the oversampled gender.
            num_papers = len(reports_associated_with_current_paper)
            if num_papers == 1:  # If there is only one report written by the oversampled gender in the current paper group...
                continue  # ...do not drop this report.
            else:  # Otherwise, randomly select one paper and drop
                report_to_drop = reports_associated_with_current_paper.sample(n=1, random_state=self._seed).index.tolist()[0]

                self._df = self._df.drop(index=(paper, report_to_drop))
                num_papers_dropped += 1
            if num_papers_dropped == imbalance:
                break  # Break out of loop once we have dropped enough papers.

    def _validate_columns(self, y, X):
        if y not in self._df.columns:
            raise ValueError("The specified dependent variable is not a variable in the dataset.")
        elif not set(X).issubset(set(self._df.columns)):
            bad_variables = set(X) - set(self._df.columns)
            raise ValueError(f"Specified independent variable(s) {bad_variables} are not present in the dataset.")
        elif len(X) == 0:
            raise ValueError("You must specify at least one dependent variable.")

    def ols_regress(self, y: str, X: List[str], model_name: str, logistic: bool, log_transform: bool, standardize: bool):
        # TODO




# BELOW: OLD


    def ols_regress(self, y, X, model_name, add_constant, logistic, log_transform, standardize):
        # Check that specified independent, dependent variables are valid.
        self._validate_columns(y, X)

        # Select specified columns from DataFrame.
        dependent_variable = self._df[y]
        independent_variables = self._df[X]

        # Instantiate an OLS regression object.
        self.models[model_name] = OLSRegression(model_name=model_name,
                                                y=dependent_variable,
                                                X=independent_variables,
                                                log_transform=log_transform,
                                                standardize=standardize,
                                                add_constant=add_constant,
                                                logistic=logistic)
        self.models[model_name].fit()

    def regularized_regress(self,
                            y,
                            X,
                            model_name,
                            method,
                            logistic,
                            log_transform,
                            standardize,
                            alphas,
                            cv_folds,
                            stratify,
                            add_constant,
                            sklearn=False,
                            l1_ratios=None,
                            adjust_alpha=False):
        # Check that specified independent, dependent variables are valid.
        self._validate_columns(y, X)

        # Select specified columns from DataFrame.
        dependent_variable = self._df[y]
        independent_variables = self._df[X]

        # Instantiate a RegularizedRegression object.
        self.models[model_name] = RegularizedRegression(model_name,
                                                        dependent_variable,
                                                        independent_variables,
                                                        logistic,
                                                        add_constant,
                                                        standardize,
                                                        log_transform,
                                                        alphas,
                                                        method,
                                                        cv_folds,
                                                        stratify,
                                                        self._seed,
                                                        l1_ratios,
                                                        self.path_to_output,
                                                        adjust_alpha)
        if sklearn:
            self.models[model_name].fit_cross_validated_sklearn()
        else:
            self.models[model_name].fit_cross_validated()

    def resample_variable_binomial(self, variable: str, p: float, ensure_balanced: bool):
        # Check that the requested column to resample is actually a column in the dataset.
        self._validate_columns(y=variable, X=[], check_length=False)

        np.random.seed(self._seed)

        female_indices = np.random.choice(np.array(np.arange(start=0, stop=len(self._df), step=1)),
                                          size=int(0.5 * len(self._df)),
                                          replace=False)
        non_female_indices = self._df.index.difference(female_indices)
        self._df.loc[female_indices, variable] = 1
        self._df.loc[non_female_indices, variable] = 0

    def _produce_summary_statistics(self,
                                    adjust_reports_with_papers: bool,
                                    normalize_documents_by_length: bool
                                    ):
        # Pie chart: number of papers with mixed-gender refereeship vs. all female refereeship vs. all male refereeship
        mean_female_by_paper = (self._df.groupby(by='_paper_', observed=True)
        .mean()['_female_']  # Get mean of female within each group.
        )
        mean_female_by_paper[(mean_female_by_paper != 0) & (mean_female_by_paper != 1)] = "Refereed by Females and Non-Females"
        mean_female_by_paper = mean_female_by_paper.replace({0: "Refereed by Non-Females Only", 1: "Refereed by Females Only"})

        mean_female_by_paper_grouped = mean_female_by_paper.groupby(mean_female_by_paper).count()  # Get counts of each unique value        
        plot_pie(x=mean_female_by_paper_grouped,
                 filepath=os.path.join(self.path_to_output, 'pie_referee_genders_by_paper.png'),
                 title="Gender Breakdowns of Paper Referees")

        # Pie chart: number of referees on each paper.
        referees_per_paper = (self._df.groupby(by='_paper_', observed=True)
                              .count()
                              .mean(axis=1)  # Every column in this DataFrame is identical, so mean across axes to reduce.
                              .astype(int)
                              .replace({1: "One Referee",
                                        2: "Two Referees",
                                        3: "Three Referees",
                                        4: "Four Referees"})
                              )
        referees_per_paper_grouped = referees_per_paper.groupby(referees_per_paper).count()  # Get counts of each unique value.
        plot_pie(x=referees_per_paper_grouped,
                 filepath=os.path.join(self.path_to_output, 'pie_num_referees_per_paper.png'),
                 title="Number of Referees Per Paper")

        # Histogram: Skewness of the word count variables.
        if adjust_reports_with_papers:
            title = "Skewness of Paper-Adjusted Wordcount Variables (" + str(self.ngrams) + "-grams)"
            filepath = os.path.join(self.path_to_output, "hist_paper_adjusted_" + str(self.ngrams) + "_grams_counts_skewness.png")
        else:
            title = "Skewness of Raw Wordcount Variables (" + str(self.ngrams) + "-grams)"
            filepath = os.path.join(self.path_to_output, "hist_raw_" + str(self.ngrams) + "_grams_counts_skewness.png")
        skewness = self._df[self.report_vocabulary].skew(axis=0)
        plot_histogram(x=skewness,
                       filepath=filepath,
                       title=title,
                       xlabel="""Skewness of Variable
                    
                    This figure calculates 
                    
                    """)

        # Histogram: Skewness of log(x+1)-transformed word count variables.
        if adjust_reports_with_papers:
            title = "Skewness of ln(x+1)-Transformed Paper-Adjusted Wordcount Variables (" + str(self.ngrams) + "-grams)"
            filepath = os.path.join(self.path_to_output, "hist_paper_adjusted_" + str(self.ngrams) + "_grams_counts_skewness_log_transformed.png")
        else:
            title = "Skewness of ln(x+1)-Transformed Raw Wordcount Variables (" + str(self.ngrams) + "-grams)"
            filepath = os.path.join(self.path_to_output, "hist_raw_" + str(self.ngrams) + "_grams_counts_skewness_log_transformed.png")
        skewness = np.log(self._df[self.report_vocabulary] + 1).skew(axis=0)
        plot_histogram(x=skewness,
                       filepath=filepath,
                       title=title,
                       xlabel="Skewness of Variable")

        # Pie chart: Breakdown of decision and recommendation
        decision_breakdown = self._df['_decision_'].value_counts()
        plot_pie(x=decision_breakdown,
                 filepath=os.path.join(self.path_to_output, 'pie_decision.png'),
                 title="Breakdown of Referee Decisions")

        recommendation_breakdown = self._df['_recommendation_'].value_counts()
        plot_pie(x=recommendation_breakdown,
                 filepath=os.path.join(self.path_to_output, 'pie_recomendation.png'),
                 title="Breakdown of Referee Recommendation")

        # Pie chart: Breakdown of referee decision and recommendation by gender.
        plot_pie_by(x=self._df['_decision_'],
                    by=self._df['_female_'],
                    filepath=os.path.join(self.path_to_output, "pie_decision_by_female.png"),
                    title="Breakdown of Referee Decision by Referee Gender",
                    by_value_labels={0: "Non-Female", 1: "Female"})

        plot_pie_by(x=self._df['_recommendation_'],
                    by=self._df['_female_'],
                    filepath=os.path.join(self.path_to_output, "pie_recommendation_by_female.png"),
                    title="Breakdown of Referee Recommendation by Referee Gender",
                    by_value_labels={0: "Non-Female", 1: "Female"})

        # Pie chart: Breakdown of the number and gender of referees for each paper.
        referee_counts_each_gender = (self._df.groupby(by=['_paper_', '_female_'], observed=True)
                                      .count()  # Get number of referees of each gender for each paper.
                                      .mean(axis=1)  # Counts are identical across columns, so take mean across columns to reduce.
                                      .unstack()  # Reshape from long to wide.
                                      .value_counts()  # Get counts of each permutation.
                                      )
        # Assign new index so that helper function can generate a labeled pie chart.
        referee_counts_each_gender = pd.Series(referee_counts_each_gender.values,
                                               index=['One Non-Female, One Female', 'Two Non-Female, One Female', 'One Non-Female, Two Female'])
        plot_pie(x=referee_counts_each_gender,
                 filepath=os.path.join(self.path_to_output, "pie_referee_gender_number_permutations.png"),
                 title="Breakdown of Paper Referees and their Genders")

        if not normalize_documents_by_length:
            # Histogram: Number of tokens in reports.
            tokens_per_report = self.tf_reports[self.report_vocabulary].sum(axis=1)
            xlabel = '''Length (''' + str(self.ngrams) + '''-Gram Tokens)
        
            Note: This figure is a histogram of the number of tokens in the sample reports after all cleaning, 
            sample restriction, vectorization, and removal of low-frequency tokens. Equivalently, this figure
            gives the distribution of report lengths in tokens immediately before analysis.
                    '''
            plot_histogram(x=tokens_per_report,
                           filepath=os.path.join(self.path_to_output,
                                                 "hist_tokens_per_report_immediately_before_analysis.png"),
                           title="Number of Tokens in Each Report (" + str(self.ngrams) + "-Grams)",
                           xlabel=xlabel)

            # Histogram: Number of tokens in papers.
            tokens_per_paper = self.tf_papers[self.report_vocabulary].sum(axis=1)
            xlabel = '''Length (''' + str(self.ngrams) + '''-Gram Tokens)

            Note: This figure is a histogram of the number of tokens in the sample papers after cleaning, 
            sample restriction, restriction to introductions, removal of thank yous, vectorization, and
            removal of low-frequency tokens. Equivalently, this figure gives the distribution of paper
            lengths in tokens immediately before analysis. Note that this figure only counts tokens 
            which also appear somewhere in the corpus of reports.
            '''
            plot_histogram(x=tokens_per_paper,
                           filepath=os.path.join(self.path_to_output,
                                                 "hist_tokens_per_paper_immediately_before_analysis.png"),
                           title="Number of Tokens in Each Paper (" + str(self.ngrams) + "-Grams)",
                           xlabel=xlabel)

        # """

        # # Histogram: Counts and log counts of token in the paper-adjusted D.T.M. for the reports.
        # if adjust_reports_with_papers:
        #     if normalize_documents_by_length:
        #         # Histogram: Frequencies in the paper-adjusted D.T.M. for the reports.
        #         token_counts = self.tf_reports_adjusted[self.report_vocabulary].to_numpy().flatten()
        #         xlabel = """Values in Paper-Adjusted Matrix

        #             Note: This figure is a histogram of the values in the normalized
        #             paper-adjusted reports matrix. To calculate this figure, I first
        #             add one to every cell of the reports D.T.M. and the papers D.T.M.
        #             Then, I normalize each cell of the reports D.T.M. by the sum of
        #             the row containing that cell. I normalize each cell of the papers
        #             D.T.M. in the same way. Then, each row (i, j) of the normalized
        #             reports D.T.M. is divided element-wise by row j of the
        #             normalized papers D.T.M.
        #             """
        #         plot_histogram(x=pd.Series(token_counts),
        #                     filepath=os.path.join(self.output_directory, "hist_paper_adjusted_matrix_immediately_before_analysis.png"),
        #                     title="Distribution of Paper-Adjusted Token Frequencies in Reports (" + str(self.ngrams) + "-Grams)",
        #                     xlabel=xlabel)
        #         # Histogram: Log frequencies in the paper-adjusted D.T.M. for the reports.
        #         token_counts = np.log(self.tf_reports_adjusted[self.report_vocabulary] + 1).to_numpy().flatten()
        #         xlabel = """Values in Paper-Adjusted Matrix

        #                     Note: This figure is a histogram of the values in the normalized
        #                     paper-adjusted reports matrix. To calculate this figure, I first
        #                     add one to every cell of the reports D.T.M. and the papers D.T.M.
        #                     Then, I normalize each cell of the reports D.T.M. by the sum of
        #                     the row containing that cell. I normalize each cell of the papers
        #                     D.T.M. in the same way. Then, each row (i, j) of the normalized
        #                     reports D.T.M. is divided element-wise by row j of the normalized
        #                     papers D.T.M. Lastly, I add 1 to each cell of this matrix and then
        #                     take the base ten log of each cell.
        #                     """
        #         plot_histogram(x=pd.Series(token_counts),
        #                     filepath=os.path.join(self.output_directory, "hist_log_paper_adjusted_matrix_immediately_before_analysis.png"),
        #                     title="Distribution of Log-Transformed Paper-Adjusted Token Frequencies in Reports (" + str(self.ngrams) + "-Grams)",
        #                     xlabel=xlabel)
        #     else:
        #         # Histogram: Counts in the paper-adjusted D.T.M. for the reports.
        #         token_counts = self.tf_reports_adjusted[self.report_vocabulary].to_numpy().flatten()
        #         xlabel = """Values in Paper-Adjusted Matrix

        #                     Note: This figure is a histogram of the values in the paper-adjusted
        #                     reports D.T.M. To produce the paper-adjusted reports D.T.M., I
        #                     subtract from each row (i, j) of the reports D.T.M. row j of the
        #                     papers D.T.M.
        #                     """
        #         plot_histogram(x=pd.Series(token_counts),
        #                     filepath=os.path.join(self.output_directory, "hist_paper_adjusted_matrix_immediately_before_analysis.png"),
        #                     title="Distribution of Paper-Adjusted Token Counts in Reports (" + str(self.ngrams) + "-Grams)",
        #                     xlabel=xlabel)
        #         # Histogram: Log counts in the paper-adjusted D.T.M. for the reports.
        #         token_counts = np.log(self.tf_reports_adjusted[self.report_vocabulary] + 1).to_numpy().flatten()
        #         xlabel = """Values in Paper-Adjusted Matrix

        #                     Note: This figure is a histogram of the values in the paper-adjusted
        #                     reports D.T.M. To produce the paper-adjusted reports D.T.M., I
        #                     subtract from each row (i, j) of the reports D.T.M. row j of the
        #                     papers D.T.M. Then, I add 1 to each cell of this D.T.M. and then
        #                     take the base ten log of each cell.
        #                     """
        #         plot_histogram(x=pd.Series(token_counts),
        #                     filepath=os.path.join(self.output_directory, "hist_log_paper_adjusted_matrix_immediately_before_analysis.png"),
        #                     title="Distribution of Log Paper-Adjusted Token Counts in Reports (" + str(self.ngrams) + "-Grams)",
        #                     xlabel=xlabel)

        # # Histogram: Counts of each token in the reports.
        # token_counts = self.tf_reports[self.report_vocabulary].sum(axis=0)
        # plot_histogram(x=token_counts,
        #                filepath=os.path.join(self.output_directory,
        #                                      "hist_token_counts_immediately_before_analysis_reports.png"),
        #                title="Distribution of Total Appearances Across Tokens in Reports (" + str(self.ngrams) + "-Grams)",
        #                xlabel='''Number of Times Token Appears

        #                         This figure plots the distribution of tokens' total apperances across reports.
        #                         To produce this figure, I first calculate the total number of times each token appears
        #                         across all reports. I then bin tokens according to those sums and display the results
        #                         above.
        #                ''')

        # # Histogram: Counts of each token in the papers.
        # token_counts = self.tf_papers[self.report_vocabulary].sum(axis=0)
        # plot_histogram(x=token_counts,
        #                filepath=os.path.join(self.output_directory,
        #                                      "hist_token_counts_immediately_before_analysis_papers.png"),
        #                title="Distribution of Total Apperances Across Tokens in Papers (" + str(self.ngrams) + "-Grams)",
        #                xlabel='''Number of Times Token Appears

        #                This figure plots the distribution of tokens' total appearances across papers.
        #                To produce this figure, I first calculate the total number of times each token appears
        #                across all papers. I then bin tokens according to those sums and display the results above.
        #                ''')

        # # Histogram: Counts of each token in the reports, log(x+1) transformed.
        # token_counts = np.log(self.tf_reports[self.report_vocabulary] + 1).sum(axis=0) 
        # plot_histogram(x=token_counts,
        #                filepath=os.path.join(self.output_directory,
        #                                      "hist_token_log_counts_immediately_before_analysis_reports.png"),
        #                title="Distribution of log(x+1)-Transformed Counts Across Reports (" + str(self.ngrams) + "-Grams)",
        #                xlabel='''Sum of log(x+1)-Transformed Counts

        #                This figure plots the distribution of tokens' log(x+1)-transformed counts,
        #                summed across reports. To produce this figure, I add 1 to each cell of the 
        #                D.T.M. for the reports. Then, I take the natural log of each cell. Lastly,
        #                I sum the D.T.M. across reports (rows) and bin the tokens according to the
        #                resulting sums.
        #                ''')

        # # Histogram: Counts of each token in the papers, log(x+1) transformed.
        # token_counts = np.log(self.tf_papers[self.report_vocabulary] + 1).sum(axis=0)
        # plot_histogram(x=token_counts,
        #                filepath=os.path.join(self.output_directory,
        #                                      "hist_token_log_counts_immediately_before_analysis_papers.png"),
        #                title="Distribution of log(x+1)-Transformed Counts Across Papers (" + str(self.ngrams) + "-Grams)",
        #                xlabel='''Sum of log(x+1)-Transformed Counts

        #                This figure plots the distribution of tokens' log(x+1)-transformed counts, 
        #                summed across papers. To produce this figure, I add 1 to each cell of the 
        #                D.T.M. for the reports. Then, I take the natural log of each cell. Lastly,
        #                I sum the D.T.M. across papers (rows) and bin the tokens according to the
        #                resulting sums.
        #                ''')

        # """

        # Histogram: Number of reports in which tokens appear.
        num_reports_where_word_appears = (self.tf_reports
                                          .mask(self.tf_reports > 0, 1)
                                          .sum(axis=0)
                                          .transpose()
                                          )

        xlabel = """Number of Reports

        This figure plots the distribution over the number of reports in which each token appears.
        It is produced after all cleaning but before the removal of low-frequency tokens. To produce this figure,
        I calculate the number of reports in which each token appears at least once. I sort tokens into bins
        based on those values to produce this figure. """
        plot_histogram(x=num_reports_where_word_appears,
                       title="Number of Reports in Which Tokens Appear (" + str(self.ngrams) + "-grams)",
                       xlabel=xlabel,
                       filepath=os.path.join(self.path_to_output, 'hist_num_reports_where_tokens_appear_' + str(self.ngrams) + '_grams.png'))

        # Hist: Average length of male vs. female reports immediately before analysis.
        report_lengths = (
            self.tf_reports
            .sum(axis=1)
        )
        male_report_lengths = report_lengths.loc[self._df['_female_'] == 0]
        male_report_lengths_mean = male_report_lengths.mean().round(3)
        male_report_lengths_se = male_report_lengths.sem().round(3)
        female_report_lengths = report_lengths.loc[self._df['_female_'] == 1]
        female_report_lengths_mean = female_report_lengths.mean().round(3)
        female_report_lengths_se = female_report_lengths.sem().round(3)

        fig, ax = plt.subplots(1, 1)
        plot_hist(ax=ax,
                  x=male_report_lengths,
                  title="Distribution of Male-Written vs. Female-Written Report Lengths",
                  xlabel='Report Length',
                  ylabel='Normalized Frequency',
                  alpha=0.5,
                  color=OI_constants.male_color.value,
                  summary_statistics_linecolors=OI_constants.male_color.value,
                  label="Male Reports \n Mean length: " + str(male_report_lengths_mean) + " (SE: " + str(male_report_lengths_se) + ")",
                  summary_statistics=['median'])
        plot_hist(ax=ax,
                  x=female_report_lengths,
                  title="Distribution of Male-Written vs. Female-Written Report Lengths",
                  xlabel='Report Length',
                  ylabel='Normalized Frequency',
                  alpha=0.5,
                  color=OI_constants.female_color.value,
                  summary_statistics_linecolors=OI_constants.female_color.value,
                  label="Female Reports \n Mean length: " + str(female_report_lengths_mean) + " (SE: " + str(female_report_lengths_se) + ")",
                  summary_statistics=['median'])
        ax.legend(loc='best', fontsize='x-small')
        plt.savefig(os.path.join(self.path_to_output, "hist_male_vs_female_report_lengths_" + str(self.ngrams) + "_grams.png"), bbox_inches='tight')
        plt.close(fig)

        # Table: Most common words for men and women (R).
        occurrences_by_gender = (
            pd.concat([self.tf_reports, self._df['_female_']], axis=1)
            .groupby(by='_female_')
            .sum()
            .transpose()
        )

        male_occurrences = pd.Series(occurrences_by_gender[0].sort_values(ascending=False),
                                     name="Most Common Words in Male-Written Reports").reset_index().iloc[:50]
        male_occurrences = (male_occurrences['index'] + ": " + male_occurrences["Most Common Words in Male-Written Reports"].astype(str)).rename(
            "\textbf{Most Common Words in Male-Written Reports}")
        female_occurrences = pd.Series(occurrences_by_gender[1].sort_values(ascending=False),
                                       name="Most Common Words in Female-Written Reports").reset_index().iloc[:50]
        female_occurrences = (female_occurrences['index'] + ": " + female_occurrences["Most Common Words in Female-Written Reports"].astype(str)).rename(
            "\textbf{Most Common Words in Female-Written Reports}")
        pd.concat([male_occurrences, female_occurrences], axis=1).to_latex(
            os.path.join(self.path_to_output, "table_most_common_words_by_gender_R_" + str(self.ngrams) + "_grams.tex"),
            index=False,
            escape=False,
            float_format="%.3f")

        # Table: Most common words for men and women (NR).
        report_lengths = self.tf_reports.sum(axis=1)  # Calculate NR_ij.      
        occurrences_by_gender = (
            pd.concat([self.tf_reports.div(report_lengths, axis=0), self._df['_female_']], axis=1)
            .groupby(by='_female_')
            .sum()
            .transpose()
        )
        male_occurrences = pd.Series(occurrences_by_gender[0].sort_values(ascending=False),
                                     name="Most Common Words in Male-Written Reports").reset_index().iloc[:50].round(3)
        male_occurrences = (male_occurrences['index'] + ": " + male_occurrences["Most Common Words in Male-Written Reports"].astype(str)).rename(
            "\textbf{Most Common Words in Male-Written Reports}")
        female_occurrences = pd.Series(occurrences_by_gender[1].sort_values(ascending=False),
                                       name="Most Common Words in Female-Written Reports").reset_index().iloc[:50].round(3)
        female_occurrences = (female_occurrences['index'] + ": " + female_occurrences["Most Common Words in Female-Written Reports"].astype(str)).rename(
            "\textbf{Most Common Words in Female-Written Reports}")
        pd.concat([male_occurrences, female_occurrences], axis=1).to_latex(
            os.path.join(self.path_to_output, "table_most_common_words_by_gender_NR_" + str(self.ngrams) + "_grams.tex"),
            index=False,
            escape=False,
            float_format="%.3f")

        # Hist: Cosine similarity between NR vectors and NP vectors, separately for males and females.
        report_lengths = self.tf_reports.sum(axis=1)  # Calculate NR_ij.      
        NR = self.tf_reports.div(report_lengths, axis=0)
        paper_lengths = self.tf_papers.sum(axis=1)  # Calculate NP_j
        NP = self.tf_papers.div(paper_lengths, axis=0)
        index = pd.MultiIndex.from_frame(self._df[['_paper_', '_refnum_']], names=['_paper_', '_refnum_'])
        columns = self._df['_paper_']
        cosine_similarities = pd.DataFrame(cosine_similarity(NR, NP), index=index, columns=columns)
        cosine_similarities = pd.Series(np.diag(cosine_similarities), index=cosine_similarities.index, name='similarity')
        genders = self._df[['_female_', '_paper_', '_refnum_']].set_index(['_paper_', '_refnum_'])
        cosine_similarities_and_genders = pd.concat([cosine_similarities, genders], axis=1)
        cosine_similarities_males = cosine_similarities_and_genders.loc[cosine_similarities_and_genders['_female_'] == 0, 'similarity']
        cosine_similarities_females = cosine_similarities_and_genders.loc[cosine_similarities_and_genders['_female_'] == 1, 'similarity']
        fig, ax = plt.subplots(1, 1)

        male_mean = cosine_similarities_males.mean().round(3)
        male_se = cosine_similarities_males.sem().round(3)
        plot_hist(ax=ax,
                  x=cosine_similarities_males,
                  title="Distribution of Cosine Similarity Between Reports and Their Associated Papers",
                  xlabel='Cosine Similarity',
                  ylabel='Normalized Frequency',
                  alpha=0.5,
                  color=OI_constants.male_color.value,
                  summary_statistics_linecolors=OI_constants.male_color.value,
                  label="Male Reports \n Mean Cosine Similarity: " + str(male_mean) + " (SE: " + str(male_se) + ")",
                  summary_statistics=['median'])

        female_mean = cosine_similarities_females.mean().round(3)
        female_se = cosine_similarities_females.sem().round(3)
        plot_hist(ax=ax,
                  x=cosine_similarities_females,
                  title="Distribution of Cosine Similarity Between Reports and Their Associated Papers",
                  xlabel='Cosine Similarity',
                  ylabel='Normalized Frequency',
                  alpha=0.5,
                  color=OI_constants.female_color.value,
                  summary_statistics_linecolors=OI_constants.female_color.value,
                  label="Female Reports \n Mean Cosine Similarity: " + str(female_mean) + " (SE: " + str(female_se) + ")",
                  summary_statistics=['median'])
        ax.legend(loc='best', fontsize='x-small')
        plt.savefig(os.path.join(self.path_to_output, "hist_male_vs_female_cosine_similarity_NR_NP_" + str(self.ngrams) + "_grams.png"), bbox_inches='tight')
        plt.close(fig)

    def get_vocabulary(self):
        return self.report_vocabulary

    def build_regularized_results_table(self, model_name, num_coefs_to_report=30):
        self._validate_model_request(model_name, expected_model_type="Regularized")

        # Store results table.
        results_table = self.models[model_name].get_results_table()

        # Validate requested number of top/bottom coefficients.
        if len(results_table) < num_coefs_to_report * 2:
            raise ValueError("There are fewer than " + str(num_coefs_to_report * 2) + " coefficients in the results table.")

        # Store metrics in a named column.
        metrics = pd.Series(results_table.loc[OutputTableConstants.REGULARIZED_REGRESSION_METRICS.value], name="Metrics")
        if not self.models[model_name]._add_constant:
            metrics = metrics.drop(labels="Constant")
        if not self.models[model_name].adjust_alpha:
            metrics = metrics.drop(labels="$\\alpha^{*}_{adjusted}$")
            metrics = metrics.drop("$\\bar{p}_{\\alpha^{*}_{adjusted}}$")
        if self.models[model_name].method != 'elasticnet':
            metrics = metrics.drop(labels="Optimal L1 Penalty Weight")
        metrics = metrics.reset_index()
        metrics = metrics['index'] + ": " + metrics['Metrics'].round(3).astype(str)

        # Store specified dummy variables in a named column.
        dummy_coefficients = pd.Series(results_table.loc[self.models[model_name]._dummy_variables], name="Dummy Variables").reset_index()
        dummy_coefficients = dummy_coefficients['index'] + ": " + dummy_coefficients['Dummy Variables'].round(3).astype(str)

        # Drop non-coefficients from the Series of coefficients and sort coefficients by value.
        non_text_coefs = self.models[model_name]._dummy_variables + OutputTableConstants.REGULARIZED_REGRESSION_METRICS.value
        coefficients_sorted = results_table.drop(labels=non_text_coefs).sort_values(ascending=False)

        # Get largest nonzero coefficients; concatenate them with the associated token.
        top_coefficients = pd.Series(coefficients_sorted.iloc[:num_coefs_to_report], name="Largest " + str(num_coefs_to_report) + " Coefficients").reset_index()
        top_coefficients = top_coefficients[top_coefficients["Largest " + str(num_coefs_to_report) + " Coefficients"] != 0]
        top_coefficients = top_coefficients['index'] + ": " + top_coefficients["Largest " + str(num_coefs_to_report) + " Coefficients"].round(3).astype(str)

        # Get smallest nonzero coefficients; concatenate them with the associated token.
        bottom_coefficients = pd.Series(coefficients_sorted.iloc[-num_coefs_to_report:],
                                        name="Smallest " + str(num_coefs_to_report) + " Coefficients").reset_index()
        bottom_coefficients = bottom_coefficients[bottom_coefficients["Smallest " + str(num_coefs_to_report) + " Coefficients"] != 0]
        bottom_coefficients = bottom_coefficients['index'] + ": " + bottom_coefficients["Smallest " + str(num_coefs_to_report) + " Coefficients"].round(
            3).astype(str)
        bottom_coefficients = bottom_coefficients.iloc[::-1].reset_index(drop=True)  # Flip order.

        results_table = pd.concat([top_coefficients, bottom_coefficients, dummy_coefficients, metrics], axis=1)
        results_table = results_table.fillna(' ')
        results_table.columns = ['Words Most Predictive of Female Referee',
                                 'Words Most Predictive of Non-Female Referee',
                                 'Dummy Variables',
                                 'Other Metrics']
        if len(dummy_coefficients) == 0:
            results_table = results_table.drop(columns='Dummy Variables')
        results_table.columns = ['\textbf{' + column + '}' for column in results_table.columns]
        results_table.to_latex(os.path.join(self.path_to_output, model_name + "_results.tex"),
                               index=False,
                               escape=False,
                               float_format="%.3f")

    def build_ols_results_table(self,
                                filename: str,
                                requested_models: List[str],
                                title: str = None,
                                show_confidence_intervals=False,
                                dependent_variable_name=None,
                                column_titles=None,
                                show_degrees_of_freedom=False,
                                rename_covariates=None,
                                covariate_order=None,
                                lines_to_add=None,
                                custom_notes=None):
        # Validate model names.
        for model_name in requested_models:
            if not model_name in self.models:
                raise ValueError("A model by that name has not been estimated.")

        # Validate model types.
        for model_name in requested_models:
            if self.models[model_name].get_model_type() != "OLS":
                raise TypeError("This function may only be used to produce output for regularized models.")

        # Grab results tables.
        results = []
        for model_name in requested_models:
            self._validate_model_request(model_name, expected_model_type="OLS")
            current_result = self.models[model_name].get_results_table()
            results.append(current_result)

        # Build table with Stargazer.
        stargazer = Stargazer(results)

        # Edit table.
        if title is not None:
            stargazer.title(title)
        stargazer.show_confidence_intervals(show_confidence_intervals)
        if dependent_variable_name is not None:
            stargazer.dependent_variable_name(dependent_variable_name)
        stargazer.show_degrees_of_freedom(show_degrees_of_freedom)
        if rename_covariates is not None:
            stargazer.rename_covariates(rename_covariates)
        if covariate_order is not None:
            stargazer.covariate_order(covariate_order)
        if lines_to_add is not None:
            for line in lines_to_add:
                stargazer.add_line(line[0], line[1], line[2])
        if custom_notes is not None:
            stargazer.add_custom_notes(custom_notes)
        if column_titles is not None:
            stargazer.custom_columns(column_titles[0], column_titles[1])

        # Write LaTeX.
        latex = stargazer.render_latex()

        # Write to file.
        with open(os.path.join(self.path_to_output, filename + ".tex"), "w") as output_file:
            output_file.write(latex)



    def add_column(self, column):
        if not isinstance(column, pd.Series):
            error_msg = "The passed column is not a pandas Series."
            raise ValueError(error_msg)
        elif len(column) != len(self._df):
            error_msg = "The specified column must contain the same number of rows as the existing dataset."
            raise ValueError(error_msg)
        elif column.index.tolist() != self._df.index.tolist():
            error_msg = "The specified column must have an idential pandas Index compared to the existing dataset."
            raise ValueError(error_msg)
        elif column.name in set(self._df.columns.tolist()):
            error_msg = "The specified column's name must not be identical to an existing column in the dataset."
            raise ValueError(error_msg)
        else:
            self._df = pd.concat([self._df, column], axis=1)

    def calculate_likelihood_ratios(self, model_name: str, model_type: str):
        self.models[model_name] = LikelihoodRatioModel(dtm=self._df[self.report_vocabulary],
                                                       document_classification_variable=self._df['_female_'],
                                                       fe_variable=self._df['_paper_'],
                                                       model_type=model_type)
        self.models[model_name].fit()

    def _validate_model_request(self, model_name, expected_model_type):
        # Validate model name.
        if not model_name in self.models:
            raise ValueError("A model by that name has not been estimated.")

        # Validate model type.
        if expected_model_type not in self.models[model_name].get_model_type():
            raise TypeError("This function may only be used to produce output for " + expected_model_type + " models.")

    def build_likelihood_results_table(self, model_name, num_ratios_to_report=40, decimal_places=3):
        self._validate_model_request(model_name, "Likelihood Ratio")

        # Store results table.
        results_table = self.models[model_name].get_results_table()

        # Build results table for pooled estimates.=================================================
        pooled_columns = ["pooled_ratios",
                          "frequency_over_documents",
                          "frequency_over_fe_groups"]
        # Round likelihood ratios and word frequencies over reports and papers.
        results_table.loc[:, "pooled_ratios"] = results_table["pooled_ratios"].round(decimal_places)
        sorted_pooled_ratios = (results_table[pooled_columns]
                                .sort_values(by='pooled_ratios')  # Sort likelihood ratios. 
                                .reset_index()  # Reset index, containing the words corresponding to each ratio, and add it as a column.
                                )
        # Get highest pooled likelihood ratios.
        highest_pooled_ratios = sorted_pooled_ratios.iloc[-num_ratios_to_report:]
        # Concatenate words with their corresponding likelihood ratios.
        highest_pooled_ratios.loc[:, 'Highest Pooled Ratios'] = (highest_pooled_ratios['index'] +
                                                                 ": " +
                                                                 highest_pooled_ratios['pooled_ratios'].astype(str)
                                                                 )
        # Get lowest likelihood ratios.
        lowest_pooled_ratios = sorted_pooled_ratios.iloc[:num_ratios_to_report]
        # Concatenate words with their corresponding likelihood ratios.
        lowest_pooled_ratios.loc[:, 'Lowest Pooled Ratios'] = (lowest_pooled_ratios['index'] +
                                                               ": " +
                                                               lowest_pooled_ratios['pooled_ratios'].astype(str)
                                                               )

        # Build results table for within-paper estimates.===========================================
        fe_columns = ["fe_ratios",
                      "frequency_over_documents",
                      "frequency_over_fe_groups"]
        # Round likelihood ratios.
        results_table.loc[:, 'fe_ratios'] = results_table['fe_ratios'].round(decimal_places)
        sorted_fe_ratios = (results_table[fe_columns]
                            .sort_values(by='fe_ratios')  # Sort likelihood ratios.
                            .reset_index()  # Reset index, containing the words corresponding to each ratio, and add it as a column.
                            )
        # Get highest likelihood ratios.
        highest_fe_ratios = sorted_fe_ratios.iloc[-num_ratios_to_report:]
        # Concatenate words with their corresponding likelihood ratios.
        highest_fe_ratios.loc[:, "Highest Within-Paper Ratios"] = (highest_fe_ratios['index'] +
                                                                   ": " +
                                                                   highest_fe_ratios['fe_ratios'].astype(str)
                                                                   )
        # Get lowest likelihood ratios.
        lowest_fe_ratios = sorted_fe_ratios.iloc[:num_ratios_to_report]
        # Concatenate words with their corresponding likelihood ratios.
        lowest_fe_ratios.loc[:, "Lowest Within-Paper Ratios"] = (lowest_fe_ratios['index'] +
                                                                 ": " +
                                                                 lowest_fe_ratios["fe_ratios"].astype(str)
                                                                 )

        # Plot likelihood ratios by the frequency of the associated tokens.
        extreme_pooled_ratios = pd.concat([highest_pooled_ratios['pooled_ratios'],
                                           lowest_pooled_ratios['pooled_ratios']],
                                          axis=0)
        extreme_fe_ratios = pd.concat([highest_fe_ratios['fe_ratios'],
                                       lowest_fe_ratios['fe_ratios']],
                                      axis=0)
        pooled_ratios_frequencies = pd.concat([highest_pooled_ratios['frequency_over_documents'],
                                               lowest_pooled_ratios['frequency_over_documents']],
                                              axis=0)
        fe_ratios_frequencies = pd.concat([highest_fe_ratios['frequency_over_documents'],
                                           lowest_fe_ratios['frequency_over_documents']],
                                          axis=0)
        fig, axs = plt.subplots(2, 1, sharex='col')
        y_list = [extreme_pooled_ratios,
                  extreme_fe_ratios]
        x_list = [pooled_ratios_frequencies,
                  fe_ratios_frequencies]
        xlabels = ["",
                   """Portion of Documents in Which Token Appears

                   This figure plots likelihood ratios by the frequency of the associated tokens across paper groups, separately for
                   the most extreme pooled sample and within-paper likelihood ratios. A \'paper group\' is defined as a group of
                   reports associated with a single paper. To produce this figure, pooled sample and within-paper likelihood ratios 
                   are first calculated for every token. These likelihood ratios become the _y_data-values of the points displayed in this graph.
                   Then, for each of the most extreme likelihood ratios, I calculate the portion of paper groups in which the associated
                   token appears at least once. Those values form the x-values of the points displayed in this graph. Note that both
                   plots share the same x-axis. 
                   """]
        ylabels = ["Likelihood Ratio", "Likelihood Ratio"]
        titles = ["Most Extreme Pooled Sample Likelihood Ratios by Frequency of Associated Tokens",
                  "Most Extreme Within-Paper Likelihood Ratios by Frequency of Associated Tokens"]
        colors = [OI_constants.p1.value, OI_constants.p3.value]
        for ax, x, y, xlabel, ylabel, title, color in zip(axs, x_list, y_list, xlabels, ylabels, titles, colors):
            ax.scatter(x, y, c=color, s=2)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
        plt.savefig(os.path.join(self.path_to_output, 'scatter_ratios_by_frequency_' + str(self.ngrams) + '_grams.png'),
                    bbox_inches='tight')

        # Produce final results table.
        highest_df = (pd.concat([highest_pooled_ratios['Highest Pooled Ratios'],
                                 highest_fe_ratios["Highest Within-Paper Ratios"]],
                                axis=1)
                      .iloc[::-1]
                      .reset_index(drop=True)
                      )
        lowest_df = (pd.concat([lowest_pooled_ratios['Lowest Pooled Ratios'],
                                lowest_fe_ratios["Lowest Within-Paper Ratios"]],
                               axis=1).reset_index(drop=True)
                     )

        pd.set_option('display.max_colwidth', -1)
        metrics = results_table['Metrics'].reset_index(drop=True).loc[:3]
        results = pd.concat([lowest_df, highest_df, metrics], axis=1)
        results.columns = ["\textbf{" + column + "}" for column in results.columns]
        results.to_latex(os.path.join(self.path_to_output, model_name + "_ratio_results.tex"), index=False, escape=False)
