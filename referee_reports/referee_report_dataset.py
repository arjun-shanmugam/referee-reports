"""Defines the RefereeReportDataset class, which produces a report-level dataset.

    @author: Arjun Shanmugam
"""
import io
import os
from typing import List

from stargazer.stargazer import Stargazer
import matplotlib.pyplot as plt
import numpy as np

from referee_reports.constants import Colors
from referee_reports.pkldir.decode import decode
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from referee_reports.figure_utilities import plot_histogram, plot_labeled_hline, plot_labeled_vline, plot_scatter_with_shaded_errors, save_figure_and_close

from referee_reports.models import LikelihoodRatioModel, OLSRegression, RegularizedRegression


class RefereeReportDataset:
    """Builds a DataFrame at the report-level containing report,
    referee, and paper test_data. Allows the user to run statistical models for text analysis.
    """
    _df: pd.DataFrame
    _reports_df: pd.DataFrame
    _papers_df: pd.DataFrame
    _output_directory: str
    _seed: int
    _ngrams: int
    _reports_vocabulary: np.ndarray
    models: dict

    def __init__(self, cleaned_pickled_reports_file: str, cleaned_pickled_papers_file: str, output_directory: str, seed: int):
        """Instantiates a RefereeReportDataset object.

        Args:
            cleaned_pickled_reports_file (str): _description_
            cleaned_pickled_papers_file (str): _description_
            output_directory (str): _description_
            seed (int): _description_
        """
        # noinspection PyTypeChecker
        self._reports_df = pd.read_csv(io.StringIO(decode(cleaned_pickled_reports_file).decode('utf-8')), index_col=['paper', 'num'])
        # noinspection PyTypeChecker
        self._papers_df = pd.read_csv(io.StringIO(decode(cleaned_pickled_papers_file).decode('utf-8')), index_col='paper')
        self._output_directory = output_directory
        self._seed = seed
        self._df = pd.DataFrame(index=self._reports_df.index)
        self._reports_vocabulary = np.empty(1)
        self.models = {}
        np.random.seed(self._seed)  # Bad practice, but must set internal seed to ensure reproducible output from sklearn.

    def build_df(self, text_representation: str, ngrams: int, restrict_to_papers_with_mixed_gender_referees: bool, balance_sample_by_gender: bool):

        self._format_non_vocabulary_columns()
        self._build_dtm(text_representation, ngrams)
        self._merge_with_referee_characteristics()
        if restrict_to_papers_with_mixed_gender_referees:
            self._restrict_to_papers_with_mixed_gender_referees()
        if balance_sample_by_gender:
            self._balance_sample_by_gender()

    def _format_non_vocabulary_columns(self):
        self._reports_df.columns = [f"_{column}_" for column in self._reports_df.columns]
        self._papers_df.columns = [f"_{column}_" for column in self._papers_df.columns]

    def _restrict_to_papers_with_mixed_gender_referees(self):
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

        # Concatenate DTM with the rest of the test_data.
        dtm = pd.DataFrame(dtm, self._reports_df.index, columns=vectorizer.get_feature_names())
        self._reports_vocabulary = vectorizer.get_feature_names()
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

    def ols_regress(self, model_name: str, y: str, X: List[str], add_constant: bool,
                    logistic: bool, log_transform: bool, standardize: bool):
        # Check that specified independent, dependent variables are valid.
        self._validate_columns(y, X)

        # Select specified columns from DataFrame.
        dependent_variable = self._df[y]
        independent_variables = self._df[X]

        # Instantiate an OLS regression object.
        self.models[model_name] = OLSRegression(model_name=model_name,
                                                y_data=dependent_variable,
                                                X_data=independent_variables,
                                                add_constant=add_constant,
                                                log_transform=log_transform,
                                                standardize=standardize)

        # Run the regression.
        self.models[model_name].fit(logistic=logistic)

    def build_ols_results_table(self,
                                filename: str,
                                requested_models: List[str],
                                title: str = None,
                                show_confidence_intervals=False,
                                dependent_variable_name=None,
                                show_degrees_of_freedom=False,
                                rename_covariates=None):
        # Validate model names.
        for model_name in requested_models:
            if model_name not in self.models:
                raise ValueError("A model by that name has not been estimated.")

        # Validate model types.
        for model_name in requested_models:
            if self.models[model_name]._model_type != "OLS":
                raise TypeError("This function may only be used to produce output for OLS models.")

        # Grab results tables.
        results = []
        for model_name in requested_models:
            current_result = self.models[model_name]._results_table
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

        # Write LaTeX.
        latex = stargazer.render_latex()

        # Write to file.
        with open(os.path.join(self._output_directory, filename + ".tex"), "w") as output_file:
            output_file.write(latex)

    def regularized_regress(self, model_name: str, y: str, X: List[str], add_constant: bool, logistic: bool, log_transform: bool, standardize: bool,
                            penalty: str,
                            stratify: bool,
                            cv_folds: int,
                            alphas: np.ndarray,
                            adjust_alpha: bool,
                            l1_ratios: np.ndarray = None,
                            n_jobs: int = 1,
                            ):
        # Check that specified independent, dependent variables are valid.
        self._validate_columns(y, X)

        # Select specified columns from DataFrame.
        dependent_variable = self._df[y]
        independent_variables = self._df[X]

        # Instantiate a RegularizedRegression object.
        self.models[model_name] = RegularizedRegression(model_name, dependent_variable, independent_variables, add_constant, log_transform, standardize)

        # Run the regression
        self.models[model_name].fit(penalty, logistic, stratify, cv_folds, self._seed, alphas, adjust_alpha, l1_ratios, n_jobs)

    def plot_regularization_path(self, model_name: str, display_adjusted_alpha: bool):
        # Validate model names.
        if model_name not in self.models:
            raise ValueError("A model by that name has not been estimated.")

        # Validate model types.
        if self.models[model_name]._model_type != "Regularized":
            raise TypeError("This function may only be used to produce output for regularized regression models.")

        # Get results.
        coefficients_sorted, regularization_path, final_parameters = self.models[model_name]._results_table

        # Plot.
        fig, ax = plt.subplots()
        if np.isnan(final_parameters["$w_{adjusted}^{L1}^*$"]):
            xlabel = "Parameter $\\alpha$ Used to Fit Model"
        else:
            xlabel = "Parameters $(\\alpha, w^{L1})$ Used to Fit Model"

        if np.isnan(final_parameters["$w^{L1}^*$"]):
            x = np.array([label[0] for label in regularization_path.index])
            xticklabels = None
        else:
            x = np.array(range(0, len(regularization_path.index)))
            xticklabels = [str(round(label[0], 2)) + ", " + str(round(label[1], 2)) for label in regularization_path.index]

        plot_scatter_with_shaded_errors(ax,
                                        x=x,
                                        y=regularization_path['mean_loss'].values,
                                        yerr=regularization_path['std_loss'].values / np.sqrt(final_parameters["C.V. folds: "]),
                                        xlabel=xlabel,
                                        xticklabels=xticklabels,
                                        ylabel=f"Mean Negative Log Loss Across {final_parameters['C.V. folds: ']} C.V. Folds")
        plot_labeled_vline(ax, x=final_parameters["$\\alpha^{*}$: "], text="$\\alpha^{*}=" + str(round(final_parameters["$\\alpha^{*}$: "], 3)) + "$")
        ax.scatter(x=final_parameters["$\\alpha^{*}$: "], y=final_parameters["$\\bar{p}_{\\alpha^*}$: "], s=50, c=Colors.P3, zorder=10)

        if display_adjusted_alpha:
            plot_labeled_vline(ax,
                               x=final_parameters["$\\alpha^{*}_{adjusted}$: "],
                               text="$\\alpha^{*}_{adjusted}=" + str(round(final_parameters["$\\alpha^{*}_{adjusted}$: "], 3)) + "$")
            plot_labeled_hline(ax,
                               y=final_parameters["$\\bar{p}_{\\alpha^*}$: "] + final_parameters["$SE_{\\alpha^*}$: "],
                               text="$\\bar{p}_{\\alpha^*} + SE_{\\alpha^*}$")

        save_figure_and_close(fig, os.path.join(self._output_directory, model_name + "_regularization_path.png"), bbox_inches='tight')

    def build_regularized_results_table(self, model_name, num_coefs_to_report=40):
        # Validate model names.
        if model_name not in self.models:
            raise ValueError("A model by that name has not been estimated.")

        # Validate model types.
        if self.models[model_name]._model_type != "Regularized":
            raise TypeError("This function may only be used to produce output for regularized regression models.")

        # Get results.
        coefficients_sorted, regularization_path, final_parameters = self.models[model_name]._results_table

        if len(coefficients_sorted) < num_coefs_to_report * 2:
            raise ValueError(
                f"{num_coefs_to_report * 2} coefficients were requested, but the requested model only contains {len(coefficients_sorted)} variables.")

        # Format final parameters column.
        final_parameters = final_parameters.reset_index()['index'].str.cat(final_parameters.round(3).fillna("--").astype(str).values, sep=" ")

        # Get largest nonzero coefficients; concatenate them with the associated token.
        top_coefficients = coefficients_sorted.iloc[:num_coefs_to_report]
        top_coefficients = top_coefficients[top_coefficients != 0]
        top_coefficients = top_coefficients.reset_index()['index'].str.cat(top_coefficients.round(3).astype(str).values, sep=": ")

        # Get smallest nonzero coefficients; concatenate them with the associated token.
        bottom_coefficients = coefficients_sorted.iloc[-num_coefs_to_report:]
        bottom_coefficients = bottom_coefficients[bottom_coefficients != 0]
        bottom_coefficients = bottom_coefficients.reset_index()['index'].str.cat(bottom_coefficients.round(3).astype(str).values, sep=": ")

        results_table = pd.concat([top_coefficients, bottom_coefficients, final_parameters], axis=1)
        results_table = results_table.fillna(' ')
        results_table.columns = ['Words Most Predictive of Female Referee',
                                 'Words Most Predictive of Non-Female Referee',
                                 'Other Metrics']
        results_table.columns = ['\textbf{' + column + '}' for column in results_table.columns]
        results_table.to_latex(os.path.join(self._output_directory, model_name + "_results.tex"),
                               index=False,
                               escape=False,
                               float_format="%.3f")

        # Plot path of alphas.

    def produce_summary_statistics(self):
        # Plot number of referees on each paper of each gender.
        def tally_referee_genders(referee_genders_for_single_paper: pd.DataFrame):
            male_referees = (referee_genders_for_single_paper == 0).sum()
            female_referees = (referee_genders_for_single_paper == 1).sum()
            return f"{female_referees} female, {male_referees} male"

        referee_gender_breakdown_counts = (self._df['_female_']
                                           .groupby(level=0)
                                           .agg(tally_referee_genders)
                                           .value_counts())
        fig, ax = plt.subplots()
        referee_gender_breakdown_counts.plot.pie(ax=ax, colors=Colors.OI_colors, ylabel="", autopct='%.2f%%')
        save_figure_and_close(fig, os.path.join(self._output_directory, "referee_count_and_gender.png"), bbox_inches='tight')

        # Plot distribution of decision and recommendation.
        for variable, filename in zip(['_decision_', '_recommendation_'], ["referee_decisions.png", "referee_recommendations.png"]):
            distribution = self._df[variable].value_counts().rename()
            fig, ax = plt.subplots()
            distribution.plot.pie(ax=ax, colors=Colors.OI_colors, ylabel="", autopct='%.2f%%')
            save_figure_and_close(fig, os.path.join(self._output_directory, filename))

        # Plot distribution of decision and recommendation, separately by gender.
        for variable, filename in zip(['_decision_', '_recommendation_'], ["referee_decisions_by_gender.png", "referee_recommendations_by_gender.png"]):
            distribution_female = self._df.loc[self._df['_female_'] == 1, variable].value_counts()
            distribution_male = self._df.loc[self._df['_female_'] == 0, variable].value_counts()
            fig, (ax1, ax2) = plt.subplots(1, 2)
            distribution_female.plot.pie(ax=ax1, colors=Colors.OI_colors, ylabel="", autopct='%.2f%%')
            ax1.set_title("Female Referees")
            distribution_male.plot.pie(ax=ax2, colors=Colors.OI_colors, ylabel="", autopct='%.2f%%')
            ax2.set_title("Non-female Referees")
            save_figure_and_close(fig, os.path.join(self._output_directory, filename))

        # Plot distribution of report lengths.
        tokens_per_report = self._df[self._reports_vocabulary].sum(axis=1)
        xlabel = "Length"
        fig, ax = plt.subplots()
        plot_histogram(ax, tokens_per_report, xlabel=xlabel)
        save_figure_and_close(fig, os.path.join(self._output_directory, "histogram_report_lengths.png"))

        # Plot distribution of report lengths, separately by gender.
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        for gender, title, ax, color in zip([1, 0], ["Written by Female Referee", "Written by Non-female Referee"], [ax1, ax2], [Colors.P1,
                                                                                                                                 Colors.P2]):
            report_lengths = self._df.loc[self._df['_female_'] == gender, self._reports_vocabulary].sum(axis=1)
            plot_histogram(ax, x=report_lengths, title=title, xlabel="Length", color=color)
        save_figure_and_close(fig, os.path.join(self._output_directory, "histogram_report_lengths_by_gender.png"))

        # Plot distribution of number of reports in which words appear.
        num_reports_where_word_appears = (self._df[self._reports_vocabulary]
                                          .mask(self._df[self._reports_vocabulary] > 0, 1)
                                          .sum(axis=0)
                                          .transpose()
                                          )
        fig, ax = plt.subplots()
        xlabel = "Number of Reports"
        plot_histogram(ax=ax, x=num_reports_where_word_appears, title="", xlabel=xlabel)
        save_figure_and_close(fig, os.path.join(self._output_directory, "histogram_report_appearances.png"))

        # Print most common words for men and women.
        male_occurrences_per_word = (self._df.loc[self._df['_female_'] == 0, self._reports_vocabulary]
                                     .sum(axis=0)
                                     .transpose()
                                     .sort_values(ascending=False)
                                     .iloc[:50]
                                     .round(3)
                                     .rename("Most Common Words, Reports by Non-Females")
                                     .reset_index())
        male_occurrences_per_word = (male_occurrences_per_word['index']
                                     + ":"
                                     + male_occurrences_per_word["Most Common Words, Reports by Non-Females"].astype(str))
        male_occurrences_per_word = male_occurrences_per_word.rename("\textbf{Most Common Words, Reports by Non-Females}")
        female_occurrences_per_word = (self._df.loc[self._df['_female_'] == 1, self._reports_vocabulary]
                                       .sum(axis=0)
                                       .transpose()
                                       .sort_values(ascending=False)
                                       .iloc[:50]
                                       .round(3)
                                       .rename("Most Common Words, Reports by Females")
                                       .reset_index())
        female_occurrences_per_word = (female_occurrences_per_word['index']
                                       + ":"
                                       + female_occurrences_per_word["Most Common Words, Reports by Females"].astype(str))
        female_occurrences_per_word = female_occurrences_per_word.rename("\textbf{Most Common Words, Reports by Females}")
        pd.concat([female_occurrences_per_word, male_occurrences_per_word], axis=1).to_latex(os.path.join(self._output_directory,
                                                                                                          "most_common_words_by_gender.tex"),
                                                                                             index=False,
                                                                                             escape=False,
                                                                                             )

    def get_reports_vocabulary(self):
        return self._reports_vocabulary

    def calculate_likelihood_ratios(self, model_name: str, model_type: str):

        # TODO: Update likelihood ratio class
        # To cater to old codebase, we have to pass in the DTM in a slightly altered form:
        dtm = self._df[self._reports_vocabulary].copy()
        dtm.index.names = ["_paper_", "_num_"]
        dtm = dtm.reset_index()
        self.models[model_name] = LikelihoodRatioModel(dtm=dtm,
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
        results.to_latex(os.path.join(self._output_directory, model_name + "_ratio_results.tex"), index=False, escape=False)
