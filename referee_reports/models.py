"""Defines the Regression class, which implements functionality that is common to OLSRegression, Regularized Regression, and PanelRegression.

Author: Arjun Shanmugam
"""
from typing import List

import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from multiprocessing import Manager
from statistics import mean

import numpy as np
import pandas as pd
from joblib import Parallel, delayed



class Regression:
    _y_data: pd.Series
    _X_data: pd.DataFrame
    _standardize: bool
    _log_transform: bool
    _add_constant: bool
    _dummy_variables: List[str]
    _model_name: str
    _model_type: str
    _results_table: pd.DataFrame

    def __init__(self,
                 model_name: str,
                 y_data: pd.Series,
                 X_data: pd.DataFrame,
                 add_constant: bool,
                 log_transform: bool,
                 standardize: bool):
        self._model_name = model_name
        self._y_data = y_data
        self._X_data = X_data
        self._standardize = standardize
        self._log_transform = log_transform
        self._add_constant = add_constant
        if type(self) == OLSRegression:
            self._model_type = "OLS"
        elif type(self) == RegularizedRegression:
            self._model_type = "Regularized"
        elif type(self) == Regression:
            self._model_type = "Base Regression Class"  # Should only trigger during testing.
        else:
            raise NotImplementedError("The method __init__ was called by an object of an unrecognized class, and the attribute _model_type"
                                      "cannot be automatically assigned. See the method definition for Regression.__init__ for more information.")

        self._results_table = None
        self._dummy_variables = None

    def _preprocess_inputs(self):
        # Store numeric and categorical columns separately.
        categorical_df = self._X_data.select_dtypes(exclude=['number'])
        numeric_df = self._X_data.select_dtypes(include=['number'])  # Setting include to 'number' selects all numeric datatypes.

        # Log(x+1) transform numeric variables.
        if self._log_transform and not numeric_df.empty:
            numeric_df = np.log(numeric_df + 1)

        # If requested, _standardize numeric variables using a Scikit-Learn Standard Scaler.
        if self._standardize and not numeric_df.empty:
            numeric_df = pd.DataFrame(StandardScaler().fit_transform(numeric_df), columns=numeric_df.columns, index=numeric_df.index)

        # Generate dummies for categorical variables. For each categorical variable, dummies are generated for each category except one.
        if not categorical_df.empty:
            categorical_df = pd.get_dummies(categorical_df, prefix=categorical_df.columns, prefix_sep=": ", drop_first=True, dtype='int64')

        # Keep track of dummy variable names.
        self._dummy_variables = categorical_df.columns.tolist()

        # Set _X_data to the preprocessed input.
        self._X_data = pd.concat([categorical_df, numeric_df], axis=1)

        # Add a constant column if specified.
        if self._add_constant:
            self._X_data.loc[:, "Constant"] = 1


class OLSRegression(Regression):

    def fit(self, logistic: bool):
        """Fit an OLS model.

        Uses StatsModels to fit an OLS model.
        """
        self._preprocess_inputs()

        if logistic:
            raise NotImplementedError("Non-regularized logistic regression has not been implemented.")
            # self.results_table = LogitOddsWrapper(sm.Logit(self.y.astype(float), self.X.astype(float)).fit(maxiter=1000))
        else:
            self._results_table = sm.OLS(self._y_data.astype(float), self._X_data.astype(float)).fit()


def get_optimal_parameters(cv_results: pd.DataFrame, penalty: str, N: int, cv_folds: int):
    if penalty == 'elasticnet':
        top_parameters_index = cv_results.sort_values(['param_C', 'param_l1_ratio'], ascending=[True, False])['mean_test_neg_log_loss'].idxmax()

    else:
        top_parameters_index = cv_results.sort_values('param_C', ascending=True)['mean_test_neg_log_loss'].idxmax()


    top_mean_test_loss = cv_results.loc[top_parameters_index, 'mean_test_neg_log_loss']
    top_se_test_loss = cv_results.loc[top_parameters_index, 'std_test_neg_log_loss'] / np.sqrt(cv_folds)

    C_star = cv_results.loc[top_parameters_index, 'param_C']
    alpha_star = 1 / (C_star * N) if penalty == 'l1' else 1 / C_star
    w_l1_star = cv_results.loc[top_parameters_index, 'param_l1_ratio'] if penalty == 'elasticnet' else np.nan

    # Restrict to models with mean test loss within one standard error of the best model's mean test loss.
    within_1_se_mask = ((cv_results['mean_test_neg_log_loss'] > top_mean_test_loss - top_se_test_loss) &
                        (cv_results['mean_test_neg_log_loss'] < top_mean_test_loss + top_se_test_loss))
    rows_within_1_se = cv_results.loc[within_1_se_mask, :]

    if len(rows_within_1_se) == 0:  # If there are no rows within 1 se, adjusted parameters are identical to normal parameters.
        top_adjusted_parameters_index = top_parameters_index
        alpha_star_adjusted = alpha_star
        w_l1_star_adjusted = w_l1_star
        top_adjusted_mean_test_loss = top_mean_test_loss
        top_adjusted_se_test_loss = top_se_test_loss
    else:
        # If penalty is elasticnet, choose the lowest C and the highest L1 ratio.
        if penalty == 'elasticnet':
            top_adjusted_parameters_index = rows_within_1_se[['param_C', 'param_l1_ratio']].sort_values(['param_C', 'param_l1_ratio'],
                                                                                                        ascending=[True, False]).index[0]
            C_star_adjusted = cv_results.loc[top_adjusted_parameters_index, 'param_l1_ratio']
            alpha_star_adjusted = 1 / C_star_adjusted
            w_l1_star_adjusted = 1 / cv_results.loc[top_adjusted_parameters_index, 'param_l1_ratio']

            top_adjusted_mean_test_loss = cv_results.loc[top_adjusted_parameters_index, 'mean_test_neg_log_loss']
            top_adjusted_se_test_loss = cv_results.loc[top_adjusted_parameters_index, 'std_test_neg_log_loss'] / np.sqrt(cv_folds)

        # If penalty is not elasticnet, choose the lowest C.
        else:

            top_adjusted_parameters_index = rows_within_1_se['param_C'].sort_values(ascending=True).index[0]
            C_star_adjusted = cv_results.loc[top_adjusted_parameters_index, 'param_C']
            alpha_star_adjusted = 1 / (C_star_adjusted * N) if penalty == 'l1' else 1 / C_star_adjusted
            w_l1_star_adjusted = np.nan

            top_adjusted_mean_test_loss = cv_results.loc[top_adjusted_parameters_index, 'mean_test_neg_log_loss']
            top_adjusted_se_test_loss = cv_results.loc[top_adjusted_parameters_index, 'std_test_neg_log_loss'] / np.sqrt(cv_folds)

    return (top_parameters_index, top_adjusted_parameters_index, alpha_star, -1 * top_mean_test_loss, top_se_test_loss,
            w_l1_star, alpha_star_adjusted, -1 * top_adjusted_mean_test_loss, top_adjusted_se_test_loss, w_l1_star_adjusted)


def custom_refit(cv_results, adjust_alpha_value: bool, penalty: str, N: int, cv_folds: int):
    cv_results = pd.DataFrame(cv_results)
    optimal_parameters = get_optimal_parameters(cv_results, penalty, N, cv_folds)

    if adjust_alpha_value:
        return optimal_parameters[1]  # Return the index of the adjusted optimal parameters.
    else:
        return optimal_parameters[0]


class RegularizedRegression(Regression):
    def fit(self,
            penalty: str,
            logistic: bool,
            stratify: bool,
            cv_folds: int,
            seed: int,
            alphas: np.ndarray,
            adjust_alpha: bool,
            l1_ratios: np.ndarray = None,
            n_jobs: int = 1):

        # Error check penalty argument.
        if penalty not in ['l1', 'l2', 'elasticnet']:
            raise ValueError("Parameter penalty only accepts \"l1\", \"l2\", and \"elasticnet\" as arguments.")

        self._preprocess_inputs()

        # Initialize appropriate estimator.
        N = len(self._X_data)
        Cs = 1 / (alphas * N) if penalty == 'l1' else 1 / alphas
        solver = 'saga' if penalty == 'elasticnet' else 'liblinear'
        param_grid = [{'C': Cs, 'l1_ratio': l1_ratios}] if penalty == 'elasticnet' else [{'C': Cs}]
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed) if stratify else KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        if logistic:
            estimator = LogisticRegression(fit_intercept=False,
                                           penalty=penalty,
                                           solver=solver,
                                           random_state=seed)
        else:
            raise NotImplementedError("Non-logistic regularized regression has not been implemented.")
            # Note: This class will produce different results from LassoCV, RidgeCV, and ElasticNetCV when running non-logistic regularized regression.
            # See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html
        grid_search = GridSearchCV(estimator=estimator,
                                   param_grid=param_grid,
                                   scoring=['neg_log_loss', 'accuracy'],
                                   refit=lambda cv_results: custom_refit(cv_results, adjust_alpha, penalty, N, cv_folds),
                                   cv=cv,
                                   n_jobs=n_jobs)
        grid_search_result = grid_search.fit(self._X_data, self._y_data)

        # Build results table.
        cv_results_df = pd.DataFrame(grid_search_result.cv_results_)
        # Sorted list of coefficients.
        coefficients_sorted = pd.Series(grid_search_result.best_estimator_.coef_.tolist()[0], index=self._X_data.columns).sort_values(ascending=False)

        # DataFrame where parameters form the index and mean(loss), se(loss) form the columns.
        regularization_path = pd.DataFrame(cv_results_df[['mean_test_neg_log_loss', 'std_test_neg_log_loss']]) * -1
        regularization_path.columns = ['mean_loss', 'std_loss']
        alphas = 1 / (cv_results_df['param_C'] * N) if penalty == 'l1' else 1 / cv_results_df['param_C']
        if penalty == 'elasticnet':
            l1s = cv_results_df['param_l1_ratio']
        cols = [alphas, l1s] if penalty == 'elasticnet' else [alphas]
        params = pd.concat(cols, axis=1).to_dict('records')

        regularization_path.index = [tuple(dictionary.values()) for dictionary in params]

        # Get final parameters.
        N = len(self._X_data)
        portion_of_coefficients_0 = (coefficients_sorted == 0).mean()  # Portion of coefficients equal to 0.
        (_, _, alpha_star, top_mean_test_loss, top_se_test_loss, w_l1_star, alpha_star_adjusted, top_adjusted_mean_test_loss, top_adjusted_se_test_loss,
         w_l1_star_adjusted) = get_optimal_parameters(cv_results_df, penalty=penalty, N=N, cv_folds=cv_folds)
        loss_after_final_refit = log_loss(self._y_data, grid_search_result.best_estimator_.predict_proba(self._X_data))
        accuracy_after_final_refit = grid_search_result.best_estimator_.score(self._X_data, self._y_data)
        final_parameter_names = ["N:",
                                 "Portion of coefficients equal to 0:",
                                 "$\\alpha^{*}$: ",
                                 "$\\bar{p}_{\\alpha^*}$: ",
                                 "$SE_{\\alpha^*}$: ",
                                 "$w^{L1}^*$",
                                 "$\\alpha^{*}_{adjusted}$: ",
                                 "$\\bar{p}_{\\alpha^{*}_{adjusted}}$: ",
                                 "$SE_{\\alpha^{*}_{adjusted}}$: ",
                                 "$w_{adjusted}^{L1}^*$",
                                 "Binary cross-entropy on final refit: ",
                                 "Accuracy on final refit: ",
                                 "C.V. folds: "]
        final_parameter_values = [N,
                                  portion_of_coefficients_0,
                                  alpha_star,
                                  top_mean_test_loss,
                                  top_se_test_loss,
                                  w_l1_star,
                                  alpha_star_adjusted,
                                  top_adjusted_mean_test_loss,
                                  top_adjusted_se_test_loss,
                                  w_l1_star_adjusted,
                                  loss_after_final_refit,
                                  accuracy_after_final_refit,
                                  cv_folds]
        final_parameters = pd.Series(final_parameter_values, index=final_parameter_names)

        self._results_table = (coefficients_sorted, regularization_path, final_parameters)


"""Defines the LikelihoodRatioModel class.
"""
class LikelihoodRatioModel:
    """Uses a document term matrix, a binary vector of document classifications, and a vector giving the entities of each document
       to calculate pooled sample and within-group likelihood ratios for the tokens in the vocabulary.
    """
    dtm: pd.DataFrame
    fe_variables: pd.Series
    results_table: pd.DataFrame

    def __init__(self, dtm: pd.DataFrame, document_classification_variable: pd.Series, fe_variable: pd.Series, model_type: str):
        """Instantiates a LikelihoodRatioModel object.

        Args:
            dtm (pd.DataFrame): The document term matrix.
            document_classification_variable (pd.Series): A Series giving the classification of each document in the document term matrix.
            fe_variable (pd.Series): A Series giving the F.E. group of each document in the document term matrix.

        Raises:
            ValueError: If document_classification_variable is not binary.
            ValueError: If the index of document_classification_variable does not match the index of dtm.
            ValueError: If the index of fe_variable does not match the index of dtm.
        """
        # Check that specified group variable is binary.
        if not np.array_equal(document_classification_variable.unique(), np.array([1, 0], dtype=np.int64)):
            raise ValueError("Group variable must be binary with int64-type values 0 and 1.")
        # Check that we can match each row of D.T.M. with a row of document_classification_variable.
        if not dtm.index.equals(document_classification_variable.index):
            raise ValueError("Index of document classification variable does not match index of D.T.M.")
        # Check that we can match each row of D.T.M. with a row of fe_variable.
        if not dtm.index.equals(fe_variable.index):
            raise ValueError("Index of F.E. variable does not match index of D.T.M.")
        self.words = dtm.columns.tolist()
        self.df = pd.concat([dtm, fe_variable, document_classification_variable], axis=1)
        self.fe_name = fe_variable.name
        self.document_classification_variable_name = document_classification_variable.name
        self.model_type = model_type
        self.results_table = pd.DataFrame()

    def get_model_type(self):
        """Returns the type of this model.

        Returns:
            str: The type of this model.
        """
        return self.model_type

    def get_results_table(self):
        """Returns the pandas DataFrame containing this model's results.

        Returns:
            pd.DataFrame: A DataFrame containing the model's results.
        """
        return self.results_table

    def _calculate_likelihood_ratio(self, t: str, dtm: pd.DataFrame, laplace_smooth: bool):
        """Calculate the likelihood ratio for a token t in a document term matrix dtm.

        Note that by passing a restricted set of rows for dtm, the likelihood ratio
        for token t can be calculated separately within groups of rows in the overarching
        document term matrix.

        Args:
            t (str): The token whose likelihood ratio will be calculated.
            dtm (pd.DataFrame): The document term matrix in which we want to calculate the likelihood ratio for token t.

        Returns:
            float: The likelihood ratio for token t.

        """
        # Calculate T for LaPlace smoothing.
        apperances_of_each_token = (dtm[self.words].sum(axis=0)  # Get series containing total counts of each word
                                    .transpose()
                                    )
        # If the passed D.T.M. is the entire corpus, every row will be nonzero
        # If the passed D.T.M. only contains documents in a single paper group, some rows will be 0.

        # Select rows greater than 0 and get length.
        if laplace_smooth:
            laplace_parameter = 1
        else:
            laplace_parameter = 0
        T = len(apperances_of_each_token.loc[apperances_of_each_token > 0]) * laplace_parameter

        # Calculate probability of token t in document class 1
        mu_hat_1_t = ((dtm.loc[dtm[self.document_classification_variable_name] == 1][t].sum() + laplace_parameter) /
                      (dtm.loc[dtm[self.document_classification_variable_name] == 1][self.words].sum().sum() + T))

        # Calculate probability of token t in document class 0
        mu_hat_0_t = ((dtm.loc[dtm[self.document_classification_variable_name] == 0][t].sum() + laplace_parameter) /
                      (dtm.loc[dtm[self.document_classification_variable_name] == 0][self.words].sum().sum() + T))

        # Return likelihood ratio for the token t.
        l_t = mu_hat_1_t / mu_hat_0_t

        return l_t

    def _fit_helper_model_0(self, word, pooled_likelihood_ratios, fe_likelihood_ratios, fe_groups, laplace_smooth):
        pooled_likelihood_ratios.append((self._calculate_likelihood_ratio(word, self.df, laplace_smooth), word))
        likelihood_ratios_within_groups = []
        for fe_group in fe_groups:
            df_for_current_group = self.df.loc[self.df[self.fe_name] == fe_group]
            # Check whether current word appears in any of the documents associated with current F.E. group.
            word_in_current_fe_group = df_for_current_group[word].sum() > 0
            if word_in_current_fe_group:
                likelihood_ratios_within_groups.append(self._calculate_likelihood_ratio(word,
                                                                                        df_for_current_group,
                                                                                        laplace_smooth))
            else:
                likelihood_ratios_within_groups.append(1)

        fe_likelihood_ratios.append((mean(likelihood_ratios_within_groups), word))

    def _fit_helper_model_1(self, word, pooled_likelihood_ratios, fe_likelihood_ratios, fe_groups, laplace_smooth):
        pooled_likelihood_ratios.append((self._calculate_likelihood_ratio(word, self.df, laplace_smooth), word))
        likelihood_ratios_within_groups = []
        for fe_group in fe_groups:
            df_for_current_group = self.df.loc[(self.df[self.fe_name] == fe_group)]
            df_for_current_group_class_0 = self.df.loc[(self.df[self.fe_name] == fe_group) & (self.df[self.document_classification_variable_name] == 0)]
            df_for_current_group_class_1 = self.df.loc[(self.df[self.fe_name] == fe_group) & (self.df[self.document_classification_variable_name] == 1)]

            word_in_current_fe_group_class_0 = df_for_current_group_class_0[word].sum() > 0
            word_in_current_fe_group_class_1 = df_for_current_group_class_1[word].sum() > 0

            if word_in_current_fe_group_class_0 and word_in_current_fe_group_class_1:
                likelihood_ratios_within_groups.append(self._calculate_likelihood_ratio(word,
                                                                                        df_for_current_group,
                                                                                        laplace_smooth))
            else:
                likelihood_ratios_within_groups.append(np.nan)
        likelihood_ratios_within_groups = pd.Series(likelihood_ratios_within_groups)
        ratio_word_tuple = (likelihood_ratios_within_groups.mean(), word)
        fe_likelihood_ratios.append(ratio_word_tuple)

    def _fit_helper_model_2(self, word, pooled_likelihood_ratios, fe_likelihood_ratios, fe_groups, laplace_smooth):
        pooled_likelihood_ratios.append((self._calculate_likelihood_ratio(word, self.df, laplace_smooth), word))
        likelihood_ratios_within_groups = []
        for fe_group in fe_groups:
            df_for_current_group = self.df.loc[(self.df[self.fe_name] == fe_group)]
            df_for_current_group_class_0 = self.df.loc[(self.df[self.fe_name] == fe_group) & (self.df[self.document_classification_variable_name] == 0)]
            df_for_current_group_class_1 = self.df.loc[(self.df[self.fe_name] == fe_group) & (self.df[self.document_classification_variable_name] == 1)]

            word_in_current_fe_group_class_0 = df_for_current_group_class_0[word].sum() > 0
            word_in_current_fe_group_class_1 = df_for_current_group_class_1[word].sum() > 0

            if word_in_current_fe_group_class_0 and word_in_current_fe_group_class_1:
                likelihood_ratios_within_groups.append(self._calculate_likelihood_ratio(word,
                                                                                        df_for_current_group,
                                                                                        laplace_smooth))
            else:
                likelihood_ratios_within_groups.append(1)
        ratio_word_tuple = (np.mean(np.array(likelihood_ratios_within_groups)), word)
        fe_likelihood_ratios.append(ratio_word_tuple)

    def _fit_helper_model_3(self, word, pooled_likelihood_ratios, fe_likelihood_ratios, fe_groups, laplace_smooth):
        pooled_likelihood_ratios.append((self._calculate_likelihood_ratio(word, self.df, laplace_smooth), word))
        likelihood_ratios_within_groups = []
        for fe_group in fe_groups:  # TODO: NOTE THAT WITHIN-PAPER LIKELIHOOD RATIOS FOR MODEL 3 ARE IDENTICAL TO THOSE IN MODEL 1
            #
            # THE ONLY DIFFERENCE IS THAT IN MODEL 3, WE DO NOT REQUIRE THAT WORDS APPEAR IN ALL OF THE REPORTS ASSOCIATED WITH AT LEAST ONE PAPER.
            #
            # IN MODEL 3, WE ONLY REQUIRE THAT THEY APPEAR IN AT LEAST ONE MALE DOCUMENT AND AT LEAST ONE FEMALE DOCUMENT.
            #
            # THIS MEANS THAT ANY TOKENS WHICH DO NOT APPEAR IN ALL OF THE REPORTS ASSOCIATED WITH ANY ONE PAPER BUT APPEAR IN AT LEAST ONE FEMALE DOCUMENT AND AT LEAST ONE MALE DOCUMENT WILL BE INCLUDED IN THIS MODEL,
            # BUT NOT INCLUDED IN MODEL 2 (OR MODEL 1, WHOSE SAMPLE RESTRICTIONS ARE IDENTICAL TO MODEL 2)
            df_for_current_group = self.df.loc[(self.df[self.fe_name] == fe_group)]
            df_for_current_group_class_0 = self.df.loc[(self.df[self.fe_name] == fe_group) & (self.df[self.document_classification_variable_name] == 0)]
            df_for_current_group_class_1 = self.df.loc[(self.df[self.fe_name] == fe_group) & (self.df[self.document_classification_variable_name] == 1)]

            word_in_current_fe_group_class_0 = df_for_current_group_class_0[word].sum() > 0
            word_in_current_fe_group_class_1 = df_for_current_group_class_1[word].sum() > 0

            if word_in_current_fe_group_class_0 and word_in_current_fe_group_class_1:
                likelihood_ratios_within_groups.append(self._calculate_likelihood_ratio(word,
                                                                                        df_for_current_group,
                                                                                        laplace_smooth))
            else:
                likelihood_ratios_within_groups.append(1)
        ratio_word_tuple = (np.mean(np.array(likelihood_ratios_within_groups)), word)
        fe_likelihood_ratios.append(ratio_word_tuple)

    def fit(self):
        """Estimate likelihood ratios for each token in the document term matrix.

        Loops through each word and calculates the pooled sample likelihood ratio
        for that word. For each word, also calculates the likelihood ratio separately
        in each F.E. group, then takes the mean of these likelihood ratios to produce a
        likelihood ratio which controls for variation across F.E. groups.
        """
        # Get all F.E. groups so we can likelihood ratio for each token within each F.E. group.
        fe_groups = self.df[self.fe_name].unique()

        # Create lists which can be shared between processes
        manager = Manager()
        pooled_likelihood_ratios = manager.list()  # 1-D. Each element is the likelihood ratio for one word.
        # 1-D. Each element is the within-paper likelihood ratio for one word, averaged across papers.
        fe_likelihood_ratios = manager.list()

        if self.model_type == "Likelihood Ratio Model 0":
            print("Fitting Model 0.")
            laplace_smooth = True
            # No need to restrict words in document term matrix.
            helper_func = self._fit_helper_model_0
        elif self.model_type == "Likelihood Ratio Model 1":
            print("Fitting Model 1.")
            laplace_smooth = False
            indicator_df = (
                self.df[self.words].where(self.df[self.words] == 0, 1)  # (row, col) contains a dummy indicating the presence of word col in document row.
            )
            indicator_df = (
                pd.concat([indicator_df, self.df[self.fe_name]], axis=1)
                .groupby(by=self.fe_name).mean()  # For each column, take the mean of the dummies by paper.
                #  In this matrix, if cell (j, t) = 1, then token t apears in all reports associated with paper j.
            )

            indicator_series = (
                indicator_df.where(indicator_df == 1,
                                   0)  # Replace values not equal to 1 with 0. Now, if cell (j, t) = 0, then token t does not appear in all reports associated with paper j.
                .sum(axis=0)  # Sum to get the total number of papers where each token t appears in all associated reports.
                .transpose()  # Convert to a Series.
            )

            self.words = indicator_series[
                indicator_series > 0].index.tolist()  # Select all words which appear in all reports associated with at least one paper group.

            helper_func = self._fit_helper_model_1
        elif self.model_type == "Likelihood Ratio Model 2":
            print("Fitting Model 2.")
            laplace_smooth = False
            indicator_df = (
                self.df[self.words].where(self.df[self.words] == 0, 1)  # (row, col) contains a dummy indicating the presence of word col in document row.
            )
            indicator_df = (
                pd.concat([indicator_df, self.df[self.fe_name]], axis=1)
                .groupby(by=self.fe_name).mean()  # For each column, take the mean of the dummies by paper.
                #  In this matrix, if cell (j, t) = 1, then token t apears in all reports associated with paper j.
            )

            indicator_series = (
                indicator_df.where(indicator_df == 1,
                                   0)  # Replace values not equal to 1 with 0. Now, if cell (j, t) = 0, then token t does not appear in all reports associated with paper j.
                .sum(axis=0)  # Sum to get the total number of papers where each token t appears in all associated reports.
                .transpose()  # Convert to a Series.
            )

            self.words = indicator_series[
                indicator_series > 0].index.tolist()  # Select all words which appear in all reports associated with at least one paper group.

            helper_func = self._fit_helper_model_2
        elif self.model_type == "Likelihood Ratio Model 3":
            print("Fitting Model 3")
            laplace_smooth = False
            indicator_df = (
                self.df[self.words].where(self.df[self.words] == 0, 1)  # Now, (row, col) contains a dummy indicating the presence of word col in document row.
            )

            indicator_df = (
                pd.concat([indicator_df, self.df[self.document_classification_variable_name]], axis=1)
                .groupby(by=self.document_classification_variable_name).sum()
            # Now (row, col) contains the number of documents of gender row containing token col.
            )

            indicator_series = (
                indicator_df.where(indicator_df == 0, 1)  # Now, (row, col) ontains an indicator for whether word col appears in any documents of gender row.
                .sum(
                    axis=0)  # Now, there is one row, (1, col). (1, col) contains 2 if word col appears at least once in male documents and at least once in female documents.
                .transpose()
            )

            self.words = indicator_series[
                indicator_series == 2].index.tolist()  # Select all words which appear in at least one male report and at least one female reports.

            # Produce a list of the most frequently occuring male-only words and most frequently occuring female-only words.
            single_gender_words = indicator_series[indicator_series == 1].index.tolist()
            single_gender_word_counts_by_gender = (
                self.df.loc[:, single_gender_words + [self.document_classification_variable_name]]
                .groupby(by=self.document_classification_variable_name).sum()
                .transpose()
                .rename(columns={0: "Occurrences in Male-Written Reports", 1: "Occurrences in Female-Written Reports"})
            )

            highest_30_male = single_gender_word_counts_by_gender['Occurrences in Male-Written Reports'].sort_values(ascending=False).reset_index()
            highest_30_male = highest_30_male['index'] + ": " + highest_30_male['Occurrences in Male-Written Reports'].astype(str)
            highest_30_female = single_gender_word_counts_by_gender['Occurrences in Female-Written Reports'].sort_values(ascending=False).reset_index()
            highest_30_female = highest_30_female['index'] + ": " + highest_30_female['Occurrences in Female-Written Reports'].astype(str)

            most_frequent_single_gender_words = pd.concat([highest_30_male, highest_30_female], axis=1).loc[0:30]
            most_frequent_single_gender_words.columns = ["\\textbf{Occurrences of Words Used Only by Males}",
                                                         "\\textbf{Occurrences of Words Used Only by Females}"]
            print(most_frequent_single_gender_words)
            most_frequent_single_gender_words.to_latex("~/Desktop/most_frequent_single_gender_bigrams.tex", index=False)
            print("Generated table of most frequent single-gender words")

            helper_func = self._fit_helper_model_3
        else:
            raise ValueError("Please specify a valid model.")

        # Restrict D.T.M. to include only the appropriate words.
        self.df = self.df.loc[:, self.words + [self.fe_name, self.document_classification_variable_name]]

        # Calculate likeilhood ratios in parallel
        _ = Parallel(n_jobs=16)(delayed(helper_func)(word,
                                                     pooled_likelihood_ratios,
                                                     fe_likelihood_ratios,
                                                     fe_groups,
                                                     laplace_smooth) for word in self.words)

        # Likelihood ratios for each word in pooled sample.
        pooled_likelihood_ratios = pd.DataFrame(list(pooled_likelihood_ratios), columns=["pooled_ratios", "words"])
        pooled_likelihood_ratios = pooled_likelihood_ratios.set_index("words")

        # Likelihood ratios for each word taken within F.E. groups and then averaged across groups.
        fe_likelihood_ratios = pd.DataFrame(list(fe_likelihood_ratios), columns=["fe_ratios", "words"])
        fe_likelihood_ratios = fe_likelihood_ratios.set_index("words")

        # Percentage of reports in which each word appears at least once.
        appearance_dtm = (self.df[self.words] > 0).mean(axis=0).transpose()
        appearances_in_reports = appearance_dtm.rename("frequency_over_documents")

        # Percentage of F.E. groups in which each word appears at least once.
        present_in_reports = self.df[self.words] > 0  # Track whether reports are present in each group.
        fe_groups = self.df[self.fe_name]  # Store associated F.E. group of each report.
        appearances_in_fe_groups = pd.concat([present_in_reports, fe_groups], axis=1).groupby(self.fe_name).any().mean(axis=0)
        appearances_in_fe_groups = appearances_in_fe_groups.rename("frequency_over_fe_groups")

        # Calculate model metrics.
        values = []
        labels = []

        values.append("Number of Tokens Meeting Sample Restrictions: " + str(len(self.words)))
        values.append("Number of Documents: " + str(len(self.df)))
        values.append("Number of F.E. Groups: " + str(len(self.df[self.fe_name].unique())))

        metrics = pd.Series(values, index=self.words[:len(values)], name='Metrics')

        # Build results table
        self.results_table = pd.concat([pooled_likelihood_ratios,
                                        fe_likelihood_ratios,
                                        appearances_in_reports,
                                        appearances_in_fe_groups,
                                        metrics],
                                       axis=1)








