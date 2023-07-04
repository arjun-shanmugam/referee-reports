"""Defines the Regression class, which implements functionality that is common to OLSRegression, Regularized Regression, and PanelRegression.
"""
import warnings
from typing import List
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression, Ridge
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler


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
