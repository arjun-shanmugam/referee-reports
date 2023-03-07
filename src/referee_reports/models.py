"""Defines the Regression class, which implements functionality that is common to OLSRegression, Regularized Regression, and PanelRegression.
"""
from typing import List
from collections import Iterable
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
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
                 logistic: bool,
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

        self._logistic = logistic
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

    def fit(self):
        """Fit an OLS model.

        Uses StatsModels to fit an OLS model.
        """
        self._preprocess_inputs()

        if self._logistic:
            raise NotImplementedError("Non-regularized logistic regression has not been implemented.")
            # self.results_table = LogitOddsWrapper(sm.Logit(self.y.astype(float), self.X.astype(float)).fit(maxiter=1000))
        else:
            self._results_table = sm.OLS(self._y_data.astype(float), self._X_data.astype(float)).fit()

class RegularizedRegression(Regression):
    def fit(self, method: str, stratify: bool, cv_folds: int, seed: int, alphas: Iterable):
        if method == 'LASSO':
            pass
        elif method == 'ridge':
            pass
        elif method == 'elasticnet':
            pass
        else:
            raise ValueError("Please specify either \'LASSO\', \'ridge\', or \'elasticnet\'.")

        self._preprocess_inputs()

        if self.stratify:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        else:
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)


