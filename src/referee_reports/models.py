"""Defines the Regression class, which implements functionality that is common to OLSRegression, Regularized Regression, and PanelRegression.
"""
from typing import Any, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class Regression:
    y: pd.Series
    X: pd.DataFrame
    standardize: bool
    log_transform: bool
    add_constant: bool
    dummy_variables: List[str]
    model_type: str
    results_table: Any

    def get_model_type(self):
        return self.model_type

    def get_results_table(self):
        return self.results_table

    def _preprocess_inputs(self, standardize_generated_dummies):
        # Store numeric and categorical columns separately.
        categorical_df = self.X.select_dtypes(exclude=['number'])
        numeric_df = self.X.select_dtypes(include=['number'])  # Setting include to 'number' selects all numeric datatypes.

        # Check that we have accounted for all columns in the provided input.
        assert set(categorical_df.columns.tolist()).union(set(numeric_df.columns.tolist())) == set(
            self.X.columns.tolist()), "Union of categorical columns and numeric columns does not equal the set of all input columns."

        if self.log_transform is not None and not numeric_df.empty:
            if self.log_transform == 'plus_one':
                numeric_df = np.log(numeric_df + 1)
            elif self.log_transform == 'regular':
                numeric_df = np.log(numeric_df)
            else:
                raise ValueError("Please specify either 'plus_one' or 'regular' for the type of log transformation desired.")

        # If requested, standardize numeric variables using a Scikit-Learn Standard Scaler.
        if self.standardize and not numeric_df.empty:
            numeric_df = pd.DataFrame(StandardScaler().fit_transform(numeric_df), columns=numeric_df.columns, index=numeric_df.index)

        # Generate dummies for categorical variables. For each categorical variable, dummies are generated for each category except one.
        if not categorical_df.empty:
            categorical_df = pd.get_dummies(categorical_df, prefix=categorical_df.columns, prefix_sep=": ", drop_first=True, dtype='int64')
            if self.standardize and standardize_generated_dummies:
                categorical_df = pd.DataFrame(StandardScaler().fit_transform(categorical_df), columns=categorical_df.columns)

        # Keep track of dummy variable names.
        self.dummy_variables = categorical_df.columns.tolist()

        # Set X to the preprocessed input.
        self.X = pd.concat([categorical_df, numeric_df], axis=1)
