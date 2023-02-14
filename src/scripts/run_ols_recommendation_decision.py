"""Referee Bias NLP Project
    
    PIs: John Friedman, Amy Handlan, Nathan Hendren
    Author: Arjun Shanmugam
"""
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from sklearn import datasets
from stargazer.stargazer import Stargazer

from referee_report_dataset import RefereeReportDataset
from run_utils import calculate_likelihood_ratios_on_entire_vocabulary

plt.switch_backend('agg')  # Prevents an error from being thrown when parallelizing sk-learn cross validation.

# Constants
PICKLED_REPORTS = "/data/home/ashanmu1/Desktop/refereebias/intermediate_data/reports.txt.pkl"
PICKLED_PAPERS = "/data/home/ashanmu1/Desktop/refereebias/intermediate_data/papers.txt.pkl"
OLS_RECOMMENDATION_DECISION_OUTPUT = "/data/home/ashanmu1/Desktop/refereebias/output/ols_recommendation_decision/"




# Calculate likelihood ratios.
dataset = RefereeReportDataset(path_to_pickled_reports=PICKLED_REPORTS,
                               path_to_pickled_papers=PICKLED_PAPERS,
                               path_to_output=OLS_RECOMMENDATION_DECISION_OUTPUT,
                               seed=7)
dataset.build_df(adjust_reports_with_papers=False,
                 normalize_documents_by_length=False,
                 restrict_to_papers_with_mixed_gender_referees=True)
pd.reset_option("display.max_colwidth")
print("REGRESSING Decision: Reject on Referee Gender")
dataset.add_column(pd.get_dummies(dataset._df['_decision_'], prefix="Decision", drop_first=True, prefix_sep=": ").loc[:, 'Decision: Revise'])
dataset.ols_regress(y='Decision: Revise',
                    X=['_female_'],
                    model_name="OLS Regression of Decision: Revise on Referee Gender",
                    add_constant=True,
                    logistic=None,
                    log_transform=None,
                    standardize=False)

print("REGRESSING REFEREE GENDER ON RECOMMENDATION")
recommendation_dummy_columns = pd.get_dummies(dataset._df['_recommendation_'], prefix="Recommendation", drop_first=True, prefix_sep=": ")
for column in recommendation_dummy_columns:
    dataset.add_column(recommendation_dummy_columns[column])
    dataset.ols_regress(y=column,
                        X=['_female_'],
                        model_name="OLS Regression of " + column + " on Referee Gender",
                        add_constant=True,
                        logistic=False,
                        log_transform=None,
                        standardize=False)


dataset.build_ols_results_table(filename="ols_regression_on_recomendation_decision",
                                requested_models=["OLS Regression of Decision: Revise on Referee Gender",
                                        "OLS Regression of Recommendation: Minor Revision on Referee Gender",
                                        "OLS Regression of Recommendation: Major Revision on Referee Gender",
                                        "OLS Regression of Recommendation: Reject on Referee Gender"],
                                title="OLS Regressions of Decision and Recommendation Dummies on Referee Gender",
                                column_titles=[["Decision: Revise", "Recommendation: Minor Revision", "Recommendation: Major Revision", "Recommendation: Reject"],[1, 1, 1, 1]],
                                show_confidence_intervals=False,
                                show_degrees_of_freedom=False,
                                lines_to_add=None,
                                custom_notes=None)
