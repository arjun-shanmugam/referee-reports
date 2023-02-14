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
ENTIRE_SAMPLE_SUMMARY_STATISTICS = "/data/home/ashanmu1/Desktop/refereebias/output/entire_sample_summary_statistics/"




# Calculate likelihood ratios.
dataset = RefereeReportDataset(path_to_pickled_reports=PICKLED_REPORTS,
                               path_to_pickled_papers=PICKLED_PAPERS,
                               path_to_output=ENTIRE_SAMPLE_SUMMARY_STATISTICS,
                               seed=7)
dataset.build_df(adjust_reports_with_papers=False,
                 normalize_documents_by_length=False,
                 restrict_to_papers_with_mixed_gender_referees=False)
print(dataset.df['_paper_'].value_counts())
print(len(dataset.df['_paper_'].value_counts()))
print(len(dataset.df['_refnum_']))