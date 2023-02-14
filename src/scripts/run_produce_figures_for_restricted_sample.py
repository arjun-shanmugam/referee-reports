"""Referee Bias NLP Project
    
    PIs: John Friedman, Amy Handlan, Nathan Hendren
    Author: Arjun Shanmugam
"""
import matplotlib.pyplot as plt

from referee_report_dataset import RefereeReportDataset

plt.switch_backend('agg')  # Prevents an error from being thrown when parallelizing sk-learn cross validation.

# Constants
PICKLED_REPORTS = "/data/home/ashanmu1/Desktop/refereebias/intermediate_data/reports.txt.pkl"
PICKLED_PAPERS = "/data/home/ashanmu1/Desktop/refereebias/intermediate_data/papers.txt.pkl"
OUTPUT = "/data/home/ashanmu1/Desktop/refereebias/output/restricted_sample_summary_statistics_1_grams/"


# Calculate likelihood ratios.
dataset = RefereeReportDataset(path_to_pickled_reports=PICKLED_REPORTS,
                               path_to_pickled_papers=PICKLED_PAPERS,
                               path_to_output=OUTPUT,
                               seed=7)
dataset.build_df(adjust_reports_with_papers=False,
                 normalize_documents_by_length=False,
                 restrict_to_papers_with_mixed_gender_referees=True,
                 ngrams=1)
dataset._produce_summary_statistics(
                                    adjust_reports_with_papers=False,
                                    normalize_documents_by_length=False)


OUTPUT = "/data/home/ashanmu1/Desktop/refereebias/output/restricted_sample_summary_statistics_2_grams/"
# Calculate likelihood ratios.
dataset = RefereeReportDataset(path_to_pickled_reports=PICKLED_REPORTS,
                               path_to_pickled_papers=PICKLED_PAPERS,
                               path_to_output=OUTPUT,
                               seed=7)
dataset.build_df(adjust_reports_with_papers=False,
                 normalize_documents_by_length=False,
                 restrict_to_papers_with_mixed_gender_referees=True,
                 ngrams=2)
dataset._produce_summary_statistics(
                                    adjust_reports_with_papers=False,
                                    normalize_documents_by_length=False)

OUTPUT = "/data/home/ashanmu1/Desktop/refereebias/output/restricted_sample_summary_statistics_3_grams/"
# Calculate likelihood ratios.
dataset = RefereeReportDataset(path_to_pickled_reports=PICKLED_REPORTS,
                               path_to_pickled_papers=PICKLED_PAPERS,
                               path_to_output=OUTPUT,
                               seed=7)
dataset.build_df(adjust_reports_with_papers=False,
                 normalize_documents_by_length=False,
                 restrict_to_papers_with_mixed_gender_referees=True,
                 ngrams=3)
dataset._produce_summary_statistics(
                                    adjust_reports_with_papers=False,
                                    normalize_documents_by_length=False,)

