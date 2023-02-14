"""Referee Bias NLP Project
    
    PIs: John Friedman, Amy Handlan, Nathan Hendren
    Author: Arjun Shanmugam
"""
import matplotlib.pyplot as plt
import numpy as np

from run_utils import regularized_regress_on_entire_vocabulary

plt.switch_backend('agg')  # Prevents an error from being thrown when parallelizing sk-learn cross validation.

# Constants
PICKLED_REPORTS = "/data/home/ashanmu1/Desktop/refereebias/intermediate_data/reports.txt.pkl"
PICKLED_PAPERS = "/data/home/ashanmu1/Desktop/refereebias/intermediate_data/papers.txt.pkl"
UNIGRAMS_OUTPUT = "/data/home/ashanmu1/Desktop/refereebias/output/NR/ridge/unigrams/"
BIGRAMS_OUTPUT = "/data/home/ashanmu1/Desktop/refereebias/output/NR/ridge/bigrams/"
TRIGRAMS_OUTPUT = "/data/home/ashanmu1/Desktop/refereebias/output/NR/ridge/trigrams/"

# Run ridge regressions of referee gender on NR.
print("REGRESSING REFEREE GENDER ON NR")
regularized_regress_on_entire_vocabulary(pickled_papers=PICKLED_PAPERS,
                                         pickled_reports=PICKLED_REPORTS,
                                         output_paths=[UNIGRAMS_OUTPUT, BIGRAMS_OUTPUT, TRIGRAMS_OUTPUT],
                                         seed=7,
                                         sklearn=False,
                                         adjust_reports_with_papers=False,
                                         normalize_documents_by_length=True,
                                         restrict_to_papers_with_mixed_gender_referees=True,
                                         balance_sample_by_categorical_column='_female_',
                                         run_gender_placebos=False, 
                                         y='_female_',
                                         model_names=["Unigram Ridge on NR", "Bigram Ridge on NR", "Trigram Ridge on NR"],
                                         method='ridge',
                                         logistic=True,
                                         standardize=True,
                                         log_transform='plus_one',
                                         alphas=np.linspace(0.0001, 5, 3000),
                                         cv_folds=5,
                                         stratify=True,
                                         add_constant=False)

print("PLACEBO REGRESSING REFEREE GENDER ON NR")
UNIGRAMS_OUTPUT = "/data/home/ashanmu1/Desktop/refereebias/output/NR/ridge/unigrams_placebo/"
BIGRAMS_OUTPUT = "/data/home/ashanmu1/Desktop/refereebias/output/NR/ridge/bigrams_placebo/"
TRIGRAMS_OUTPUT = "/data/home/ashanmu1/Desktop/refereebias/output/NR/ridge/trigrams_placebo/"
regularized_regress_on_entire_vocabulary(pickled_papers=PICKLED_PAPERS,
                                         pickled_reports=PICKLED_REPORTS,
                                         output_paths=[UNIGRAMS_OUTPUT, BIGRAMS_OUTPUT, TRIGRAMS_OUTPUT],
                                         seed=7,
                                         sklearn=False,
                                         adjust_reports_with_papers=False,
                                         normalize_documents_by_length=True,
                                         restrict_to_papers_with_mixed_gender_referees=True,
                                         balance_sample_by_categorical_column='_female_',
                                         run_gender_placebos=True,  # Randomly re-assign gender.
                                         y='_female_',
                                         model_names=["Unigram Ridge on NR (Placebo)",
                                                      "Bigram Ridge on NR (Placebo)",
                                                      "Trigram Ridge on NR (Placebo)"],
                                         method='ridge',
                                         logistic=True,
                                         standardize=True,
                                         log_transform='plus_one',
                                         alphas=np.linspace(0.0001, 5, 3000),
                                         cv_folds=5,
                                         stratify=True,
                                         add_constant=False)
