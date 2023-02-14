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
UNIGRAMS_OUTPUT = "/data/home/ashanmu1/Desktop/refereebias/output/R/elasticnet/unigrams/"
BIGRAMS_OUTPUT = "/data/home/ashanmu1/Desktop/refereebias/output/R/elasticnet/bigrams/"
TRIGRAMS_OUTPUT = "/data/home/ashanmu1/Desktop/refereebias/output/R/elasticnet/trigrams/"

# Run LASSO regressions of referee gender on R.
print("REGRESSING REFEREE GENDER ON R")
regularized_regress_on_entire_vocabulary(pickled_papers=PICKLED_PAPERS,
                                         pickled_reports=PICKLED_REPORTS,
                                         output_paths=[UNIGRAMS_OUTPUT, BIGRAMS_OUTPUT, TRIGRAMS_OUTPUT],
                                         seed=7,
                                         sklearn=True,
                                         adjust_reports_with_papers=False,
                                         normalize_documents_by_length=False,
                                         restrict_to_papers_with_mixed_gender_referees=True,
                                         balance_sample_by_categorical_column='_female_',
                                         y='_female_',
                                         model_names=["Unigram Elastic-Net on R", "Bigram Elastic-Net on R", "Trigram Elastic-Net on R"],
                                         method='elasticnet',
                                         logistic=True,
                                         standardize=True,
                                         log_transform='plus_one',
                                         alphas=np.linspace(0.001, .5, 50),
                                         l1_ratios=np.linspace(0.95, 1, 50),
                                         cv_folds=5,
                                         stratify=True,
                                         add_constant=True,
                                         metrics_list=["Average Testing Set Accuracy Across Folds",
                                                       "Portion of coefficients equal to 0",
                                                       "Constant",
                                                       "Optimal alpha",
                                                       "Optimal L1 Penalty Weight"])
