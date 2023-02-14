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
UNIGRAMS_OUTPUT = "/data/home/ashanmu1/Desktop/refereebias/output/R/LASSO/unigrams/"
BIGRAMS_OUTPUT = "/data/home/ashanmu1/Desktop/refereebias/output/R/LASSO/bigrams/"
TRIGRAMS_OUTPUT = "/data/home/ashanmu1/Desktop/refereebias/output/R/LASSO/trigrams/"


# Run LASSO regressions of referee gender on R.
print("REGRESSING REFEREE GENDER ON R")
regularized_regress_on_entire_vocabulary(pickled_papers=PICKLED_PAPERS,
                                         pickled_reports=PICKLED_REPORTS,
                                         output_paths=[UNIGRAMS_OUTPUT, BIGRAMS_OUTPUT, TRIGRAMS_OUTPUT],
                                         seed=7,
                                         sklearn=False,
                                         adjust_reports_with_papers=False,
                                         normalize_documents_by_length=False,
                                         restrict_to_papers_with_mixed_gender_referees=True,
                                         balance_sample_by_categorical_column='_female_',
                                         run_gender_placebos=False, 
                                         y='_female_',
                                         model_names=["Unigram LASSO on R", "Bigram LASSO on R", "Trigram LASSO on R"],
                                         method='LASSO',
                                         logistic=True,
                                         standardize=True,
                                         log_transform='plus_one',
                                         alphas=np.linspace(0.0001, .25, 1000),
                                         cv_folds=5,
                                         stratify=True,
                                         add_constant=False)

print("PLACEBO REGRESSING REFEREE GENDER ON R")
UNIGRAMS_OUTPUT = "/data/home/ashanmu1/Desktop/refereebias/output/R/LASSO/unigrams_placebo/"
BIGRAMS_OUTPUT = "/data/home/ashanmu1/Desktop/refereebias/output/R/LASSO/bigrams_placebo/"
TRIGRAMS_OUTPUT = "/data/home/ashanmu1/Desktop/refereebias/output/R/LASSO/trigrams_placebo/"
regularized_regress_on_entire_vocabulary(pickled_papers=PICKLED_PAPERS,
                                         pickled_reports=PICKLED_REPORTS,
                                         output_paths=[UNIGRAMS_OUTPUT, BIGRAMS_OUTPUT, TRIGRAMS_OUTPUT],
                                         seed=7,
                                         sklearn=False,
                                         adjust_reports_with_papers=False,
                                         normalize_documents_by_length=False,
                                         restrict_to_papers_with_mixed_gender_referees=True,
                                         balance_sample_by_categorical_column='_female_',
                                         run_gender_placebos=True,  # Randomly re-assign gender.
                                         y='_female_',
                                         model_names=["Unigram LASSO on R (Placebo)",
                                                      "Bigram LASSO on R (Placebo)",
                                                      "Trigram LASSO on R (Placebo)"],
                                         method='LASSO',
                                         logistic=True,
                                         standardize=True,
                                         log_transform='plus_one',
                                         alphas=np.linspace(0.0001, .25, 1000),
                                         cv_folds=5,
                                         stratify=True,
                                         add_constant=False)

print("REGRESSING REFEREE GENDER ON R WITH ADJUSTED ALPHA")                                                      
UNIGRAMS_OUTPUT = "/data/home/ashanmu1/Desktop/refereebias/output/R/LASSO/unigrams_adjusted_alpha/"
BIGRAMS_OUTPUT = "/data/home/ashanmu1/Desktop/refereebias/output/R/LASSO/bigrams_adjusted_alpha/"
TRIGRAMS_OUTPUT = "/data/home/ashanmu1/Desktop/refereebias/output/R/LASSO/trigrams_adjusted_alpha/"
regularized_regress_on_entire_vocabulary(pickled_papers=PICKLED_PAPERS,
                                         pickled_reports=PICKLED_REPORTS,
                                         output_paths=[UNIGRAMS_OUTPUT, BIGRAMS_OUTPUT, TRIGRAMS_OUTPUT],
                                         seed=7,
                                         sklearn=False,
                                         adjust_reports_with_papers=False,
                                         normalize_documents_by_length=False,
                                         restrict_to_papers_with_mixed_gender_referees=True,
                                         balance_sample_by_categorical_column='_female_',
                                         run_gender_placebos=False,  # Randomly re-assign gender.
                                         y='_female_',
                                         model_names=["Unigram LASSO on R (Adjusted Alpha)",
                                                      "Bigram LASSO on R (Adjusted Alpha)",
                                                      "Trigram LASSO on R (Adjusted Alpha)"],
                                         method='LASSO',
                                         logistic=True,
                                         standardize=True,
                                         log_transform='plus_one',
                                         alphas=np.linspace(0.0001, .25, 1000),
                                         cv_folds=5,
                                         stratify=True,
                                         add_constant=False,
                                         adjust_alpha=True)
