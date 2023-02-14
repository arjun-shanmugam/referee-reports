"""Referee Bias NLP Project
    
    PIs: John Friedman, Amy Handlan, Nathan Hendren
    Author: Arjun Shanmugam
"""
import matplotlib.pyplot as plt
import numpy as np

from run_utils import regularized_regress_on_entire_vocabulary

plt.switch_backend('agg')  # Prevents an error from being thrown when parallelizing.

# Constants
PICKLED_REPORTS = "/data/home/ashanmu1/Desktop/refereebias/intermediate_data/reports.txt.pkl"
PICKLED_PAPERS = "/data/home/ashanmu1/Desktop/refereebias/intermediate_data/papers.txt.pkl"
UNIGRAMS_OUTPUT = "/data/home/ashanmu1/Desktop/refereebias/output/NR/LASSO/unigrams/"
BIGRAMS_OUTPUT = "/data/home/ashanmu1/Desktop/refereebias/output/NR/LASSO/bigrams/"
TRIGRAMS_OUTPUT = "../../output/NR/LASSO/trigrams/"

# Run LASSO regressions of referee gender on NR
print("REGRESSING REFEREE GENDER ON NR")
regularized_regress_on_entire_vocabulary(pickled_papers=PICKLED_PAPERS,
                                         pickled_reports=PICKLED_REPORTS,
                                         output_paths=[UNIGRAMS_OUTPUT, BIGRAMS_OUTPUT, TRIGRAMS_OUTPUT],
                                         seed=7,
                                         adjust_reports_with_papers=False,
                                         sklearn=False,
                                         normalize_documents_by_length=True,
                                         restrict_to_papers_with_mixed_gender_referees=True,
                                         run_gender_placebos=False,
                                         balance_sample_by_categorical_column='_female_',
                                         y='_female_',
                                         model_names=["Unigram LASSO on NR", "Bigram LASSO on NR", "Trigram LASSO on NR"],
                                         method='LASSO',
                                         logistic=True,
                                         standardize=True,
                                         log_transform='plus_one',
                                         alphas=np.linspace(0.0001, .25, 1000),
                                         cv_folds=5,
                                         stratify=True,
                                         add_constant=False)


print("REGRESSING PLACEBO GENDER ON NR")                                                   
UNIGRAMS_OUTPUT = "/data/home/ashanmu1/Desktop/refereebias/output/NR/LASSO/unigrams_placebo/"
BIGRAMS_OUTPUT = "/data/home/ashanmu1/Desktop/refereebias/output/NR/LASSO/bigrams_placebo/"
TRIGRAMS_OUTPUT = "/data/home/ashanmu1/Desktop/refereebias/output/NR/LASSO/trigrams_placebo/"
regularized_regress_on_entire_vocabulary(pickled_papers=PICKLED_PAPERS,
                                         pickled_reports=PICKLED_REPORTS,
                                         output_paths=[UNIGRAMS_OUTPUT, BIGRAMS_OUTPUT, TRIGRAMS_OUTPUT],
                                         seed=7,
                                         adjust_reports_with_papers=False,
                                         sklearn=False,
                                         normalize_documents_by_length=True,
                                         restrict_to_papers_with_mixed_gender_referees=True,
                                         run_gender_placebos=True,
                                         balance_sample_by_categorical_column='_female_',
                                         y='_female_',
                                         model_names=["Unigram LASSO on NR (Placebo)", "Bigram LASSO on NR (Placebo)", "Trigram LASSO on NR (Placebo)"],
                                         method='LASSO',
                                         logistic=True,
                                         standardize=True,
                                         log_transform='plus_one',
                                         alphas=np.linspace(0.0001, .25, 1000),
                                         cv_folds=5,
                                         stratify=True,
                                         add_constant=False)

print("REGRESSING REFEREE GENDER ON NR WITH ADJUSTED ALPHA")
UNIGRAMS_OUTPUT = "/data/home/ashanmu1/Desktop/refereebias/output/NR/LASSO/unigrams_adjusted_alpha/"
BIGRAMS_OUTPUT = "/data/home/ashanmu1/Desktop/refereebias/output/NR/LASSO/bigrams_adjusted_alpha/"
TRIGRAMS_OUTPUT = "/data/home/ashanmu1/Desktop/refereebias/output/NR/LASSO/trigrams_adjusted_alpha/"
regularized_regress_on_entire_vocabulary(pickled_papers=PICKLED_PAPERS,
                                         pickled_reports=PICKLED_REPORTS,
                                         output_paths=[UNIGRAMS_OUTPUT, BIGRAMS_OUTPUT, TRIGRAMS_OUTPUT],
                                         seed=7,
                                         adjust_reports_with_papers=False,
                                         sklearn=False,
                                         normalize_documents_by_length=True,
                                         restrict_to_papers_with_mixed_gender_referees=True,
                                         run_gender_placebos=False,
                                         balance_sample_by_categorical_column='_female_',
                                         y='_female_',
                                         model_names=["Unigram LASSO on NR (Adjusted Alpha)", "Bigram LASSO on NR (Adjusted Alpha)", "Trigram LASSO on NR (Adjusted Alpha)"],
                                         method='LASSO',
                                         logistic=True,
                                         standardize=True,
                                         log_transform='plus_one',
                                         alphas=np.linspace(0.0001, .25, 1000),
                                         cv_folds=5,
                                         stratify=True,
                                         add_constant=False,
                                         adjust_alpha=True)
