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
UNIGRAMS_OUTPUT = "/data/home/ashanmu1/Desktop/refereebias/output/R_tilde/LASSO/unigrams/"
BIGRAMS_OUTPUT = "/data/home/ashanmu1/Desktop/refereebias/output/R_tilde/LASSO/bigrams/"
TRIGRAMS_OUTPUT = "/data/home/ashanmu1/Desktop/refereebias/output/R_tilde/LASSO/trigrams/"

# Run LASSO regressions of referee gender on R-tilde
"""print("REGRESSING REFEREE GENDER ON R-TILDE")
regularized_regress_on_entire_vocabulary(pickled_papers=PICKLED_PAPERS,
                                         pickled_reports=PICKLED_REPORTS,
                                         output_paths=[UNIGRAMS_OUTPUT, BIGRAMS_OUTPUT, TRIGRAMS_OUTPUT],
                                         seed=7,
                                         adjust_reports_with_papers=True,
                                         normalize_documents_by_length=False,
                                         restrict_to_papers_with_mixed_gender_referees=True,
                                         balance_sample_by_categorical_column='_female_',
                                         y='_female_',
                                         sklearn=False,
                                         run_gender_placebos=False,
                                         model_names=["Unigram LASSO on R-tilde", "Bigram LASSO on R-tilde", "Trigram LASSO on R-tilde"],
                                         method='LASSO',
                                         logistic=True,
                                         standardize=True,
                                         log_transform='plus_one',
                                         alphas=np.linspace(0.001, 0.25, 1000),
                                         cv_folds=5,
                                         stratify=True,
                                         add_constant=False,
                                         metrics_list=["Average Testing Set Accuracy Across Folds",
                                                       "Portion of coefficients equal to 0",
                                                       "Optimal alpha"])
print("REGRESSING PLACEBO REFEREE GENDER ON R-TILDE")
UNIGRAMS_OUTPUT = "/data/home/ashanmu1/Desktop/refereebias/output/R_tilde/LASSO/unigrams_placebo/"
BIGRAMS_OUTPUT = "/data/home/ashanmu1/Desktop/refereebias/output/R_tilde/LASSO/bigrams_placebo/"
TRIGRAMS_OUTPUT = "/data/home/ashanmu1/Desktop/refereebias/output/R_tilde/LASSO/trigrams_placebo/"
regularized_regress_on_entire_vocabulary(pickled_papers=PICKLED_PAPERS,
                                         pickled_reports=PICKLED_REPORTS,
                                         output_paths=[UNIGRAMS_OUTPUT, BIGRAMS_OUTPUT, TRIGRAMS_OUTPUT],
                                         seed=7,
                                         adjust_reports_with_papers=True,
                                         normalize_documents_by_length=False,
                                         restrict_to_papers_with_mixed_gender_referees=True,
                                         balance_sample_by_categorical_column='_female_',
                                         y='_female_',
                                         sklearn=False,
                                         run_gender_placebos=True,
                                         model_names=["Unigram LASSO on R-tilde (Placebo)", "Bigram LASSO on R-tilde (Placebo)", "Trigram LASSO on R-tilde (Placebo)"],
                                         method='LASSO',
                                         logistic=True,
                                         standardize=True,
                                         log_transform='plus_one',
                                         alphas=np.linspace(0.001, 0.25, 1000),
                                         cv_folds=5,
                                         stratify=True,
                                         add_constant=False,
                                         metrics_list=["Average Testing Set Accuracy Across Folds",
                                                       "Portion of coefficients equal to 0",
                                                       "Optimal alpha"])
                                                       """

print("REGRESSING REFEREE GENDER ON R-TILDE WITH ADJUSTED ALPHA")
UNIGRAMS_OUTPUT = "/data/home/ashanmu1/Desktop/refereebias/output/R_tilde/LASSO/unigrams_adjusted_alpha/"
BIGRAMS_OUTPUT = "/data/home/ashanmu1/Desktop/refereebias/output/R_tilde/LASSO/bigrams_adjusted_alpha/"
TRIGRAMS_OUTPUT = "/data/home/ashanmu1/Desktop/refereebias/output/R_tilde/LASSO/trigrams_adjusted_alpha/"
regularized_regress_on_entire_vocabulary(pickled_papers=PICKLED_PAPERS,
                                         pickled_reports=PICKLED_REPORTS,
                                         output_paths=[UNIGRAMS_OUTPUT, BIGRAMS_OUTPUT, TRIGRAMS_OUTPUT],
                                         seed=7,
                                         adjust_reports_with_papers=True,
                                         normalize_documents_by_length=False,
                                         restrict_to_papers_with_mixed_gender_referees=True,
                                         balance_sample_by_categorical_column='_female_',
                                         y='_female_',
                                         sklearn=False,
                                         run_gender_placebos=False,
                                         model_names=["Unigram LASSO on R-tilde (Adjusted Alpha)", "Bigram LASSO on R-tilde (Adjusted Alpha)", "Trigram LASSO on R-tilde (Adjusted Alpha)"],
                                         method='LASSO',
                                         logistic=True,
                                         standardize=True,
                                         log_transform='plus_one',
                                         alphas=np.linspace(0.001, 0.25, 1000),
                                         cv_folds=5,
                                         stratify=True,
                                         add_constant=False,
                                         metrics_list=["Average Testing Set Accuracy Across Folds",
                                                       "Portion of coefficients equal to 0",
                                                       "Optimal alpha"],
                                        adjust_alpha=True)
