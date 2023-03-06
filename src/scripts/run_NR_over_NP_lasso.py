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
UNIGRAMS_OUTPUT = "/data/home/ashanmu1/Desktop/refereebias/output/NR_over_NP/LASSO/unigrams/"
BIGRAMS_OUTPUT = "/data/home/ashanmu1/Desktop/refereebias/output/NR_over_NP/LASSO/bigrams/"
TRIGRAMS_OUTPUT = "/data/home/ashanmu1/Desktop/refereebias/output/NR_over_NP/LASSO/trigrams/"

# Run LASSO regressions of referee gender on NR/NP.
"""print("REGRESSING REFEREE GENDER ON NR/NP")
regularized_regress_on_entire_vocabulary(pickled_papers=PICKLED_PAPERS,
                                         pickled_reports=PICKLED_REPORTS,
                                         output_paths=[UNIGRAMS_OUTPUT, BIGRAMS_OUTPUT, TRIGRAMS_OUTPUT],
                                         _seed=7,
                                         sklearn=False,
                                         run_gender_placebos=False,
                                         adjust_reports_with_papers=True,
                                         normalize_documents_by_length=True,
                                         restrict_to_papers_with_mixed_gender_referees=True,
                                         balance_sample_by_categorical_column='_female_',
                                         _y_data='_female_',
                                         model_names=["Unigram LASSO on NR over NP", "Bigram LASSO on NR over NP", "Trigram LASSO on NR over NP"],
                                         method='LASSO',
                                         logistic=True,
                                         _standardize=True,
                                         _log_transform='regular',
                                         alphas=np.linspace(0.001, .25, 1000),
                                         cv_folds=5,
                                         stratify=True,
                                         _add_constant=False,
                                         metrics_list=["Average Testing Set Accuracy Across Folds",
                                                       "Portion of coefficients equal to 0",
                                                       "Optimal alpha"])
print("REGRESSING PLACEBO REFEREE GENDER ON NR/NP")                                                      
UNIGRAMS_OUTPUT = "/data/home/ashanmu1/Desktop/refereebias/output/NR_over_NP/LASSO/unigrams_placebo/"
BIGRAMS_OUTPUT = "/data/home/ashanmu1/Desktop/refereebias/output/NR_over_NP/LASSO/bigrams_placebo/"
TRIGRAMS_OUTPUT = "/data/home/ashanmu1/Desktop/refereebias/output/NR_over_NP/LASSO/trigrams_placebo/"
regularized_regress_on_entire_vocabulary(pickled_papers=PICKLED_PAPERS,
                                         pickled_reports=PICKLED_REPORTS,
                                         output_paths=[UNIGRAMS_OUTPUT, BIGRAMS_OUTPUT, TRIGRAMS_OUTPUT],
                                         _seed=7,
                                         sklearn=False,
                                         run_gender_placebos=True,
                                         adjust_reports_with_papers=True,
                                         normalize_documents_by_length=True,
                                         restrict_to_papers_with_mixed_gender_referees=True,
                                         balance_sample_by_categorical_column='_female_',
                                         _y_data='_female_',
                                         model_names=["Unigram LASSO on NR over NP (Placebos)", "Bigram LASSO on NR over NP (Placebos)", "Trigram LASSO on NR over NP (Placebos)"],
                                         method='LASSO',
                                         logistic=True,
                                         _standardize=True,
                                         _log_transform='regular',
                                         alphas=np.linspace(0.001, .25, 1000),
                                         cv_folds=5,
                                         stratify=True,
                                         _add_constant=False,
                                         metrics_list=["Average Testing Set Accuracy Across Folds",
                                                       "Portion of coefficients equal to 0",
                                                       "Optimal alpha"])"""
                                                       
print("REGRESSING REFEREE GENDER ON NR/NP WITH ADJUSTED ALPHA")
UNIGRAMS_OUTPUT = "/data/home/ashanmu1/Desktop/refereebias/output/NR_over_NP/LASSO/unigrams_adjusted_alpha/"
BIGRAMS_OUTPUT = "/data/home/ashanmu1/Desktop/refereebias/output/NR_over_NP/LASSO/bigrams_adjusted_alpha/"
TRIGRAMS_OUTPUT = "/data/home/ashanmu1/Desktop/refereebias/output/NR_over_NP/LASSO/trigrams_adjusted_alpha/"
regularized_regress_on_entire_vocabulary(pickled_papers=PICKLED_PAPERS,
                                         pickled_reports=PICKLED_REPORTS,
                                         output_paths=[UNIGRAMS_OUTPUT, BIGRAMS_OUTPUT, TRIGRAMS_OUTPUT],
                                         seed=7,
                                         sklearn=False,
                                         run_gender_placebos=False,
                                         adjust_reports_with_papers=True,
                                         normalize_documents_by_length=True,
                                         restrict_to_papers_with_mixed_gender_referees=True,
                                         balance_sample_by_categorical_column='_female_',
                                         y='_female_',
                                         model_names=["Unigram LASSO on NR over NP (Adjusted Alpha)", "Bigram LASSO on NR over NP (Adjusted Alpha)", "Trigram LASSO on NR over NP (Adjusted Alpha)"],
                                         method='LASSO',
                                         logistic=True,
                                         standardize=True,
                                         log_transform='regular',
                                         alphas=np.linspace(0.001, .25, 1000),
                                         cv_folds=5,
                                         stratify=True,
                                         add_constant=False,
                                         metrics_list=["Average Testing Set Accuracy Across Folds",
                                                       "Portion of coefficients equal to 0",
                                                       "Optimal alpha"],
                                         adjust_alpha=True)

                                                       
