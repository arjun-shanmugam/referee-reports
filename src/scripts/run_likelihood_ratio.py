"""Referee Bias NLP Project
    
    PIs: John Friedman, Amy Handlan, Nathan Hendren
    Author: Arjun Shanmugam
"""
import matplotlib.pyplot as plt

from run_utils import calculate_likelihood_ratios_on_entire_vocabulary

plt.switch_backend('agg')  # Prevents an error from being thrown when parallelizing sk-learn cross validation.

# Constants
PICKLED_REPORTS = "/data/home/ashanmu1/Desktop/refereebias/intermediate_data/reports.txt.pkl"
PICKLED_PAPERS = "/data/home/ashanmu1/Desktop/refereebias/intermediate_data/papers.txt.pkl"
"""
UNIGRAMS_OUTPUT = "/data/home/ashanmu1/Desktop/refereebias/output/likelihood_ratios/unigrams/"
BIGRAMS_OUTPUT = "/data/home/ashanmu1/Desktop/refereebias/output/likelihood_ratios/bigrams/"
TRIGRAMS_OUTPUT = "/data/home/ashanmu1/Desktop/refereebias/output/likelihood_ratios/trigrams/"

# Calculate likelihood ratios.
print("CALCULATING LIKELIHOOD RATIOS (MODEL 0)")
calculate_likelihood_ratios_on_entire_vocabulary(pickled_papers=PICKLED_PAPERS,
                                                 pickled_reports=PICKLED_REPORTS,
                                                 output_paths=[UNIGRAMS_OUTPUT, BIGRAMS_OUTPUT, TRIGRAMS_OUTPUT],
                                                 _seed=7,
                                                 _model_type="Likelihood Ratio Model 0")



UNIGRAMS_OUTPUT = "/data/home/ashanmu1/Desktop/refereebias/output/likelihood_ratios_model_1/unigrams/"
BIGRAMS_OUTPUT = "/data/home/ashanmu1/Desktop/refereebias/output/likelihood_ratios_model_1/bigrams/"
TRIGRAMS_OUTPUT = "/data/home/ashanmu1/Desktop/refereebias/output/likelihood_ratios_model_1/trigrams/"
print("CALCULATING LIKELIHOOD RATIOS (MODEL 1)")
calculate_likelihood_ratios_on_entire_vocabulary(pickled_papers=PICKLED_PAPERS,
                                                 pickled_reports=PICKLED_REPORTS,
                                                 output_paths=[UNIGRAMS_OUTPUT, BIGRAMS_OUTPUT, TRIGRAMS_OUTPUT],
                                                 _seed=7,
                                                 _model_type="Likelihood Ratio Model 1")


UNIGRAMS_OUTPUT = "/data/home/ashanmu1/Desktop/refereebias/output/likelihood_ratios_model_2/unigrams/"
BIGRAMS_OUTPUT = "/data/home/ashanmu1/Desktop/refereebias/output/likelihood_ratios_model_2/bigrams/"
TRIGRAMS_OUTPUT = "/data/home/ashanmu1/Desktop/refereebias/output/likelihood_ratios_model_2/trigrams/"
print("CALCULATING LIKELIHOOD RATIOS (MODEL 2)")
calculate_likelihood_ratios_on_entire_vocabulary(pickled_papers=PICKLED_PAPERS,
                                                 pickled_reports=PICKLED_REPORTS,
                                                 output_paths=[UNIGRAMS_OUTPUT, BIGRAMS_OUTPUT, TRIGRAMS_OUTPUT],
                                                 _seed=7,
                                                 _model_type="Likelihood Ratio Model 2")


                                                 """


UNIGRAMS_OUTPUT = "/data/home/ashanmu1/Desktop/refereebias/output/likelihood_ratios_model_3/unigrams/"
BIGRAMS_OUTPUT = "/data/home/ashanmu1/Desktop/refereebias/output/likelihood_ratios_model_3/bigrams/"
TRIGRAMS_OUTPUT = "/data/home/ashanmu1/Desktop/refereebias/output/likelihood_ratios_model_3/trigrams/"
print("CALCULATING LIKELIHOOD RATIOS (MODEL 3)")
calculate_likelihood_ratios_on_entire_vocabulary(pickled_papers=PICKLED_PAPERS,
                                                 pickled_reports=PICKLED_REPORTS,
                                                 output_paths=[UNIGRAMS_OUTPUT, BIGRAMS_OUTPUT, TRIGRAMS_OUTPUT],
                                                 seed=7,
                                                 model_type="Likelihood Ratio Model 3")


