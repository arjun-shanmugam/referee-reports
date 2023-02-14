""" Defines useful utility functions for the running of models.
"""
from typing import List

from referee_report_dataset import RefereeReportDataset


def calculate_likelihood_ratios_on_entire_vocabulary(pickled_papers: str,
                                            pickled_reports: str,
                                            output_paths: List[str],
                                            seed: int,
                                            model_type: str):
    # Unigram Likelihood Ratio Model=================================================================================
    print("=======================================================================================================")
    print("UNIGRAMS")      
    unigrams = RefereeReportDataset(path_to_pickled_reports=pickled_reports,
                                            path_to_pickled_papers=pickled_papers,
                                            path_to_output=output_paths[0],
                                            seed=seed)                                        
    unigrams.build_df(adjust_reports_with_papers=False,
                                restrict_to_papers_with_mixed_gender_referees=True,
                                normalize_documents_by_length=False,
                                ngrams=1)
    unigrams.calculate_likelihood_ratios(model_name="Unigram " + model_type, model_type=model_type)
    unigrams.build_likelihood_results_table(model_name="Unigram " + model_type)
    print("Done!")
    # Bigram Likelihood Ratio Model==================================================================================
    print("=======================================================================================================")
    print("LIKELIHOOD RATIOS: BIGRAMS")  
    bigrams = RefereeReportDataset(path_to_pickled_reports=pickled_reports,
                                            path_to_pickled_papers=pickled_papers,
                                            path_to_output=output_paths[1],
                                            seed=seed) 
    bigrams.build_df(adjust_reports_with_papers=False,
                            normalize_documents_by_length=False,
                            restrict_to_papers_with_mixed_gender_referees=True,
                            ngrams=2)
    bigrams.calculate_likelihood_ratios(model_name="Bigram " + model_type, model_type=model_type)
    bigrams.build_likelihood_results_table(model_name="Bigram " + model_type)
    print("Done!")
    # Trigram Likelihood Ratio Model=================================================================================
    print("=======================================================================================================")
    print("LIKELIHOOD RATIOS: TRIGRAMS")  
    trigrams = RefereeReportDataset(path_to_pickled_reports=pickled_reports,
                                            path_to_pickled_papers=pickled_papers,
                                            path_to_output=output_paths[2],
                                            seed=seed)
    trigrams.build_df(adjust_reports_with_papers=False,
                                normalize_documents_by_length=False,
                                restrict_to_papers_with_mixed_gender_referees=True,
                                ngrams=3)
    trigrams.calculate_likelihood_ratios(model_name="Trigram " + model_type, model_type=model_type)
    trigrams.build_likelihood_results_table(model_name="Trigram " + model_type)
    print("Done!")
                                       
def regularized_regress_on_entire_vocabulary(pickled_papers: str,
                                            pickled_reports: str,
                                            output_paths: List[str],
                                            seed: int,
                                            sklearn: bool,
                                            adjust_reports_with_papers: bool,
                                            normalize_documents_by_length: bool,
                                            restrict_to_papers_with_mixed_gender_referees: str,
                                            balance_sample_by_categorical_column: str,
                                            y: str,
                                            model_names: List[str],
                                            method: str,
                                            logistic: bool,
                                            standardize: bool,
                                            log_transform: str,
                                            alphas,
                                            run_gender_placebos: bool,
                                            cv_folds: int,
                                            stratify: bool,
                                            add_constant: bool,
                                            l1_ratios=None,
                                            adjust_alpha: bool=False):
    # Unigram LASSO Regressions ====================================================================================
    print("=======================================================================================================")
    print("UNIGRAMS")  
    unigrams = RefereeReportDataset(path_to_pickled_reports=pickled_reports,
                                                    path_to_pickled_papers=pickled_papers,
                                                    path_to_output=output_paths[0],
                                                    seed=seed)
    unigrams.build_df(adjust_reports_with_papers=adjust_reports_with_papers,
                                    normalize_documents_by_length=normalize_documents_by_length,
                                    restrict_to_papers_with_mixed_gender_referees=restrict_to_papers_with_mixed_gender_referees,
                                    balance_sample_by_categorical_column=balance_sample_by_categorical_column,
                                    ngrams=1)
    if run_gender_placebos:                               
        unigrams.resample_variable_binomial(variable='_female_', p=0.5, ensure_balanced=False)                                
    unigrams.regularized_regress(y=y,
                                sklearn=sklearn,
                                X=unigrams.get_vocabulary(),
                                model_name=model_names[0],
                                method=method,
                                logistic=logistic,
                                standardize=standardize,
                                log_transform=log_transform,
                                alphas=alphas,
                                cv_folds=cv_folds,
                                stratify=stratify,
                                add_constant=add_constant,
                                l1_ratios=l1_ratios,
                                adjust_alpha=adjust_alpha)
    unigrams.build_regularized_results_table(model_name=model_names[0],
                                                )                                                                  
    print("Done!")
    # Bigram LASSO Regressions======================================================================================
    print("=======================================================================================================")
    print("BIGRAMS")  
    bigrams = RefereeReportDataset(path_to_pickled_reports=pickled_reports,
                                               path_to_pickled_papers=pickled_papers,
                                               path_to_output=output_paths[1],
                                               seed=seed)
    bigrams.build_df(adjust_reports_with_papers=adjust_reports_with_papers,
                                    normalize_documents_by_length=normalize_documents_by_length,
                                    restrict_to_papers_with_mixed_gender_referees=restrict_to_papers_with_mixed_gender_referees,
                                    balance_sample_by_categorical_column=balance_sample_by_categorical_column,
                                    ngrams=2)
    if run_gender_placebos:                               
        bigrams.resample_variable_binomial(variable='_female_', p=0.5, ensure_balanced=False)                                       
    bigrams.regularized_regress(y=y,
                                sklearn=sklearn,
                                X=bigrams.get_vocabulary(),
                                model_name=model_names[1],
                                method=method,
                                logistic=logistic,
                                standardize=standardize,
                                log_transform=log_transform,
                                alphas=alphas,
                                cv_folds=cv_folds,
                                stratify=stratify,
                                add_constant=add_constant,
                                l1_ratios=l1_ratios,
                                adjust_alpha=adjust_alpha)
    bigrams.build_regularized_results_table(model_name=model_names[1],
                                            )  
    print("Done!")
    # Trigram LASSO Model, Counts===============================================================
    print("=======================================================================================================")
    print("TRIGRAMS")  
    trigrams = RefereeReportDataset(path_to_pickled_reports=pickled_reports,
                                               path_to_pickled_papers=pickled_papers,
                                               path_to_output=output_paths[2],
                                               seed=seed)
    trigrams.build_df(adjust_reports_with_papers=adjust_reports_with_papers,
                                    normalize_documents_by_length=normalize_documents_by_length,
                                    restrict_to_papers_with_mixed_gender_referees=restrict_to_papers_with_mixed_gender_referees,
                                    balance_sample_by_categorical_column=balance_sample_by_categorical_column,
                                    ngrams=3)
    if run_gender_placebos:                               
        trigrams.resample_variable_binomial(variable='_female_', p=0.5, ensure_balanced=False)   
    trigrams.regularized_regress(y=y,
                                sklearn=sklearn,
                                X=trigrams.get_vocabulary(),
                                model_name=model_names[2],
                                method=method,
                                logistic=logistic,
                                standardize=standardize,
                                log_transform=log_transform,
                                alphas=alphas,
                                cv_folds=cv_folds,
                                stratify=stratify,
                                add_constant=add_constant,
                                l1_ratios=l1_ratios,
                                adjust_alpha=adjust_alpha)
    trigrams.build_regularized_results_table(model_name=model_names[2],
                                                )  
    print("Done!")
