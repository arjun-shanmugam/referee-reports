import numpy as np
import pandas as pd
import random
import os
import sys

sys.path.insert(0, os.getcwd())

from ai_code import run_prompts
from ai_code import run_prompts_json
from referee_reports.document_readers import ReportReader

def report_reader_merged(raw_pickled_documents_directory: str = "test_data/raw/reports-pkl/",
    cleaned_pickled_output_directory: str = "",
    referee_characteristics_file: str = "test_data/raw/referee_gender_nonames.csv"):
    """
    returns a ReportReader object after having been merged with the referee characteristics
    """

    report_reader = ReportReader(raw_pickled_documents_directory, cleaned_pickled_output_directory, referee_characteristics_file)
    report_reader._validate_raw_data()
    report_reader._filter_duplicate_documents()
    report_reader._format_index()
    report_reader._decode_text()
    report_reader._merge_referee_characteristics()
    report_reader._tokenize_text()
    return report_reader

def get_full_df(report_reader):
    """
    creating df for AI ingestion
    """
    # reading in base table
    df = report_reader._df
    df.reset_index([0, 1])
    df = df.reset_index(['paper','num'])
    df['id'] = df['paper'] + "-" + df['num'].astype(str)
    df['category'] = np.where(df['female'] == 1, 'A', 'B') # female is A, male is B
    df = df[['id', 'cleaned_text', 'category']]
    avg = 0
    for i in range(len(df['cleaned_text'])):
        avg += len(df['cleaned_text'][i])
    print(f"avg len of cleaned text of all report: {avg/len(df['cleaned_text'])}")
    return df

if __name__ == '__main__':
    """
    1) create report, dataframe, and jsons (train and test)
    2) run prompts
    """

    # set seeds for sampling
    np.random.seed(3)
    random.seed(3)

    # create report, df, and csv - TRAIN and TEST
    report_reader_train = report_reader_merged("../../data/alapre/oscar_data/reports-pkl/", "", "../../data/alapre/oscar_data/data/key_sheet_nonames_230228.csv")
    df = get_full_df(report_reader_train)
    df = df.sample(frac=1, random_state = 3)
    df.reset_index(drop=True, inplace=True)

    df_train = df.iloc[:50]
    df_test = df.iloc[50:100]
    df_test = df_test.drop(["category"], axis = 1)
    print(f"train len: {len(df_train)}")
    print(f"test len: {len(df_test)}")
    print(df_train.columns)
    print(df_test.columns)
    print(df_train['id'][0:10])
    train_json = df_train.to_json(orient="records")
    test_json = df_test.to_json(orient="records")

    # run prompts
    run_prompts_json.run_parametric_series_prompts(train_json, test_json)
    run_prompts_json.run_nonparametric_series_prompts(train_json, test_json)