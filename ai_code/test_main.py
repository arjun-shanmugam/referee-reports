import numpy as np
import pandas as pd
import random
import os
import sys

sys.path.insert(0, os.getcwd())

from ai_code import run_prompts
from ai_code import run_prompts_json
from ai_code import main
from referee_reports.document_readers import ReportReader

def test_report_reader():
    """
    quick test to ensure all data is correct for mergered Report Reader
    """
    report_reader = main.report_reader_merged()
    actual_df = report_reader._df.drop(columns=['full_filename', 'raw_text'])
    expected_df = pd.DataFrame([["Reject", "Reject", 1, 1, 0, 1, 0],
                                ["Major Revision", "Revise", 1, 1, 0, 1, 0],
                                ["Accept", "Reject", 1, 1, 0, 1, 0],
                                ["Reject", "Reject", 1, 1, 0, 1, 0],
                                ["Reject", "Reject", 1, 0, 0, 1, np.nan],
                                ["Major Revision", "Revise", 1, 0, 0, 1, np.nan],
                                ["Accept", "Reject", 1, 0, 0, 1, np.nan]],
                               columns=["recommendation", "decision", "female", "author_1_female", "author_2_female", "author_3_female", "author_4_female"],
                               index=pd.MultiIndex.from_tuples([('99-99999', 1), ('99-99999', 2), ('99-99999', 3), ('99-99999', 4),
                                                                ('99-99998', 1), ('99-99998', 2), ('99-99998', 3)]))
    expected_df.index = expected_df.index.rename(['paper', 'num'])
    pd.testing.assert_frame_equal(actual_df, expected_df, check_like=True, check_dtype=False)
    print("passed!")

def test_main():
    # unpickle data and and create dfs to be made into jsons!

    df_train = pd.DataFrame({"idx": ['id0','id1', 'id2','id3'], "raw_text": ["hi!", "bye", "hello!", "goodbye"], "category": ["A", "B", "A", "B"]})
    train_json = df_train.to_json()
    print(train_json)
    df_test = pd.DataFrame({"idx": ['id0','id1'], "raw_text": ["hey!", "see you later"]})
    test_json = df_test.to_json()
    print(test_json)
    
    # run prompts
    run_prompts_json.run_parametric_series_prompts(train_json, test_json)
    run_prompts_json.run_nonparametric_series_prompts(train_json, test_json)

if __name__ == '__main__':
    test_report_reader()
    test_main()