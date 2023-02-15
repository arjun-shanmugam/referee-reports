"""Referee Bias NLP Project

    PIs: John Friedman, Amy Handlan, Nathan Hendren
    Author: Arjun Shanmugam
"""
from referee_reports.document_readers import PaperReader, ReportReader
import matplotlib.pyplot as plt
plt.switch_backend('agg')  # Prevents an error from being thrown when parallelizing using joblib.


# Constants======================================================================================================
ON_STRONGHOLD = False
if ON_STRONGHOLD:
    RAW_PICKLED_REPORTS = "/data/referee/pkldir/reports-pkl/"
    RAW_PICKLED_PAPERS = "/data/referee/pkldir/papers-pkl/"
    REFEREE_CHARACTERISTICS = "/data/referee/data/referee_gender_nonames.csv"
else:
    RAW_PICKLED_REPORTS = "../../data/raw/reports-pkl/"
    RAW_PICKLED_PAPERS = "../../data/raw/papers-pkl/"
    REFEREE_CHARACTERISTICS = "../../data/raw/referee_gender_nonames.csv"
CLEANED_PICKLED_OUTPUT = "../../data/intermediate/"
PREPROCESSING_OUTPUT = "../../output/preprocessing/"

# Read Papers====================================================================================================
print("READING PAPERS")
paper_reader = PaperReader(RAW_PICKLED_PAPERS,
                           CLEANED_PICKLED_OUTPUT)
paper_reader.build_df()
paper_reader.analyze_repeated_documents("filetype_analysis_papers.png")
print("Done!")
print("\n\n\n\n")


# Read Reports===================================================================================================
print("READING REPORTS")
report_reader = ReportReader(raw_pickled_documents=RAW_PICKLED_REPORTS,
                             cleaned_pickled_output=CLEANED_PICKLED_OUTPUT,
                             path_to_output=PREPROCESSING_OUTPUT,
                             path_to_referee_characteristics=REFEREE_CHARACTERISTICS)
report_reader.build_df()
report_reader.analyze_repeated_documents("filetype_analysis_reports.png")
print("Done!")
print("\n\n\n\n")

