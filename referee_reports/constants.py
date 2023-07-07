"""
Author: Arjun Shanmugam
Referee Bias Project
"""
class NLPConstants:
    NLTK_STOPWORDS_ENGLISH = {'her', 'for', 'that', 'an', 'here', 'nor', 'same', 'mustn', 'wouldn', 'again', 'our',
                              'your', "hasn't", 'down', 'his', 'had', 'too', "didn't", "haven't", 'then', 'yours',
                              're', 'can', 'yourself', 'didn', "wouldn't", 'at', 'you', 'how', 'now', 'other',
                              'through', 'themselves', 'further', 'are', 'was', 'out', 's', "don't", "won't", 'itself',
                              "shouldn't", 'them', 'shan', 'who', 'doing', 'just', 'been', "you'll", 'having', 'has',
                              'any', "shan't", 'where', 'i', 'whom', 'the', "you'd", 'under', 'don', 'only', 'while',
                              'after', 'what', "you've", 'until', 'yourselves', 'than', 'doesn', 'does', 'so', "it's",
                              've', 'to', 'own', 'wasn', 'weren', 'isn', 'theirs', 'himself', "hadn't", "hadn't", 'in',
                              'of', 'these', 'a', "doesn't", 'needn', 'because', 'aren', "couldn't", 'hasn', 't', '_y_data',
                              'up', 'when', 'few', 'this', 'will', 'all', 'hers', 'll', "mightn't", 'between', 'their',
                              'be', 'with', 'myself', 'being', 'he', 'did', 'about', 'as', 'but', 'shouldn', 'him',
                              'some', 'herself', 'before', 'there', 'very', 'more', 'over', 'on', "isn't", 'have',
                              "musn't", 'do', 'o', 'not', 'ourselves', 'from', 'my', 'm', 'me', 'its', 'once', 'those',
                              "weren't", 'by', 'were', 'they', 'against', 'why', 'should', 'such', "you're", 'we',
                              'most', 'couldn', 'off', 'd', 'won', "needn't", "wasn't", 'into', "should've", "aren't",
                              'if', 'ain', 'or', "that'll", 'it', 'hadn', 'below', 'ma', 'both', 'is', 'am', 'each',
                              "she's", 'ours', 'during', 'haven', 'mightn', 'no', 'and', 'above', 'she', 'which'}

class Colors:
    P1 = "#29B6A4"
    P2 = "#FAA523"
    P3 = "#003A4F"
    P4 = "#7F4892"
    P5 = "#A4CE4E"
    P6 = "#2B8F43"
    P7 = "#0073A2"
    P8 = "#E54060"
    P9 = "#FFD400"
    P10 = "#6BBD45"

    OI_colors = [P1, P2, P3, P4, P5, P6, P7, P8, P9, P10]

    SUMMARY_STATISTICS_COLOR = 'black'
    LABELING_COLOR = 'grey'
    TREATMENT_COLOR = 'red'
    CONTROL_COLOR = 'blue'
