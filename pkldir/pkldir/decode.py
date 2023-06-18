import os
import pickle


def decode(filename):
    full_file_path = os.path.abspath(filename)

    with open(full_file_path, 'rb') as f:
        doc = pickle.load(f)
    return doc.decode()