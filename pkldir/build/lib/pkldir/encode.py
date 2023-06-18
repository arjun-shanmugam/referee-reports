import argparse
import pickle
import os 
import shutil 

from . import Base64Encoded

def _clean_old_pkldir(output_dir):
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    return None 


def encode(filename):
    full_file_path = os.path.abspath(filename)
    return Base64Encoded(full_file_path)



def encode_dir(input_dir):

    files = os.listdir(input_dir)

    output_dir = input_dir + "-pkl"

    _clean_old_pkldir(output_dir) 

    os.mkdir(output_dir)

    print("Encoding the following: ")
    for filename in files: 
        print(f'  {filename}')
        full_file_path = os.path.join(input_dir, filename)
        pickle.dump(encode(full_file_path), open(os.path.join(output_dir, filename + ".pkl"), "wb"))

