import pkg_resources
import pkldir 
import pickle
import tempfile 




def test_encode(filename):
    print("Testing encoding...")
    enc = pkldir.encode(filename)
    tmp = tempfile.TemporaryFile() 

    with open(tmp.name, 'wb') as f:
        pickle.dump(enc, f)

    assert 1 == 1 


# pkldir.decode("/Users/pstey/Desktop/out.pkl")

if __name__ == "__main__":
    data_file = pkg_resources.resource_filename('pkldir', "data/two-cities.txt")
