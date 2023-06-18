import os
import pickle

def decode(filename):
    full_file_path = os.path.abspath(filename)

    with open(full_file_path, 'rb') as f:

        doc = renamed_load(f)
    return doc.decode()

class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        renamed_module = module
        if module == "pkldir.encoding.encoding":
            renamed_module = "referee_reports.pkldir.encoding.encoding"
        return super(RenameUnpickler, self).find_class(renamed_module, name)


def renamed_load(file_obj):
    return RenameUnpickler(file_obj).load()


def renamed_loads(pickled_bytes):
    file_obj = io.BytesIO(pickled_bytes)
    return renamed_load(file_obj)