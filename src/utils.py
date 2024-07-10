import os
import pickle


def save_file_as_pickle(path,obj_name):
    os.makedirs(os.path.dirname(path),exist_ok=True)
    with open(path,"wb") as path:
        pickle.dump(obj_name,path)
        print("\t\t\t\t\t\t\tpickle file save")