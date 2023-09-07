import numpy as np
import pickle
import os
cwd = os.getcwd()
for folder in os.listdir(os.getcwd()):
    if os.path.isdir(folder):
        with open(os.path.join(cwd, folder, "fid_hist.pkl"), "rb") as f:
            arr = 100*((np.array(pickle.load(f))/100)**2)
        print(arr)
        with open(os.path.join(cwd, folder, "fid_hist.pkl"), "wb") as f:
            pickle.dump(arr, f)