import numpy as np
import os
import argparse

def get_args(parser):
    parser.add_argument('--input_dir', type = str, help = "directory where param_idx_l.npy will be stored (PLEASE MAKE SURE TO EDIT THE SCRIPT FIRST!)")
    return args

def main(args):
    #give your list here
    param_idx_l = list(range(0, 400, 80)) + list(range(400, 600, 20))
    print(param_idx_l)
    np.save(os.path.join(args.input_dir, "param_idx_l.npy"), np.array(param_idx_l))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "script to create parameter index list")
    args = get_args(parser)
    main(args)
