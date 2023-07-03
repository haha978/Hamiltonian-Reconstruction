import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from cycler import cycler
import argparse

def load_l(name):
    output = None
    with open(os.path.join(name), "rb") as fp:
        output = pickle.load(fp)
    return output

def get_mark_cycler():
    marker_cycler = (cycler(color=["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"]) +
                 cycler(linestyle=["none", "none", "none", "none", "none", "none", "none"]) +
                 cycler(marker=["4", "2", "3", "1", "+", "x", "."]))
    return marker_cycler

def make_avg_plot():
    """
    function that creates a plot that averages HR distances to show better correlation with energy
    truncates the plot at an appropriate energy if needed
    """
    #SET avg_num HERE:
    avg_num = 1
    #SET TRUNCATION HERE:
    trunc = False

    HR_dist_hist = load_l("HR_param_idx_l_10000shots_ionq.qpu_p1_0.0065_p2_0.0398.pkl")
    E_hist = load_l("noisy_E_param_idx_l_10000_shots_ionq.qpu__p1_0.0065_p2_0.0398.pkl")
    fid_hist = load_l("fid_param_idx_l_p1_0.0065_p2_0.0398.pkl")
    param_idx_l = np.load("param_idx_l.npy", allow_pickle = True)
    if trunc:
        HR_dist_hist = HR_dist_hist[:-7]
        E_hist = E_hist[:-7]
        fid_hist = fid_hist[:-7]
        param_idx_l = param_idx_l[:-7]
    if avg_num % 2 == 1:
        n = avg_num // 2
        param_idx_l_avg = param_idx_l[n: len(param_idx_l)-n]
    else:
        n = avg_num // 2 - 1
        param_idx_l_trunc = param_idx_l[n: len(param_idx_l)-n]
        param_idx_l_avg = []
        for i in list(range(len(param_idx_l_trunc)-1)):
            param_idx_avg = (param_idx_l_trunc[i] + param_idx_l_trunc[i+1])/2
            param_idx_l_avg.append(param_idx_avg)
    HR_dist_hist_avg = []
    for idx in range(0, len(HR_dist_hist) - avg_num + 1):
        HR_dist_avg = 0
        for i in range(idx, idx+avg_num):
            HR_dist_avg += HR_dist_hist[i]
        HR_dist_avg = HR_dist_avg/avg_num
        HR_dist_hist_avg.append(HR_dist_avg)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    fig, ax = plt.subplots()
    ax.scatter(param_idx_l, E_hist, alpha = 1, c = "#0072B2", s = 20, marker = ".", label = "Energy")
    ax.set_xlabel('VQE Iterations', fontsize = 12)
    ax.set_ylabel("Energy", fontsize = 12)
    ax.legend(bbox_to_anchor=(1.28, 1.30), fontsize = 12)
    ax2 = ax.twinx()
    ax2.scatter(param_idx_l_avg, HR_dist_hist_avg, alpha = 1, c = "#D55E00", s = 20, marker=".", label = "HR distance")
    ax2.scatter(param_idx_l, fid_hist, c = '#009E73', alpha = 0.8, marker=".", label = "Fidelity")
    ax2.set_ylabel("HR distance | Fidelity", fontsize = 12)
    ax2.legend(bbox_to_anchor=(1.28, 1.22), fontsize = 12)
    if trunc:
        plt.savefig(f"plot_2c_avg_{avg_num}_trunc.svg", dpi = 300, bbox_inches='tight')
    else:
        plt.savefig(f"plot_2c_avg_{avg_num}.svg", dpi = 300, bbox_inches='tight')

def make_plot():
    #creates plot with full data
    E_hist = load_l("noisy_E_param_idx_l_10000_shots_ionq.qpu__p1_0.0065_p2_0.0398.pkl")
    HR_dist_hist = load_l("HR_param_idx_l_10000shots_ionq.qpu_p1_0.0065_p2_0.0398.pkl")
    fid_hist = load_l("fid_param_idx_l_p1_0.0065_p2_0.0398.pkl")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    fig, ax = plt.subplots()
    param_idx_l = np.load("param_idx_l.npy", allow_pickle = True)
    ax.scatter(param_idx_l, E_hist, alpha = 1, c = "#0072B2", s = 20, marker = ".", label = "Energy")
    ax.set_xlabel('VQE Iterations', fontsize = 12)
    ax.set_ylabel("Energy", fontsize = 12)
    ax.legend(bbox_to_anchor=(1.28, 1.30), fontsize = 12)
    ax2 = ax.twinx()
    ax2.scatter(param_idx_l, HR_dist_hist, alpha = 1, c = "#D55E00", s = 20, marker=".", label = "HR distance")
    ax2.scatter(param_idx_l, fid_hist, c = '#009E73', alpha = 0.8, marker=".", label = "Fidelity")
    ax2.set_ylabel("HR distance | Fidelity", fontsize = 12)
    ax2.legend(bbox_to_anchor=(1.28, 1.22), fontsize = 12)
    plt.savefig("plot_2c.svg", dpi = 300, bbox_inches='tight')

def make_plot_trunc():
    #creates truncated plot where it get rids of the last few iterations that the energy increased.
    E_hist = load_l("noisy_E_param_idx_l_10000_shots_ionq.qpu__p1_0.0065_p2_0.0398.pkl")
    HR_dist_hist = load_l("HR_param_idx_l_10000shots_ionq.qpu_p1_0.0065_p2_0.0398.pkl")
    fid_hist = load_l("fid_param_idx_l_p1_0.0065_p2_0.0398.pkl")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    fig, ax = plt.subplots()
    param_idx_l = np.load("param_idx_l.npy", allow_pickle = True)
    #truncate each of the list
    E_hist = E_hist[:-7]
    HR_dist_hist = HR_dist_hist[:-7]
    fid_hist = fid_hist[:-7]
    param_idx_l = param_idx_l[:-7]
    ax.scatter(param_idx_l, E_hist, alpha = 1, c = "#0072B2", s = 20, marker = ".", label = "Energy")
    ax.set_xlabel('VQE Iterations', fontsize = 12)
    ax.set_ylabel("Energy", fontsize = 12)
    ax.legend(bbox_to_anchor=(1.28, 1.30), fontsize = 12)
    ax2 = ax.twinx()
    ax2.scatter(param_idx_l, HR_dist_hist, alpha = 1, c = "#D55E00", s = 20, marker=".", label = "HR distance")
    ax2.scatter(param_idx_l, fid_hist, c = '#009E73', alpha = 0.8, marker=".", label = "Fidelity")
    ax2.set_ylabel("HR distance | Fidelity", fontsize = 12)
    ax2.legend(bbox_to_anchor=(1.28, 1.22), fontsize = 12)
    plt.savefig("plot_2c_trunc.svg", dpi = 300, bbox_inches='tight')

def main():
    make_avg_plot()

if __name__ == '__main__':
    main()
