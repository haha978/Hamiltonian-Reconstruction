import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from cycler import cycler

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

def make_plot():
    avg_num = 20
    param_idx_l = np.load("param_idx_l.npy", allow_pickle = True)
    E_hist = load_l("noisy_E_hist.pkl")
    E_hist = [E_hist[idx] for idx in param_idx_l]
    HR_dist_hist = load_l("HR_dist_hist.pkl")
    HR_dist_hist = [HR_dist_hist[idx] for idx in param_idx_l]
    fid_hist = load_l("fid_hist.pkl")
    fid_hist = [fid_hist[idx] for idx in param_idx_l]
    E_hist_ionq = load_l("noisy_E_hist_ionq.pkl")
    HR_dist_hist_ionq = load_l("HR_dist_hist_ionq.pkl")
    fid_hist_dep = load_l("fid_hist_dep.pkl")

    #get average
    param_idx_l_avg = get_hist_avg(avg_num, param_idx_l)
    E_hist_avg = get_hist_avg(avg_num, E_hist)
    HR_dist_hist_avg = get_hist_avg(avg_num, HR_dist_hist)
    fid_hist_avg = get_hist_avg(avg_num, fid_hist)
    #get average for dep
    E_hist_ionq_avg = get_hist_avg(avg_num, E_hist_ionq)
    HR_dist_hist_ionq_avg = get_hist_avg(avg_num, HR_dist_hist_ionq)
    fid_hist_dep_avg = get_hist_avg(avg_num, fid_hist_dep)

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    fig, ax = plt.subplots()
    ax.scatter(param_idx_l, E_hist, alpha = 0.1, c = "#0072B2", s = 20, marker = ".")
    ax.scatter(param_idx_l_avg, E_hist_avg, alpha = 1, c = "#0072B2", s = 20, marker = ".", label = "Energy (no dep)")
    ax.scatter(param_idx_l, E_hist_ionq, alpha = 0.1, c = "#E69F00", s = 20, marker = ".")
    ax.scatter(param_idx_l_avg, E_hist_ionq_avg, alpha = 1, c = "#E69F00", s = 20, marker = ".", label = "Energy (ionq)")

    ax.set_xlabel('VQE Iterations', fontsize = 12)
    ax.set_ylabel("Energy", fontsize = 12)
    ax.legend(bbox_to_anchor=(1.5, 1.30), fontsize = 12)
    ax2 = ax.twinx()
    ax2.scatter(param_idx_l, HR_dist_hist, alpha = 0.1, c = "#D55E00", s = 20, marker=".")
    ax2.scatter(param_idx_l_avg, HR_dist_hist_avg, alpha = 1, c = "#D55E00", s = 20, marker=".", label = "HR distance (no dep)")
    ax2.scatter(param_idx_l, HR_dist_hist_ionq, alpha = 0.1, c = "#F0E442", s = 20, marker=".")
    ax2.scatter(param_idx_l_avg, HR_dist_hist_ionq_avg, alpha = 1, c = "#F0E442", s = 20, marker=".", label = "HR distance (ionq)")

    ax2.scatter(param_idx_l, fid_hist, c = '#56B4E9', alpha = 0.1, marker=".")
    ax2.scatter(param_idx_l_avg, fid_hist_avg, c = '#56B4E9', alpha = 1, marker=".", label = "Fidelity (no dep)")
    ax2.scatter(param_idx_l, fid_hist_dep, c = '#CC79A7', alpha = 0.1, marker=".")
    ax2.scatter(param_idx_l_avg, fid_hist_dep_avg, c = '#CC79A7', alpha = 1, marker=".", label = "Fidelity (dep)")

    ax2.set_ylabel("HR distance | Fidelity", fontsize = 12)
    ax2.legend(bbox_to_anchor=(1.5, 1.14), fontsize = 12)
    plt.savefig("plot_2a_E_fid_HR.svg", dpi = 300, bbox_inches='tight')

def make_plot2():
    avg_num = 20
    param_idx_l = np.load("param_idx_l.npy", allow_pickle = True)
    E_hist = load_l("noisy_E_hist.pkl")
    E_hist = [E_hist[idx] for idx in param_idx_l]
    HR_dist_hist = load_l("HR_dist_hist.pkl")
    HR_dist_hist = [HR_dist_hist[idx] for idx in param_idx_l]
    fid_hist = load_l("fid_hist.pkl")
    fid_hist = [fid_hist[idx] for idx in param_idx_l]
    E_hist_ionq = load_l("noisy_E_hist_ionq.pkl")
    HR_dist_hist_ionq = load_l("HR_dist_hist_ionq.pkl")
    fid_hist_dep = load_l("fid_hist_dep.pkl")

    #get average
    param_idx_l_avg = get_hist_avg(avg_num, param_idx_l)
    E_hist_avg = get_hist_avg(avg_num, E_hist)
    HR_dist_hist_avg = get_hist_avg(avg_num, HR_dist_hist)
    fid_hist_avg = get_hist_avg(avg_num, fid_hist)
    #get average for dep
    E_hist_ionq_avg = get_hist_avg(avg_num, E_hist_ionq)
    HR_dist_hist_ionq_avg = get_hist_avg(avg_num, HR_dist_hist_ionq)
    fid_hist_dep_avg = get_hist_avg(avg_num, fid_hist_dep)

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    fig, ax = plt.subplots()
    ax.set_xlabel('VQE Iterations', fontsize = 12)
    ax.scatter(param_idx_l, HR_dist_hist, alpha = 0.1, c = "#D55E00", s = 20, marker=".")
    ax.scatter(param_idx_l_avg, HR_dist_hist_avg, alpha = 1, c = "#D55E00", s = 20, marker=".", label = "HR distance (no dep)")
    ax.scatter(param_idx_l, HR_dist_hist_ionq, alpha = 0.1, c = "#F0E442", s = 20, marker=".")
    ax.scatter(param_idx_l_avg, HR_dist_hist_ionq_avg, alpha = 1, c = "#F0E442", s = 20, marker=".", label = "HR distance (ionq)")

    ax.scatter(param_idx_l, fid_hist, c = '#56B4E9', alpha = 0.1, marker=".")
    ax.scatter(param_idx_l_avg, fid_hist_avg, c = '#56B4E9', alpha = 1, marker=".", label = "Fidelity (no dep)")
    ax.scatter(param_idx_l, fid_hist_dep, c = '#CC79A7', alpha = 0.1, marker=".")
    ax.scatter(param_idx_l_avg, fid_hist_dep_avg, c = '#CC79A7', alpha = 1, marker=".", label = "Fidelity (dep)")

    ax.set_ylabel("HR distance | Fidelity", fontsize = 12)
    ax.legend(bbox_to_anchor=(1.5, 1.14), fontsize = 12)
    plt.savefig("plot_2a_fid_HR.svg", dpi = 300, bbox_inches='tight')

def get_hist_avg(avg_num, hist):
    hist_avg = []
    for idx in list(range(len(hist) - avg_num + 1)):
        avg = 0
        for i in range(idx, idx + avg_num):
            avg += hist[i]
        avg = avg / avg_num
        hist_avg.append(avg)
    return hist_avg

def main():
    make_plot()
    make_plot2()
    #make_avg_plot()

if __name__ == '__main__':
    main()
