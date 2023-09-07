import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import argparse
import subprocess
import copy


parser = argparse.ArgumentParser(description='A test program.')

parser.add_argument("--hr_run", type = int, default= -1, help="-1 if need to run HR, else no")
parser.add_argument("--make_plots", type = int, default= -1, help="-1 if need to make plots, else no")
args = parser.parse_args()

### 
# saved_dir = "/home/mas763/HR_simulations/HR_folder/hr_p1p2_plot_n_qbts_4_J_0.5_shots_10000_TEST"
# run_dir = "/home/mas763/HR_simulations/HR_folder/"
# run_str = "python real_noise_VQE_load_param.py --n_qbts 4 --J 0.5 --shots 10000" + " " \
#             "--init_param /home/mas763/HR_simulations/4_qubits_TFIM_X+0.5ZZ/4qubits_8params/GND_E__-4.1863__angles_file.dat" + " " \
#             "--start_idx 570 --end_idx 599" + " " + "--fid 0" + " "

# saved_dir = "/home/mas763/HR_simulations/HR_folder/hr_p1p2_plot_n_qbts_10_J_0.5_shots_10000"
# run_dir = "/home/mas763/HR_simulations/HR_folder/"
# run_str = "python real_noise_VQE_load_param.py --n_qbts 10 --J 0.5 --shots 10000" + " " \
#             "--init_param /home/mas763/HR_simulations/10_qubits_TFIM_X+0.5ZZ/10qubits_20params/GND_E__-10.5546__angles_file.dat" + " " \
#             "--start_idx 3700 --end_idx 3725" + " " + "--fid 0" + " "
cwd = os.getcwd()
saved_dir = os.path.join(cwd,"hr_p1p2_plot_n_qbts_11_J_0.5_depolarization_only")
run_dir = cwd
vqe_path = os.path.join(cwd, "11_qubits_TFIM_X+0.5ZZ/11qubits_22params/GND_E__-11.6108__angles_file.dat")
run_str = "python real_noise_VQE_load_param_noiseless.py --n_qbts 11 --J 0.5 --shots 10000" + " " \
            "--init_param /home/mas763/HR_simulations/11_qubits_TFIM_X+0.5ZZ/11qubits_22params/GND_E__-11.6108__angles_file.dat" + " " \
            "--start_idx 9300 --end_idx 9325" + " " + "--fid 0" + " "

num_cores = 40
num_samples = 24
n_qbts = 11

### Decide Scales
points_per_axis = 150

## P2 Values
y_axis_start = 10**-4
y_axis_end = .2

## P1 Values
x_axis_start = 10**-6
x_axis_end = 10**-1

def round_to_n(x, n):
    return round(x, -int(np.floor(np.sign(x) * np.log10(abs(x)))) + n)


### Axes
y_axis = np.zeros(points_per_axis)
y_axis[0] = y_axis_start
y_axis[-1] = y_axis_end
x_axis = np.zeros(points_per_axis)
x_axis[0] = x_axis_start
x_axis[-1] = x_axis_end

### Making Log Log Grid
for i in range(1, points_per_axis-1):

    x_axis[i] = 10**(np.log10(x_axis[i-1]) + (np.log10(x_axis_end) - np.log10(x_axis_start))/(points_per_axis-1))
    y_axis[i] = 10**(np.log10(y_axis[i-1]) + (np.log10(y_axis_end) - np.log10(y_axis_start))/(points_per_axis-1))

### Order of Data Points for collection and plotting
p1_vals = [round_to_n(i,8) for i in x_axis for j in y_axis]
p2_vals = [round_to_n(j,8) for i in x_axis for j in y_axis]
# p1_vals = [i for i in x_axis for j in y_axis]
# p2_vals = [j for i in x_axis for j in y_axis]
p1_vals_temp = copy.deepcopy(p1_vals)
p2_vals_temp = copy.deepcopy(p2_vals)

### Data Collection Arrays
HR_distance_p1_p2 = np.zeros(len(p1_vals))
energy_p1_p2 = np.zeros(len(p1_vals))
fidelity_p1_p2 = np.zeros(len(p1_vals))

if args.hr_run == -1:

    ### Data Collection loop
    ### Move to correct directory
    current_path = os.getcwd()
    print(current_path)
    if not os.path.isdir(saved_dir):
        os.mkdir(saved_dir)
    os.chdir(run_dir)

    ### Delete Values Already Found
#     for i in range(len(p1_vals_temp)-1,-1,-1):
#         if os.path.exists(saved_dir + "/" + "p1__" + str(round_to_n(p1_vals_temp[i],8)) + "p2__" + str(round_to_n(p2_vals_temp[i],8)) + "/HR_dist_hist.pkl") and len(pickle.load(open(saved_dir + "/" + "p1__" + str(round_to_n(p1_vals_temp[i],8)) + "p2__" + str(round_to_n(p2_vals_temp[i],8)) + "/HR_dist_hist.pkl", "rb"))) >= num_samples:
            
    for i in range(len(p1_vals_temp)-1,-1,-1):
        ## print(p1_vals[i],p2_vals[i]) ## Debug Step for Pickle
        if os.path.exists(saved_dir + "/" + "p1__" + str(p1_vals_temp[i]) + "p2__" + str(p2_vals_temp[i]) + "/HR_dist_hist.pkl") and len(pickle.load(open(saved_dir + "/" + "p1__" + str(p1_vals_temp[i]) + "p2__" + str(p2_vals_temp[i]) + "/HR_dist_hist.pkl", "rb"))) >= num_samples:

            del p1_vals_temp[i]
            del p2_vals_temp[i]

    print(len(p1_vals_temp))

    for i in range(0,int(np.floor(len(p1_vals_temp)/num_cores+1))):
        sub_arr = []
        print(i*num_cores)
        if i == int(len(p1_vals_temp)/num_cores):
            for j in range(len(p1_vals_temp)%num_cores):
                sub_arr.append(run_str + "--p1 "+ str(p1_vals_temp[i*num_cores+j]) + " " + "--p2 " + str(p2_vals_temp[i*num_cores+j]) + " " + "--output_dir " + saved_dir + "/" + "p1__" + str(p1_vals_temp[i*num_cores+j]) + "p2__" + str(p2_vals_temp[i*num_cores+j]))
        else:    
            for j in range(num_cores):
                sub_arr.append(run_str + "--p1 "+ str(p1_vals_temp[i*num_cores+j]) + " " + "--p2 " + str(p2_vals_temp[i*num_cores+j]) + " " + "--output_dir " + saved_dir + "/" + "p1__" + str(p1_vals_temp[i*num_cores+j]) + "p2__" + str(p2_vals_temp[i*num_cores+j]))
        if len(sub_arr) != 0:
            procs = [subprocess.Popen(sub, shell=True) for sub in sub_arr]
            for p in procs:
                p.wait()
            os.chdir(run_dir)


    # for i, p1 in enumerate(x_axis):

    #     for j, p2 in enumerate(y_axis):
    #         os.system(run_str + "--p1 "+ str(p1) + " " + "--p2 " + str(p2) + " " + "--output_dir " + saved_dir +"/" + "p1__" + str(round(p1, 6)) + "p2__" + str(round(p2, 6)))
    #         os.chdir(run_dir)

if args.make_plots == -1:
    for i in range(len(p1_vals)):
        
        os.chdir(saved_dir + "/" +  "p1__" + str(p1_vals[i]) +  "p2__" + str(p2_vals[i]))

        HR_dist_f = open("HR_dist_hist.pkl", "rb")
        HR = pickle.load(HR_dist_f)
        HR_dist_f.close()

        E_hist_f = open("E_hist.pkl", "rb")
        E = pickle.load(E_hist_f)
        E_hist_f.close()

        fid_hist_f = open("fid_hist.pkl", "rb")
        fid = pickle.load(fid_hist_f)
        fid_hist_f.close()

        HR_distance_p1_p2[i] = np.average(HR)
        energy_p1_p2[i] = np.average(E)
        fidelity_p1_p2[i] = np.average(fid)

    
    os.chdir(saved_dir)
    with open("HR_distance_p1_p2","wb") as HR_distance_p1_p2_f:
        pickle.dump(HR_distance_p1_p2,HR_distance_p1_p2_f)
    
    with open("energy_p1_p2","wb") as energy_p1_p2_f:
        pickle.dump(energy_p1_p2,energy_p1_p2_f)

    with open("fidelity_p1_p2","wb") as fidelity_p1_p2_f:
        pickle.dump(fidelity_p1_p2,fidelity_p1_p2_f)
        
    if os.path.exists(run_dir + "/plots_p1p2_folder") == False:
        os.mkdir(run_dir + "/plots_p1p2_folder")
    os.chdir(run_dir + "/plots_p1p2_folder")
#     #plt.rcParams['text.usetex'] == False
#     plt.rcParams['font.family'] = 'serif'
#     plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
#     plt.rcParams["font.size"] = 12
#     fontProperties = {'font.family': 'serif','font.serif':['Times New Roman'] + plt.rcParams['font.serif'],
#     'font.weight' : 'normal', 'font.size' : 12}
    new_rc_params = {
    "font.family": 'serif',
    "font.size": 12,
    "font.serif": ['Times New Roman'] + plt.rcParams['font.serif'],
    "svg.fonttype": 'none',
    'text.usetex': False,
    "svg.fonttype": 'none'}
    plt.rcParams.update(new_rc_params)

    #print(fidelity_p1_p2)
    
    ### SQUARE FIDELITY
    fidelity_p1_p2 = np.array(fidelity_p1_p2)
    fidelity_p1_p2 = 100*((fidelity_p1_p2/100)**2)

    bins=(np.logspace(np.log10(x_axis_start), np.log10(x_axis_end), points_per_axis+1),np.logspace(np.log10(y_axis_start), np.log10(y_axis_end), points_per_axis+1))
    fig = plt.figure()
    ax = plt.axes()
    plt.hist2d(p1_vals,p2_vals, weights = HR_distance_p1_p2, bins = bins, cmap = "Reds",linewidth=0,rasterized=True)
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.colorbar()
    plt.clim(0,1)
    plt.xlabel("One Qubit Gate Error Rate",fontsize = 12)
    plt.ylabel("Two Qubit Gate Error Rate",fontsize = 12)
    plt.title("HR Distance Versus One and Two Qubit Gate Error Rates for 11 Qubit Ising 22 Params",fontsize = 12)
    plt.savefig("HR_vs_p1_p2_plot" + str(n_qbts)+ "qubits_depolarization.svg",dpi = 300, bbox_inches='tight')

    fig2 = plt.figure()
    ax2 = plt.axes()
    plt.hist2d(p1_vals,p2_vals, weights = energy_p1_p2, bins = bins, cmap = "Reds",linewidth=0,rasterized=True)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    plt.colorbar()
    plt.xlabel("One Qubit Gate Error Rate",fontsize = 12)
    plt.ylabel("Two Qubit Gate Error Rate",fontsize = 12)
    plt.title("Energy Versus One and Two Qubit Gate Error Rates for 11 Qubit Ising 22 Params",fontsize = 12)
    plt.savefig("Energy_vs_p1_p2_plot" + str(n_qbts)+ "qubits_depolarization.svg",dpi = 300, bbox_inches='tight',format='svg')

    fig3 = plt.figure()
    ax3 = plt.axes()
    plt.hist2d(p1_vals,p2_vals, weights = fidelity_p1_p2, bins = bins, cmap = "Reds",linewidth=0,rasterized=True)
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    plt.colorbar()
    cbar = ax3.collections[0].colorbar
    plt.clim(0,100)
    cbar.set_ticks([0,25,50,75, 100])
    cbar.set_ticklabels(["0%","25%", "50%","75%", "100%"])
    plt.xlabel("One Qubit Gate Error Rate",fontsize = 12)
    plt.ylabel("Two Qubit Gate Error Rate",fontsize = 12)
    plt.title("Fidelity Versus One and Two Qubit Gate Error Rates for 11 Qubit Ising 22 Params",fontsize = 12)
    plt.savefig("Fidelity_vs_p1_p2_plot" + str(n_qbts)+ "qubits_depolarization.png",dpi = 300, bbox_inches='tight')
    
    print(plt.rcParams['font.serif'])