import pickle
import numpy as np
import os
import subprocess
import scipy
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

dict_arr = []
parent_dir = os.getcwd()
n_layers = 3
remove_measurement = True
HR_collection = np.zeros((8, 50))
Fid_collection = np.zeros((8, 50))
correlations = np.zeros(8)
remove_measurement = True
rerun = True
num_points = 20

new_rc_params = {
    "font.family": 'serif',
    "font.size": 12,
    "font.serif": ['Times New Roman'] + plt.rcParams['font.serif'],
    "svg.fonttype": 'none',
    'text.usetex': False,
    "svg.fonttype": 'none'}
plt.rcParams.update(new_rc_params)


if rerun == True:
    for nqbts in range (4, 12): 

        ### Find the Nparams value for run
        Nparams = 0
        if nqbts % 2 == 0:
            for i in range(n_layers):
                if i % 2 == 0:
                    Nparams += nqbts
                else:
                    Nparams += (nqbts - 2)
        else:
            for i in range(n_layers):
                Nparams += (nqbts - 1)
        ### Get VQE angles for HR dist calculation
        folder = os.path.join(parent_dir, "noiseless_opts/"+str(nqbts)+"_qubits_TFIM_X+0.5ZZ/" \
                            +str(nqbts)+"qubits_"+str(Nparams)+"params")
        os.chdir(folder)
        angles = []
        for file in os.listdir(folder):
            if file.endswith("angles_file.dat"):
                with open(file, "rb") as angles_file:
                    angles = pickle.load(angles_file)
                angles_file.close()
            if file.endswith("estimates.dat"):
                with open(file, "rb") as estimates_file:
                    energies = pickle.load(estimates_file)
                estimates_file.close()

        qubit_dir = os.path.join(parent_dir,"correlation_folder/"+str(nqbts)+"_q")

        if os.path.exists(qubit_dir) == False:
            print(True)
            os.mkdir(qubit_dir)

        p1_str = str(.01) ## 99% P1 Value
        y_axis_start = 10**-4 ## P2 Start
        y_axis_end = .2 ## P2 End
        p2_arr = np.linspace(y_axis_start, y_axis_end, 50) ## P2 Values

        for i, p2 in enumerate(p2_arr):
            
            p2_str = str(p2)

            os.chdir(qubit_dir)
            ### Check if the data exists already
            if os.path.exists(p2_str) == False:
                os.mkdir(p2_str)
                os.chdir(p2_str)
                with open("angles_file.dat", "wb") as angles_file:
                    pickle.dump(angles, angles_file)
                angles_file.close()

                os.chdir(parent_dir)
                print(p2_str)

                ### Run Simulation
                run_arr = ['python3.9', 'real_noise_VQE_noiseless.py', '--J', '0.5', '--shots', '10000', '--n_qbts', str(nqbts),\
                "--init_param", str(os.path.join(qubit_dir,p2_str +"/angles_file.dat")),\
                "--start_idx", str((len(angles)-num_points-1)),"--fid", "0", "--p1", p1_str, "--p2", p2_str,\
                "--output_dir", str(os.path.join(qubit_dir,p2_str))]
                p = subprocess.Popen(run_arr)
                p.wait()
            else:
                os.chdir(p2_str)
            # print(value)

            ### Get HR distance Average for P1 P2 value
            with open(os.path.join(qubit_dir,p2_str + "/HR_dist_hist.pkl"), "rb") as HR_file:
                HR_collection[nqbts-4, i] = np.average(pickle.load(HR_file))
            HR_file.close()

            ### Get Fidelity Average for P1 P2 value
            with open(os.path.join(qubit_dir,p2_str + "/fid_hist.pkl"), "rb") as Fid_file:
                arr = np.array(pickle.load(Fid_file))
                print(arr)
                Fid_collection[nqbts-4, i] = np.average(arr)
            Fid_file.close()

        ### For each n_qbt value run regression on 50 points
        slope, intercept, r_value, p_value, std_err = stats.linregress(Fid_collection[nqbts-4], HR_collection[nqbts-4])
        print("R-Squared Val", r_value**2)
        correlations[nqbts-4] = r_value

    ### Dump Correlations
    os.chdir(parent_dir)
    with open("correlations.pkl", "wb") as corr_file:
        pickle.dump(correlations, corr_file)
    corr_file.close()


### Load Correlations and make plot
with open("correlations.pkl", "rb") as corr_file:
        correlations = pickle.load(corr_file)
corr_file.close()

print(correlations)

fig = plt.figure(figsize=(6,6))
ax = plt.axes()
ax.plot(range(4, 12), correlations[:8], c = 'cornflowerblue', marker = ".",  zorder = 1)
ax.set_ylim([-1.0, -0.9])
ax.set_xlabel('Number of Qubits',fontsize = 12)
ax.set_ylabel("Fidelity",fontsize = 12)
ax.set_title("Noise")
ax.set_xlabel("Number of Qubits")
ax.set_ylabel("Correlation Between Fidelity and HR Distance")
plt.savefig("correlations_ALA.svg",dpi = 300, bbox_inches='tight')
plt.clf()