import qiskit
from qiskit import QuantumCircuit, Aer
from qiskit.visualization import plot_histogram
from qiskit.tools.monitor import job_monitor
from azure.quantum.qiskit import AzureQuantumProvider
from qiskit import transpile
import numpy as np
import argparse
from utils import distanceVecFromSubspace, get_exp_cross
import pickle
import matplotlib.pyplot as plt
import os

HR_dist_hist = []

def get_args(parser):
    parser.add_argument('--input_dir', type = str, help = "directory where hyperparam_dict.npy exists and HR distances and plots will be stored")
    parser.add_argument('--nth', type=int, default=1, help ="for every nth energy, get HR distance (default: 1)")
    parser.add_argument('--load', type=int, default = 0, help = "if starting HR distance measurement from mid-terminated point args.load = 1, else 0 (default: 0)")
    args = parser.parse_args()
    return args

def Q_Circuit(N_qubits, var_params, h_l):
    circ = QuantumCircuit(N_qubits, N_qubits)
    param_idx = 0
    for i in range(N_qubits):
        circ.ry(var_params[param_idx], i)
        param_idx += 1
    for i in range(0, N_qubits, 2):
        circ.cx(i, i+1)
    for i in range(1, N_qubits-1, 2):
        circ.cx(i, i+1)
    for h_idx in h_l:
        circ.h(h_idx)
    return circ

def get_measurement(n_qbts, var_params, backend, h_l, hyperparam_dict, param_idx):
    measurement_path = os.path.join(args.input_dir, "measurement", f"{param_idx}th_param_{''.join([str(e) for e in h_l])}qbt_h_gate.npy")
    if os.path.exists(measurement_path):
        #no need to save as it is already saved
        measurement = np.load(measurement_path, allow_pickle = "True").item()
    else:
        circ = Q_Circuit(n_qbts, var_params, h_l)
        circ.measure(list(range(n_qbts)), list(range(n_qbts)))
        circ = transpile(circ, backend)
        job = backend.run(circ, shots = hyperparam_dict["shots"])
        if hyperparam_dict["backend"] != "aer_simulator":
            job_id = job.id()
            job_monitor(job)
        result = job.result()
        measurement = dict(result.get_counts())
        np.save(measurement_path, measurement)
    return measurement

def get_params(params_dir_path, param_idx):
    var_params = np.load(os.path.join(params_dir_path, f"var_params_{param_idx}.npy"))
    return var_params

def get_HR_distance(hyperparam_dict, exp_val_dict, param_idx, params_dir_path, backend):
    cov_mat = np.zeros((2,2))
    n_qbts = hyperparam_dict["n_qbts"]
    cov_mat[0, 0] = exp_val_dict["exp_X_sqr"] - exp_val_dict["exp_X"]**2
    cov_mat[1, 1] = exp_val_dict["exp_ZZ_sqr"] - exp_val_dict["exp_ZZ"]**2
    cross_val = 0
    z_indices = [[i, i+1] for i in range(n_qbts) if i != (n_qbts-1)]
    for h_idx in range(n_qbts):
        h_l = [h_idx]
        var_params = get_params(params_dir_path, param_idx)
        cross_m = get_measurement(n_qbts, var_params, backend, h_l, hyperparam_dict, param_idx)
        for z_ind in z_indices:
            if h_idx not in z_ind:
                indices = h_l + z_ind
                cross_val += get_exp_cross(cross_m, indices)
    cov_mat[0,1] = cross_val - exp_val_dict["exp_X"]*exp_val_dict["exp_ZZ"]
    cov_mat[1,0] = cov_mat[0,1]
    val, vec = np.linalg.eigh(cov_mat)
    argsort = np.argsort(val)
    val, vec = val[argsort], vec[:, argsort]
    orig_H = np.array([1, hyperparam_dict["J"]])
    orig_H = orig_H/np.linalg.norm(orig_H)
    HR_dist = distanceVecFromSubspace(orig_H, vec[:, :1])
    return HR_dist

def main(args):
    global HR_dist_hist
    provider = AzureQuantumProvider(resource_id = "/subscriptions/58687a6b-a9bd-4f79-b7af-1f8f76760d4b/resourceGroups/AzureQuantum/providers/Microsoft.Quantum/Workspaces/HamiltonianReconstruction",\
                                    location = "West US")
    if not os.path.exists(os.path.join(args.input_dir,"hyperparam_dict.npy")):
        raise ValueError( "input directory must be a valid input path that contains hyperparam_dict.npy")
    if not os.path.isdir(os.path.join(args.input_dir, "measurement")):
        os.makedirs(os.path.join(args.input_dir, "measurement"))
    #LOAD All the data provided here
    hyperparam_dict = np.load(os.path.join(args.input_dir, "hyperparam_dict.npy"), allow_pickle = True).item()
    print("This is hyperparameter dictionary loaded: ", hyperparam_dict)
    params_dir_path = os.path.join(args.input_dir,"params_dir")
    with open(os.path.join(args.input_dir, "exp_X_l.pkl"), "rb") as fp:
        exp_X_l = pickle.load(fp)
    with open(os.path.join(args.input_dir, "exp_ZZ_l.pkl"), "rb") as fp:
        exp_ZZ_l = pickle.load(fp)
    with open(os.path.join(args.input_dir, "exp_X_sqr_l.pkl"), "rb") as fp:
        exp_X_sqr_l = pickle.load(fp)
    with open(os.path.join(args.input_dir, "exp_ZZ_sqr_l.pkl"), "rb") as fp:
        exp_ZZ_sqr_l = pickle.load(fp)
    #Load backend
    print([backend.name() for backend in provider.backends()])
    backend_name = hyperparam_dict["backend"]
    if backend_name == "aer_simulator":
         backend = Aer.get_backend(backend_name)
    else:
        backend = provider.get_backend(backend_name)

    #get every nth HR distance
    if args.load == 1:
        #load HR distance
        with open(os.path.join(args.input_dir, "HR_dist_hist.pkl"), "rb") as fp:
            HR_dist_hist = pickle.load(fp)
        #THIS WILL NOT ALWAYS WORK....
        start_idx = len(HR_dist_hist)
    elif args.load == 0:
        start_idx = 0
    else:
        raise ValueError("args.load value must be 0 or 1")

    for param_idx in range(start_idx, len(exp_X_l),args.nth):
        exp_val_dict = {}
        exp_val_dict["exp_X"], exp_val_dict["exp_ZZ"] = exp_X_l[param_idx], exp_ZZ_l[param_idx]
        exp_val_dict["exp_X_sqr"], exp_val_dict["exp_ZZ_sqr"] =  exp_X_sqr_l[param_idx], exp_ZZ_sqr_l[param_idx]
        HR_dist = get_HR_distance(hyperparam_dict, exp_val_dict, param_idx, params_dir_path, backend)
        print("This is HR distance: ", HR_dist)
        HR_dist_hist.append(HR_dist)
        with open(os.path.join(args.input_dir, "HR_dist_hist.pkl"), "wb") as fp:
            pickle.dump(HR_dist_hist, fp)
    #get ground state Energy
    gst_E = hyperparam_dict["gst_E"]
    J = hyperparam_dict["J"]
    n_qbts = hyperparam_dict["n_qbts"]
    shots = hyperparam_dict["shots"]
    #reconstruct energy history
    E_hist = []
    for i in range(len(exp_X_l)):
        E_hist.append(exp_X_l[i] + J*exp_ZZ_l[i])


    fig, ax = plt.subplots()
    VQE_steps = np.array(list(range(len(E_hist))))
    ax.scatter(VQE_steps, E_hist, c = 'b', alpha = 0.8, marker = ".", label = "Energy")
    ax.set_xlabel('VQE Iterations')
    ax.set_ylabel("Energy")
    ax.legend(bbox_to_anchor=(1.28, 1.30), fontsize = 10)
    title = "VQE 1-D "+ str(n_qbts) +" qubits TFIM" + "\n" + f"J: {J}, shots: {shots}" + '\n' + 'True Ground energy: ' + \
            str(round(gst_E, 3)) + '\n' + 'Estimated Ground Energy: '+ str(round(float(min(E_hist)), 3))  + '\n' "Backend name: " + backend_name
    plt.title(title, fontdict = {'fontsize' : 15})
    ax2 = ax.twinx()
    ax2.scatter(list(range(0, len(E_hist), args.nth)), HR_dist_hist, c = 'r', alpha = 0.8, marker=".", label = "HR distance")
    ax2.set_ylabel("HR distance")
    ax2.legend(bbox_to_anchor=(1.28, 1.22), fontsize = 10)
    plt.savefig(args.input_dir+'/'+  str(n_qbts)+"qubits_" + f"layers_shots_{shots}_HR_dist.png", dpi = 300, bbox_inches='tight')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "VQE for 1-D TFIM with non-periodic boundary condition")
    args = get_args(parser)
    main(args)
