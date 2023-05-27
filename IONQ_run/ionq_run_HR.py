import qiskit
from qiskit import QuantumCircuit, Aer
from qiskit.visualization import plot_histogram
from qiskit.tools.monitor import job_monitor
from azure.quantum.qiskit import AzureQuantumProvider
from qiskit import transpile
import numpy as np
import argparse
from utils import distanceVecFromSubspace, get_exp_cross, get_exp_X, get_exp_ZZ
import pickle
import matplotlib.pyplot as plt
import os

HR_dist_hist = []

def get_args(parser):
    parser.add_argument('--input_dir', type = str, help = "directory where hyperparam_dict.npy exists and HR distances and plots will be stored")
    parser.add_argument('--shots', type=int, default=1000, help = "number of shots during HamiltonianReconstuction (default: 1000)")
    parser.add_argument('--backend', type = str, default = "aer_simulator", help = "backend for ionq runs (aer_simulator, ionq.simulator, ionq.qpu, ionq.qpu.aria-1, default = aer_simulator)")
    parser.add_argument('--param_idx_l', action = 'store_true', help = "if there is param_idx_l, then use param_idx_l.npy in input_dir \
                                to load the parameter index list to measure corresponding HR distances")
    args = parser.parse_args()
    return args

def Q_Circuit(N_qubits, var_params, h_l, n_layers):
    circ = QuantumCircuit(N_qubits, N_qubits)
    param_idx = 0
    for i in range(N_qubits):
        circ.h(i)
    if N_qubits % 2 == 0:
        for layer in range(n_layers):
            if layer % 2 == 0:
                for i in range(0, N_qubits, 2):
                    circ.cx(i, i+1)
                for i in range(N_qubits):
                    circ.ry(var_params[param_idx], i)
                    param_idx += 1
            else:
                for i in range(1, N_qubits-1, 2):
                    circ.cx(i, i+1)
                for i in range(1, N_qubits-1):
                    circ.ry(var_params[param_idx], i)
                    param_idx += 1
    else:
        for layer in range(n_layers):
            if layer % 2 == 0:
                for i in range(0, N_qubits-1, 2):
                    circ.cx(i, i+1)
                for i in range(N_qubits-1):
                    circ.ry(var_params[param_idx], i)
                    param_idx += 1
            else:
                for i in range(1, N_qubits, 2):
                    circ.cx(i, i+1)
                for i in range(1, N_qubits):
                    circ.ry(var_params[param_idx], i)
                    param_idx += 1
    for h_idx in h_l:
        circ.h(h_idx)
    return circ

def get_measurement(n_qbts, var_params, backend, h_l, hyperparam_dict, param_idx):
    measurement_path = os.path.join(args.input_dir, "measurement", f"{param_idx}th_param_{''.join([str(e) for e in h_l])}qbt_h_gate.npy")
    if os.path.exists(measurement_path):
        #no need to save as it is already saved
        measurement = np.load(measurement_path, allow_pickle = "True").item()
    else:
        circ = Q_Circuit(n_qbts, var_params, h_l, hyperparam_dict["n_layers"])
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

def get_HR_distance(hyperparam_dict, param_idx, params_dir_path, backend):
    cov_mat = np.zeros((2,2))
    n_qbts = hyperparam_dict["n_qbts"]
    #need to delete the below as well
    z_l, x_l = [], [i for i in range(n_qbts)]
    var_params = get_params(params_dir_path, param_idx)
    z_m = get_measurement(n_qbts, var_params, backend, z_l, hyperparam_dict, param_idx)
    x_m = get_measurement(n_qbts, var_params, backend, x_l, hyperparam_dict, param_idx)
    exp_X, exp_ZZ = get_exp_X(x_m, 1),  get_exp_ZZ(z_m, 1)
    cov_mat[0, 0] =  get_exp_X(x_m, 2) - exp_X**2
    cov_mat[1, 1] = get_exp_X(z_m, 2) - exp_ZZ**2
    cross_val = 0
    z_indices = [[i, i+1] for i in range(n_qbts) if i != (n_qbts-1)]
    for h_idx in range(n_qbts):
        h_l = [h_idx]
        cross_m = get_measurement(n_qbts, var_params, backend, h_l, hyperparam_dict, param_idx)
        for z_ind in z_indices:
            if h_idx not in z_ind:
                indices = h_l + z_ind
                cross_val += get_exp_cross(cross_m, indices)
    cov_mat[0,1] = cross_val - exp_X*exp_ZZ
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
    hyperparam_dict_loaded = np.load(os.path.join(args.input_dir, "hyperparam_dict.npy"), allow_pickle = True).item()
    params_dir_path = os.path.join(args.input_dir,"params_dir")
    backend_name = args.backend
    if backend_name == "aer_simulator":
         backend = Aer.get_backend(backend_name)
    else:
        backend = provider.get_backend(backend_name)

    hyperparam_dict = {}
    hyperparam_dict["gst_E"] = hyperparam_dict_loaded["gst_E"]
    hyperparam_dict["J"] = hyperparam_dict_loaded["J"]
    hyperparam_dict["n_qbts"] = hyperparam_dict_loaded["n_qbts"]
    hyperparam_dict["n_layers"] = hyperparam_dict_loaded["n_layers"]
    #need to use a new backend
    hyperparam_dict["shots"] = args.shots
    hyperparam_dict["backend"] = backend_name
    print("This is hyperparameter dictionary newly constructed: ", hyperparam_dict)
    #number of shots
    shots = args.shots
    #get ground state Energy
    gst_E = hyperparam_dict["gst_E"]
    J = hyperparam_dict["J"]
    n_qbts = hyperparam_dict["n_qbts"]
    n_layers = hyperparam_dict["n_layers"]

    with open(os.path.join(args.input_dir, "E_hist.pkl"), "rb") as fp:
        E_hist = pickle.load(fp)

    #TODO need to load with special purposes
    if args.param_idx_l:
        param_idx_l_path = os.path.join(args.input_dir, "param_idx_l.npy")
        assert os.path.isfile(param_idx_l_path), "there is no param_idx_l.npy file in input_dir"
        param_idx_l = np.load(param_idx_l_path, allow_pickle = "True")
        print("This is parameters we are sampling: ", param_idx_l)
    else:
        param_idx_l = list(range(len(E_hist)))

    #get every nth HR distance
    for param_idx in param_idx_l:
        HR_dist = get_HR_distance(hyperparam_dict, param_idx, params_dir_path, backend)
        print("This is HR distance: ", HR_dist)
        HR_dist_hist.append(HR_dist)
        with open(os.path.join(args.input_dir, "HR_dist_hist.pkl"), "wb") as fp:
            pickle.dump(HR_dist_hist, fp)

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
    ax2.scatter(param_idx_l, HR_dist_hist, c = 'r', alpha = 0.8, marker=".", label = "HR distance")
    ax2.set_ylabel("HR distance")
    ax2.legend(bbox_to_anchor=(1.28, 1.22), fontsize = 10)
    plt.savefig(args.input_dir+'/'+  str(n_qbts)+"qubits_"+ str(n_layers)+f"layers_shots_{shots}_HR_dist.png", dpi = 300, bbox_inches='tight')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "HR for 1-D TFIM with non-periodic boundary condition")
    args = get_args(parser)
    main(args)
