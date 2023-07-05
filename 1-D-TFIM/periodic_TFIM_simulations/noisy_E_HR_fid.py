import sys
import qiskit
from qiskit import QuantumCircuit, Aer
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
from qiskit.tools.monitor import job_monitor
from azure.quantum.qiskit import AzureQuantumProvider
from qiskit import transpile
import numpy as np
import argparse
import pickle
import matplotlib.pyplot as plt
import os
from utils_periodic import distanceVecFromSubspace, get_exp_cross, get_exp_X, get_exp_ZZ, get_fidelity, get_Hamiltonian


def Q_Circuit(N_qubits, var_params, h_l, n_layers):
    circ = QuantumCircuit(N_qubits, N_qubits)
    param_idx = 0
    for i in range(N_qubits):
        circ.h(i)
    for layer in range(n_layers):
        for j in range(layer%2, N_qubits, 2):
            circ.cx(j%N_qubits, (j+1)%N_qubits)
            circ.ry(var_params[param_idx], j%N_qubits)
            param_idx += 1
            circ.ry(var_params[param_idx], (j+1)%N_qubits)
            param_idx += 1
    for h_idx in h_l:
        circ.h(h_idx)
    return circ

def get_args(parser):
    parser.add_argument('--input_dir', type = str, help = "directory where HR_hyperparam_dict.npy, parameter directory, and measurement directory exists. HR distances and plots will be stored in this given path.")
    parser.add_argument('--p1', type = float, help = "p1 depolarization value (overrides p1 value in HR_hyperparam_dict.npy)")
    parser.add_argument('--p2', type = float, help = "p2 depolarization value (overrides p2 value in HR_hyperparam_dict.npy)")
    args = parser.parse_args()
    return args

def get_fid(hyperparam_dict, param_idx, params_dir_path, ground_state, backend):
    var_params = get_params(params_dir_path, param_idx)
    n_qbts = hyperparam_dict["n_qbts"]
    circ = Q_Circuit(n_qbts, var_params, [], hyperparam_dict["n_layers"])
    if hyperparam_dict['p1'] == 0 and hyperparam_dict['p2'] == 0:
        circ.save_statevector()
        result = backend.run(circ).result()
        statevector = result.get_statevector(circ)
        statevector = np.array(statevector)
        fid = np.absolute(np.vdot(statevector, ground_state))
    else:
        circ.save_density_matrix()
        result = backend.run(circ).result()
        den_mat = result.data(0)['density_matrix']
        fid = get_fidelity(ground_state, den_mat)
    return fid.real

def get_measurement(n_qbts, var_params, backend, h_l, hyperparam_dict, param_idx):
    num_shots = hyperparam_dict["shots"]
    backendnm = hyperparam_dict["backend"]
    p1, p2 = hyperparam_dict["p1"], hyperparam_dict["p2"]
    measurement_path = os.path.join(args.input_dir, "measurement", f"{param_idx}th_param_{''.join([str(e) for e in h_l])}qbt_h_gate.npy")
    if os.path.exists(measurement_path):
        #no need to save as it is already saved
        measurement = np.load(measurement_path, allow_pickle = "True").item()
    else:
        raise ValueError("Doesn't have measurement for corresponding idx")
    return measurement

def get_params(params_dir_path, param_idx):
    var_params = np.load(os.path.join(params_dir_path, f"var_params_{param_idx}.npy"))
    return var_params

def get_noisy_E(hyperparam_dict, param_idx, params_dir_path, backend):
    """
    Obtain Energy values obtained from hardware runs / simulations with depolarization and shot noise.
    """
    n_qbts = hyperparam_dict["n_qbts"]
    z_l, x_l = [], [i for i in range(n_qbts)]
    var_params = get_params(params_dir_path, param_idx)
    z_m = get_measurement(n_qbts, var_params, backend, z_l, hyperparam_dict, param_idx)
    x_m = get_measurement(n_qbts, var_params, backend, x_l, hyperparam_dict, param_idx)
    E = get_exp_X(x_m, 1) + hyperparam_dict["J"]*get_exp_ZZ(z_m, 1)
    print("This is noisy energy: ", E)
    return E

def get_gst(n_qbts, J):
    Ham = get_Hamiltonian(n_qbts, J)
    eigen_values, eigen_vectors = np.linalg.eig(Ham)
    Ground_energy = np.amin(eigen_values)
    index = np.argwhere(eigen_values == Ground_energy)[0][0]
    #Obtain ground state's wavefunction
    ground_wf = eigen_vectors[:, index]
    ground_wf = (1./np.linalg.norm(ground_wf))*ground_wf
    return ground_wf

def main(args):
    HR_hyperparam_dict_path = os.path.join(args.input_dir, "HR_hyperparam_dict.npy")
    if not os.path.exists(HR_hyperparam_dict_path):
        raise ValueError( "args.input_dir must be a valid input path that contains HR_hyperparam_dict.npy")

    #LOAD the hyperparamter data from HR here
    hyperparam_dict = np.load(HR_hyperparam_dict_path, allow_pickle = True).item()
    params_dir_path = os.path.join(args.input_dir,"params_dir")

    #args.p1 and args.p2 always override hyperparmeter
    hyperparam_dict["p1"], hyperparam_dict["p2"]  = args.p1, args.p2
    shots, backend, p1, p2 = hyperparam_dict["shots"], hyperparam_dict["backend"], hyperparam_dict["p1"], hyperparam_dict["p2"]

    if not os.path.isdir(os.path.join(args.input_dir, "measurement")):
        raise ValueError("measurement directory does not exist in args.input_dir")

    with open(os.path.join(args.input_dir, "E_hist.pkl"), "rb") as fp:
        E_hist = pickle.load(fp)

    gst_E = hyperparam_dict["gst_E"]
    n_qbts = hyperparam_dict["n_qbts"]
    J = hyperparam_dict["J"]
    n_layers = hyperparam_dict["n_layers"]

    noisy_E_hist = []
    fid_hist = []

    param_idx_l_path = os.path.join(args.input_dir, "param_idx_l.npy")

    if os.path.isfile(param_idx_l_path):
        param_idx_l = np.load(param_idx_l_path, allow_pickle = True)
    else:
        param_idx_l = list(range(len(E_hist)))
    print(param_idx_l)

    with open(os.path.join(args.input_dir, "HR_dist_hist.pkl"), "rb") as fp:
        HR_dist_hist = pickle.load(fp)

    img_name = f"layers_shots_{shots}_shots_{backend}_p1_{p1}_p2_{p2}_noisy_E_HR_fid.svg"

    #calculate noisy_E_hist and save it
    for param_idx in param_idx_l:
        noisy_E_hist.append(get_noisy_E(hyperparam_dict, param_idx, params_dir_path, backend))
        with open(os.path.join(args.input_dir, f"noisy_E_hist.pkl"), "wb") as fp:
            pickle.dump(noisy_E_hist, fp)

    #calculate fidelity
    #get_gst
    gst_path = os.path.join(args.input_dir, "gst.npy")
    if os.path.isfile(gst_path):
        gst = np.load(gst_path, allow_pickle = True)
    else:
        gst = get_gst(n_qbts, J)
        np.save(gst_path, gst)
    #backend initialization for fidelity
    if p1 == 0 and p2 == 0:
        fid_backend = AerSimulator()
    else:
        noise_model = NoiseModel()
        p1_error = depolarizing_error(hyperparam_dict["p1"], 1)
        p2_error = depolarizing_error(hyperparam_dict["p2"], 2)
        noise_model.add_all_qubit_quantum_error(p1_error, ['h','ry'])
        noise_model.add_all_qubit_quantum_error(p2_error, ['cx'])
        fid_backend = AerSimulator(method = 'density_matrix', noise_model = noise_model)

    for param_idx in param_idx_l:
        fid = get_fid(hyperparam_dict, param_idx, params_dir_path, gst, fid_backend)
        print(f"{param_idx}th fidelity: ", fid)
        fid_hist.append(fid)
        with open(os.path.join(args.input_dir, f"fid_hist.pkl"), "wb") as fp:
            pickle.dump(fid_hist, fp)

    #create plots
    fig, ax = plt.subplots()
    ax.scatter(param_idx_l, noisy_E_hist, c = 'b', alpha = 0.8, marker = ".", label = "Energy")
    ax.set_xlabel('VQE Iterations')
    ax.set_ylabel("Energy")
    ax.legend(bbox_to_anchor=(1.28, 1.30), fontsize = 10)
    title = "VQE 1-D "+ f"1-D TFIM {n_qbts} qubits \n" + f"J: {J}, shots: {shots}" + \
                    '\n' + 'True Ground energy: ' + str(round(gst_E, 3))
    if not (hyperparam_dict['p1'] == 0 and hyperparam_dict['p2'] == 0):
        title = title + '\n' + f"p1: {p1}, p2: {p2}"
    plt.title(title, fontdict = {'fontsize' : 15})
    ax2 = ax.twinx()
    ax2.scatter(param_idx_l, HR_dist_hist, c = 'r', alpha = 0.8, marker=".", label = "HR distance")
    ax2.scatter(param_idx_l, fid_hist, c = 'g', alpha = 0.8, marker=".", label = "Fidelity")
    ax2.set_ylabel("HR distance | Fidelity")
    ax2.legend(bbox_to_anchor=(1.28, 1.22), fontsize = 10)
    plt.savefig(args.input_dir+'/'+  str(n_qbts)+"qubits_"+ str(n_layers)+img_name, dpi = 300, bbox_inches='tight')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "HR for 2-D J1-J2 model")
    args = get_args(parser)
    main(args)
