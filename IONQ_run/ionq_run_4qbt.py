import qiskit
from qiskit import QuantumCircuit, Aer
from qiskit.visualization import plot_histogram
from qiskit.tools.monitor import job_monitor
from azure.quantum.qiskit import AzureQuantumProvider
from qiskit import transpile
import numpy as np
import argparse
from utils import get_exp_X, get_exp_ZZ
from qiskit.algorithms.optimizers import IMFIL
from functools import partial
import pickle
import matplotlib.pyplot as plt
import os

E_hist = []
exp_X_l = []
exp_ZZ_l = []
exp_X_sqr_l = []
exp_ZZ_sqr_l = []

def get_args(parser):
    parser.add_argument('--n_qbts', type = int, default = 4, help = "number of qubits (default: must be 4)")
    parser.add_argument('--J', type = float, default = 0.5, help = "(default J: 0.5)")
    parser.add_argument('--shots', type = int, default = 10000, help = "Number of shots (default: 10000)")
    parser.add_argument('--max_iter', type = int, default = 10000, help = "maximum number of iterations (default: 10000)")
    parser.add_argument('--output_dir', type = str, default = ".", help = "output directory being used (default: .)")
    parser.add_argument('--init_param', type = str, default = "NONE", help = "parameters for initialization (default: NONE)")
    parser.add_argument('--backend', type = str, default = "aer_simulator", help = "backend for ionq runs (aer_simulator, ionq.simulator, ionq.qpu, ionq.qpu.aria-1, default = aer_simulator)")
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

def get_measurement(n_qbts, var_params, backend, shots, h_l):
    circ = Q_Circuit(n_qbts, var_params, h_l)
    circ.measure(list(range(n_qbts)), list(range(n_qbts)))
    circ = transpile(circ, backend)
    job = backend.run(circ, shots = shots)
    if args.backend != "aer_simulator":
        job_id = job.id()
        job_monitor(job)
    result = job.result()
    measurement = dict(result.get_counts())
    return measurement

def get_E(var_params, n_qbts, shots, J, backend):
    z_l, x_l = [], [i for i in range(n_qbts)]
    z_m = get_measurement(n_qbts, var_params, backend, shots, z_l)
    x_m = get_measurement(n_qbts, var_params, backend, shots, x_l)
    # maybe save x_m and z_m for future.
    exp_X, exp_ZZ = get_exp_X(x_m, 1), get_exp_ZZ(z_m, 1)
    exp_X_sqr, exp_ZZ_sqr = get_exp_ZZ(x_m, 2), get_exp_ZZ(z_m, 2)
    E = exp_X + J * exp_ZZ
    E_hist.append(E)
    exp_X_l.append(exp_X)
    exp_ZZ_l.append(exp_ZZ)
    exp_X_sqr_l.append(exp_X_sqr)
    exp_ZZ_sqr_l.append(exp_ZZ_sqr)
    with open(os.path.join(args.output_dir, "exp_X_l.pkl"), "wb") as fp:
        pickle.dump(exp_X_l, fp)
    with open(os.path.join(args.output_dir, "exp_ZZ_l.pkl"), "wb") as fp:
        pickle.dump(exp_ZZ_l, fp)
    with open(os.path.join(args.output_dir, "exp_X_sqr_l.pkl"), "wb") as fp:
        pickle.dump(exp_X_sqr_l, fp)
    with open(os.path.join(args.output_dir, "exp_ZZ_sqr_l.pkl"), "wb") as fp:
        pickle.dump(exp_ZZ_sqr_l, fp)
    np.save(os.path.join(args.output_dir, "params_dir", f"var_params_{len(E_hist)-1}.npy"), var_params)
    print("This is energy: ", E)
    return E

def main(args):
    if not os.path.exists(os.path.join(args.output_dir,"params_dir")):
        os.makedirs(os.path.join(args.output_dir,"params_dir"))
    Nparams = 4
    if args.init_param == "NONE":
        var_params = np.random.uniform(low = -np.pi, high = np.pi, size = Nparams)
    else:
        param_PATH = os.path.join(args.init_param)
        var_params = np.load(param_PATH)
        assert (len(var_params) == Nparams), "loaded params needs to have the same length as the Nparams"
    bounds = np.tile(np.array([-np.pi, np.pi]),(Nparams,1))

    if args.backend == "aer_simulator":
        backend = Aer.get_backend('aer_simulator')
    else:
        provider = AzureQuantumProvider(resource_id = "/subscriptions/58687a6b-a9bd-4f79-b7af-1f8f76760d4b/resourceGroups/AzureQuantum/providers/Microsoft.Quantum/Workspaces/HamiltonianReconstruction",\
                                        location = "West US")
        backend = provider.get_backend(args.backend)
    print("Using backend from qiskit: ", args.backend)
    imfil = IMFIL(maxiter = args.max_iter)
    get_E_func = partial(get_E, n_qbts = args.n_qbts, shots = args.shots, J = args.J, backend = backend)
    result = imfil.minimize(get_E_func, x0 = var_params, bounds = bounds)
    fig, ax = plt.subplots()
    VQE_steps = np.array(list(range(len(E_hist))))
    ax.scatter(VQE_steps, E_hist, c = 'b', alpha = 0.8, marker = ".", label = "Energy")
    ax.set_xlabel('VQE Iterations')
    ax.set_ylabel("Energy")
    ax.legend(bbox_to_anchor=(1.28, 1.30), fontsize = 10)
    try:
        gst_E = np.load(f"/root/research/HR/31AUG2022/readout_noise_simulation/gst_E_dict_J_{args.J}_no_periodic.npy",allow_pickle = True).item()[args.n_qbts]
    except:
        raise ValueError(f"no corresponding index to ground state energy J value to {args.n_qbts} qubits")
    title = "VQE 1-D "+ str(args.n_qbts) +" qubits TFIM" + "\n" + f"J: {args.J}, shots: {args.shots}" + '\n' + 'True Ground energy: ' + \
            str(round(gst_E, 3)) + '\n' + 'Estimated Ground Energy: '+ str(round(float(min(E_hist)), 3)) + '\n' "Backend name: " + args.backend
    plt.title(title, fontdict = {'fontsize' : 15})
    plt.savefig(args.output_dir+'/'+  str(args.n_qbts)+f"qubits_shots_{args.shots}.png", dpi = 300, bbox_inches='tight')
    #Create hyperparam_dict for Hamiltonian Reconstruction
    hyperparam_dict = {}
    hyperparam_dict["n_qbts"], hyperparam_dict["J"] = args.n_qbts, args.J
    hyperparam_dict["shots"] = args.shots
    hyperparam_dict["backend"], hyperparam_dict["gst_E"] = args.backend, gst_E
    np.save(os.path.join(args.output_dir, "hyperparam_dict.npy"), hyperparam_dict)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "VQE for 1-D TFIM with non-periodic boundary condition")
    args = get_args(parser)
    main(args)
