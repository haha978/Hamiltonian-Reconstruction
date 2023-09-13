import sys
sys.path.insert(0, "../")
from utils import get_Hx, get_Hzz
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
from qiskit.algorithms.optimizers import IMFIL
from functools import partial
import pickle
import matplotlib.pyplot as plt
import os

E_hist = []

def get_args(parser):
    parser.add_argument('--n_qbts', type = int, default = 6, help = "number of qubits (default: 6)")
    parser.add_argument('--J', type = float, default = 0.5, help = "(default J: 0.5)")
    parser.add_argument('--max_iter', type = int, default = 10000, help = "maximum number of iterations (default: 10000)")
    parser.add_argument('--n_layers', type = int, default = 3, help = "number of ALA ansatz layers needed (default: 3)")
    parser.add_argument('--output_dir', type = str, default = ".", help = "output directory being used (default: .)")
    parser.add_argument('--init_param', type = str, default = "NONE", help = "parameters for initialization (default: NONE)")
    args = parser.parse_args()
    return args

def Q_Circuit(N_qubits, var_params):
    circ = QuantumCircuit(N_qubits, N_qubits)
    param_idx = 0
    for i in range(N_qubits):
        circ.h(i)
    if N_qubits % 2 == 0:
        for layer in range(args.n_layers):
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
        for layer in range(args.n_layers):
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
    return circ

def get_E(var_params, n_qbts, J, backend, Hx, Hzz):
    circ = Q_Circuit(n_qbts, var_params)
    circ.save_statevector()
    circ = transpile(circ, backend)
    job = backend.run(circ)
    result = job.result()
    wf = result.get_statevector(circ)
    E = wf.expectation_value(Hx).real + J * wf.expectation_value(Hzz).real
    E_hist.append(E)
    with open(os.path.join(args.output_dir, "E_hist.pkl"), "wb") as fp:
        pickle.dump(E_hist, fp)
    np.save(os.path.join(args.output_dir, "params_dir", f"var_params_{len(E_hist)-1}.npy"), var_params)
    print("This is energy: ", E)
    return E

def main(args):
    if not os.path.exists(os.path.join(args.output_dir,"params_dir")):
        os.makedirs(os.path.join(args.output_dir,"params_dir"))

    Nparams = 0
    if args.n_qbts % 2 == 0:
        for i in range(args.n_layers):
            if i % 2 == 0:
                Nparams += args.n_qbts
            else:
                Nparams += (args.n_qbts - 2)
    else:
        for i in range(args.n_layers):
            Nparams += (args.n_qbts - 1)

    if args.init_param == "NONE":
        var_params = np.random.uniform(low = -np.pi, high = np.pi, size = Nparams)
    else:
        param_PATH = os.path.join(args.init_param)
        var_params = np.load(param_PATH)
        assert len(var_params) == Nparams, "loaded params needs to have the same length as the Nparams"

    bounds = np.tile(np.array([-np.pi, np.pi]), (Nparams,1))

    try:
        dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        gst_E = np.load(os.path.join(dir_path, f"gst_E_dict_J_{args.J}_no_periodic.npy"), allow_pickle = True).item()[args.n_qbts]
    except:
        raise ValueError(f"no corresponding index to ground state energy J value to {args.n_qbts} qubits")

    hyperparam_dict = {}
    hyperparam_dict["n_qbts"], hyperparam_dict["J"] = args.n_qbts, args.J
    hyperparam_dict["n_layers"] = args.n_layers
    hyperparam_dict["gst_E"] = gst_E
    hyperparam_dict["p1"] = 0
    hyperparam_dict["p2"] = 0

    #get ops_l
    Hx, Hzz = get_Hx(args.n_qbts), get_Hzz(args.n_qbts)

    backend = Aer.get_backend('aer_simulator_statevector')
    title = "VQE 1-D "+ str(args.n_qbts) +" qubits TFIM" + "\n" + f"J: {args.J}" + '\n' + 'True Ground energy: ' + \
            str(round(gst_E, 3)) + '\n'

    imfil = IMFIL(maxiter = args.max_iter)
    get_E_func = partial(get_E, n_qbts = args.n_qbts, J = args.J, backend = backend, Hx = Hx, Hzz = Hzz)
    result = imfil.minimize(get_E_func, x0 = var_params, bounds = bounds)
    fig, ax = plt.subplots()
    VQE_steps = np.array(list(range(len(E_hist))))
    title += 'Estimated Ground Energy: '+ str(round(float(min(E_hist)), 3))
    ax.scatter(VQE_steps, E_hist, c = 'b', alpha = 0.8, marker = ".", label = "Energy")
    ax.set_xlabel('VQE Iterations')
    ax.set_ylabel("Energy")
    ax.legend(bbox_to_anchor=(1.28, 1.30), fontsize = 10)
    plt.title(title, fontdict = {'fontsize' : 15})
    plt.savefig(args.output_dir+'/'+  str(args.n_qbts)+"qubits_"+ str(args.n_layers)+f"layers.png", dpi = 300, bbox_inches='tight')
    # Save hyperparam_dict for Hamiltonian Reconstruction
    np.save(os.path.join(args.output_dir, "VQE_hyperparam_dict.npy"), hyperparam_dict)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "VQE for 1-D TFIM with non-periodic boundary condition")
    args = get_args(parser)
    main(args)
