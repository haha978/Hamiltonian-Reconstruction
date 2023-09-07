import os
os.environ["OMP_NUM_THREADS"] = "24"
os.environ["OPENBLAS_NUM_THREADS"] = "24"
from qiskit import QuantumCircuit, execute, Aer, quantum_info
print(Aer.backends())
from qiskit.visualization import plot_histogram
import qiskit.providers.aer.noise as noise
from qiskit import transpile
import numpy as np
from qiskit.algorithms.optimizers import SPSA
import argparse
from functools import partial
import numpy as np
import pickle
import cirq
import matplotlib.pyplot as plt
#from scipy.linalg import sqrtm
from utils import get_exp_X, get_exp_ZZ, mem_to_mem_num, \
            get_Hamiltonian, get_GST_E_and_wf, get_xz_key, \
            get_exp_cross, distanceVecFromSubspace, \
            get_Hamiltonian_gpu,get_GST_E_and_wf_gpu
from utils import get_Hx, get_Hzz
from qiskit.algorithms.optimizers import IMFIL
import matplotlib.pyplot as plt

param_hist = []
E_hist = []
HR_dist_hist_noiseless = []
fid_hist = []
m_dict_idx = 0

def get_args(parser):
    parser.add_argument('--n_qbts', type = int, default = 6, help = "number of qubits (default: 6)")
    parser.add_argument('--J', type = float, default = 0.5, help = "(default J: 0.5)")
    parser.add_argument('--shots', type = int, default = 10000, help = "Number of shots (default: 10000)")
    parser.add_argument('--p1', type = float, default = 0.001, help = "probability of one-qubit gate depolarization noise (p1: 0.001)")
    parser.add_argument('--p2', type = float, default = 0.01, help = "probability of two-qubit gate depolarization noise (p2: 0.01)")
    parser.add_argument('--output_dir', type = str, default = ".", help = "output directory being used (default: .)")
    parser.add_argument('--init_param', type = str, default = "NONE", help = "parameters for initialization (default: NONE)")
    parser.add_argument('--start_idx', type = int, default = 0, help = "start index of init_param")
    parser.add_argument('--end_idx', type = int, default = -1, help = "end index of init_param")
    parser.add_argument('--gpu', type = int, default = -1, help = "gpu = -1 if using cpu, else gpu indicates the gpu number used")
    parser.add_argument('--fid', type = int, default = -1, help = "fidelity, -1 = no, else yes")
    args = parser.parse_args()
    return args

# def Q_Circuit(N_qubits, var_params):
#     circ = QuantumCircuit(N_qubits)
#     param_idx = 0
#     for i in range(N_qubits):
#         circ.h(i)
#     a = len(var_params)//(2*N_qubits - 2)
#     b = len(var_params)%(2*N_qubits - 2)
#     if b == 0:
#         n_layers = 2*a
#     else:
#         assert(b == N_qubits), "wrong number of parameters"
#         n_layers = 2*a + 1
#     for layer in range(n_layers):
#         if layer % 2 == 0:
#             for i in range(0, N_qubits, 2):
#                 circ.cx(i, i+1)
#             for i in range(N_qubits):
#                 circ.ry(var_params[param_idx], i)
#                 param_idx += 1
#         else:
#             for i in range(1, N_qubits-1, 2):
#                 circ.cx(i, i+1)
#             for i in range(1, N_qubits-1):
#                 circ.ry(var_params[param_idx], i)
#                 param_idx += 1
#     return circ

# def expected_op_test(op, d_m):
#     return d_m.expectation_value(op, qargs=None)

# def expected_op1_op2_test(op1, op2, d_m):
#     return d_m.expectation_value(np.matmul(op1,op2), qargs=None)

def expected_op(op, d_m):
    return np.trace(np.matmul(op, d_m)).real

def expected_op1_op2(op1, op2, d_m):
    return np.trace(np.matmul(op1, np.matmul(op2, d_m))).real

def Q_Circuit(N_qubits, var_params):

    circ = QuantumCircuit(N_qubits, N_qubits)
    param_idx = 0
    n_layers = int(len(var_params) / N_qubits)
    for i in range(N_qubits):
        circ.h(i)

    for layer_idx in range(n_layers):
        
        for i in range(N_qubits-1):
            circ.cx(i, i+1)
            circ.ry(var_params[param_idx], i+1)
            circ.cx(i, i+1)
            param_idx += 1
        circ.ry(var_params[param_idx], 0)
        param_idx += 1
        
    return circ

def get_density_matrix(n_qbts, var_params, backend, noise_model):
    circ = Q_Circuit(n_qbts, var_params)
    circ.save_density_matrix()
    backend.set_options(basis_gates = noise_model.basis_gates,
        noise_model = noise_model)
    result = backend.run(circ).result()
    density_matrix = np.array(result.data(0)['density_matrix'])
    #density_matrix_2 = result.data(0)['density_matrix']

    return density_matrix

def get_HR_distance_noiseless(J, d_m, ops_l):
    ops_n = len(ops_l)
    #intialize covariance matrix, with all its entries being zeros.
    cov_mat = np.zeros((ops_n, ops_n), dtype=float)
    for i1 in range(ops_n):
        for i2 in range(i1, ops_n):
            O1_O2 = 1/2 * (expected_op1_op2(ops_l[i1], ops_l[i2], d_m) + expected_op1_op2(ops_l[i2], ops_l[i1], d_m) )
            #test_O2 = 1/2 * (expected_op1_op2_test(ops_l[i1], ops_l[i2], d_m2) + expected_op1_op2_test(ops_l[i2], ops_l[i1], d_m2) )
            #print(O1_O2, test_O2)
            O1 = expected_op(ops_l[i1], d_m)
            #test_O1 = expected_op_test(ops_l[i1], d_m2)
            #print(O1, test_O1)
            O2 = expected_op(ops_l[i2], d_m)
            cov_mat[i1, i2] = O1_O2 - O1*O2
            cov_mat[i2, i1] = cov_mat[i1, i2]
    # print("noiseless diag element: ", cov_mat[0, 1])
    # print("noiseless cov[1, 1]: ", cov_mat[1, 1])
    # print("noiseless cov[0, 0]: ", cov_mat[0, 0])
    val, vec = np.linalg.eigh(cov_mat)
    argsort = np.argsort(val)
    val, vec = val[argsort], vec[:, argsort]
    orig_H = np.array([1, J])
    orig_H = orig_H/np.linalg.norm(orig_H)
    HR_dist = distanceVecFromSubspace(orig_H, vec[:, :1])

    return HR_dist

def get_fidelity(wf, mat):
    fid = np.sqrt(np.matmul(np.conj(wf),np.matmul(mat, wf)))
    return fid.real

def Gnd_est_cheat(density_matrix, Ham):
    
    estimate = np.trace(np.matmul(density_matrix, Ham)).real
    
    return estimate

def main(args):
    global E_hist, HR_dist_hist_noiseless
    current_path = os.getcwd()
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    os.chdir(args.output_dir)
    if not os.path.isdir("mnts_dicts"):
        os.mkdir("mnts_dicts")


    backend = Aer.get_backend("aer_simulator_density_matrix")

    # Add errors to noise model
    if args.p1 + args.p2 <= 0.0:
        noise_model = "no_depolarization"
        print(noise_model)
    else:
        error_1 = noise.depolarizing_error(args.p1, 1)
        error_2 = noise.depolarizing_error(args.p2, 2)
        noise_model = noise.NoiseModel()
        noise_model.add_all_qubit_quantum_error(error_1, ['ry', 'h'])
        noise_model.add_all_qubit_quantum_error(error_2, ['cx'])

    if args.gpu < 10: 
        Ham = get_Hamiltonian(args.n_qbts, args.J)
        gst_E, gst_wf = get_GST_E_and_wf(Ham)
        print(f"This is actual ground state energy: {gst_E}")
    else:
        Ham = get_Hamiltonian_gpu(args.n_qbts, args.J)
        gst_E, gst_wf = get_GST_E_and_wf_gpu(Ham)
        print(f"This is actual ground state energy: {gst_E}")

    param_hist_f = open(os.path.join(current_path, args.init_param),'rb')
    param_hist = pickle.load(param_hist_f)
    param_hist_f.close()
    param_hist = param_hist[args.start_idx :args.end_idx]

    with open("angles_file.dat", "wb") as angles_file_f:
            pickle.dump(param_hist, angles_file_f)
    angles_file_f.close()

    for i, params in enumerate(param_hist):
        m_dict_path = None
        #m_dict_path = os.path.join("mnts_dicts",f"m_dict_{i}.npy")
        m_dict = {}
        density_mat = get_density_matrix(args.n_qbts, params, backend, noise_model)
        ops_l = []
        ops_l.append(get_Hx(args.n_qbts))
        ops_l.append(get_Hzz(args.n_qbts))
        HR_dist_noiseless = get_HR_distance_noiseless(args.J, density_mat, ops_l)
        print("This is HR distance: ", HR_dist_noiseless)
        HR_dist_hist_noiseless.append(HR_dist_noiseless)
        with open("HR_dist_hist.pkl", "wb") as HR_dist_hist_f:
            pickle.dump(HR_dist_hist_noiseless, HR_dist_hist_f)
        x_key, z_key = 'x'*args.n_qbts, 'z'*args.n_qbts

        E = Gnd_est_cheat(density_mat, Ham)
        #print("Energy", E)
        E_hist.append(E)
        with open("E_hist.pkl", "wb") as E_f:
            pickle.dump(E_hist, E_f)

        ###Fidelity
        if args.fid != -1:

            fidelity = 100*get_fidelity(gst_wf, density_mat)
            #print(fidelity)
            fid_hist.append(fidelity)
            with open("fid_hist.pkl", "wb") as fid_hist_f:
                pickle.dump(fid_hist, fid_hist_f)
    print("HR_Average_noiseless", np.average(HR_dist_hist_noiseless))
    

    plt.figure(figsize = (10,10), dpi = 300)
    plt.grid()
    fig, ax = plt.subplots()
    VQE_steps = np.array(list(range(len(E_hist))))
    ax.scatter(VQE_steps, E_hist, c = 'b', alpha = 0.8, marker = ".", label = "Energy")
    ax.set_xlabel('VQE Iterations')
    ax.set_ylabel("Energy")
    ax.set_ylim([round(gst_E, 3), np.max(E_hist)])
    ax.legend(bbox_to_anchor=(1.28, 1.30), fontsize = 10)
    title = "VQE 1-D "+ str(args.n_qbts) +" qubits TFIM" + "\n" + f"J: {args.J}, shots: {args.shots}" + '\n' + 'True Ground energy: ' + \
            str(round(gst_E, 3)) + '\n' + 'Estimated Ground Energy: '+ str(round(float(min(E_hist)), 3)) + '\n' + \
            f'depolarization noise, $p_1$:{args.p1}, $p_2$:{args.p2}'
    plt.title(title, fontdict = {'fontsize' : 15})
    ax2 = ax.twinx()
    ax2.scatter(VQE_steps, HR_dist_hist_noiseless, c = 'r', alpha = 0.8, marker=".", label = "HR distance")
    ax2.set_ylabel("HR distance")
    ax2.legend(bbox_to_anchor=(1.28, 1.22), fontsize = 10)
    plt.savefig(f"VQE_HR_{args.n_qbts}qbts_{args.shots}_plot.png", dpi = 300, bbox_inches='tight')
    os.chdir(current_path)
    #breakpoint()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "VQE for 1-D TFIM with non-periodic boundary condition")
    args = get_args(parser)
    main(args)