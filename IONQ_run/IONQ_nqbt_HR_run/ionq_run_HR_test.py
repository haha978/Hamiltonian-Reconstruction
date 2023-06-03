import sys
sys.path.insert(0, "../")
from IONQ_nqbt_HR_run.ionq_run_HR import Q_Circuit, get_params
from qiskit import QuantumCircuit, Aer
from qiskit import transpile
import numpy as np
from utils import distanceVecFromSubspace, get_exp_cross, get_exp_X, get_exp_ZZ
import os

def expected_op(op, wf):
    return np.vdot(wf, np.matmul(op, wf)).real

def expected_op1_op2(op1, op2, wf):

    return np.vdot(wf, np.matmul(op1, np.matmul(op2, wf))).real

def get_measurement(n_qbts, var_params, backend, h_l, hyperparam_dict, param_idx):
    circ = Q_Circuit(n_qbts, var_params, h_l, hyperparam_dict["n_layers"])
    circ.measure(list(range(n_qbts)), list(range(n_qbts)))
    circ = transpile(circ, backend)
    job = backend.run(circ, shots = hyperparam_dict["shots"])
    if hyperparam_dict["backend"] != "aer_simulator":
        job_id = job.id()
        job_monitor(job)
    result = job.result()
    measurement = dict(result.get_counts())
    return measurement

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
    cov_mat[1, 1] = get_exp_ZZ(z_m, 2) - exp_ZZ**2
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
    print("shot_noise diag element: ", cov_mat[0, 1])
    print("shot_noise cov[1, 1]: ", cov_mat[1, 1])
    print("shot_noise cov[0, 0]: ", cov_mat[0, 0])
    #print("diag element noisy: ", cov_mat[0, 1])
    val, vec = np.linalg.eigh(cov_mat)
    argsort = np.argsort(val)
    val, vec = val[argsort], vec[:, argsort]
    orig_H = np.array([1, hyperparam_dict["J"]])
    orig_H = orig_H/np.linalg.norm(orig_H)
    HR_dist = distanceVecFromSubspace(orig_H, vec[:, :1])
    return HR_dist

def get_Hx(N_qubits):
    sig_x = np.array([[0., 1.], [1., 0.]])
    Hx = 0
    for i in range(N_qubits):
        temp = [np.eye(2)]*N_qubits
        temp[i] = sig_x
        tempSum = temp[0]
        for j in range(1, N_qubits):
            tempSum = np.kron(tempSum, temp[j])
        Hx += tempSum
    return Hx

def get_Hzz(N_qubits):
    sig_z = np.array([[1., 0.], [0., -1.]])
    Hz = 0
    for i in range(N_qubits-1):
        temp = [np.eye(2)]*N_qubits
        temp[i] = sig_z
        temp[(i+1)] = sig_z
        tempSum = temp[0]
        for j in range(1, N_qubits):
            tempSum = np.kron(temp[j], tempSum)
        Hz += tempSum
    return Hz

def get_Hamiltonian(N_qubits, J):
    Hx = get_Hx(N_qubits)
    Hz = get_Hzz(N_qubits)
    return Hx + J*Hz

def get_statevector(n_qbts, var_params, n_layers, backend):
    circ = Q_Circuit(n_qbts, var_params, [], n_layers)
    circ.save_statevector()
    result = backend.run(circ).result()
    statevector = np.array(result.get_statevector(circ))
    return statevector


def get_HR_distance_noiseless(hyperparam_dict, wf, ops_l):
    ops_n = len(ops_l)
    #intialize covariance matrix, with all its entries being zeros.
    cov_mat = np.zeros((ops_n, ops_n), dtype=float)
    for i1 in range(ops_n):
        for i2 in range(i1, ops_n):
            O1_O2 = 1/2 * (expected_op1_op2(ops_l[i1], ops_l[i2], wf) + expected_op1_op2(ops_l[i2], ops_l[i1], wf) )
            O1 = expected_op(ops_l[i1], wf)
            O2 = expected_op(ops_l[i2], wf)
            cov_mat[i1, i2] = O1_O2 - O1*O2
            cov_mat[i2, i1] = cov_mat[i1, i2]
    print("noiseless diag element: ", cov_mat[0, 1])
    print("noiseless cov[1, 1]: ", cov_mat[1, 1])
    print("noiseless cov[0, 0]: ", cov_mat[0, 0])
    val, vec = np.linalg.eigh(cov_mat)
    argsort = np.argsort(val)
    val, vec = val[argsort], vec[:, argsort]
    orig_H = np.array([1, hyperparam_dict["J"]])
    orig_H = orig_H/np.linalg.norm(orig_H)
    HR_dist = distanceVecFromSubspace(orig_H, vec[:, :1])
    return HR_dist


def test1():
    #input_dir can be any output_dir of IONQ_nqbt_HR_run/ionq_run.py and IONQ_nqbt_HR_run/ionq_run_HR.py
    input_dir = "/root/research/HR/PAPER_FIGURES/1-D-TFIM/IONQ_run/IONQ_nqbt_HR_run/tr1/"
    params_dir_path = os.path.join(input_dir, "params_dir")
    param_idx = 200
    var_params = get_params(params_dir_path, param_idx)
    hyperparam_dict= np.load(os.path.join(input_dir, "hyperparam_dict.npy"), allow_pickle = True).item()
    n_qbts = hyperparam_dict["n_qbts"]
    hyperparam_dict["backend"] = "aer_simulator"
    hyperparam_dict["shots"] = 100000
    backend = Aer.get_backend(hyperparam_dict["backend"])
    statevector = get_statevector(n_qbts, var_params, hyperparam_dict["n_layers"], backend)
    ops_l = []
    ops_l.append(get_Hx(n_qbts))
    ops_l.append(get_Hzz(n_qbts))
    HR_dist = get_HR_distance_noiseless(hyperparam_dict, statevector, ops_l)
    #print("This is noiseless HR distance: ", HR_dist)
    for _ in range(10):
        HR_dist_noisy = get_HR_distance(hyperparam_dict, param_idx, params_dir_path, backend)
        #print("This is noisy HR distance: ", HR_dist_noisy)

def main():
    test1()

if __name__ == '__main__':
    main()
