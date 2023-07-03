import os
import numpy as np
from IBM_HR_dist_fix import get_cov_mat, Q_Circuit
from IBM_HR_dist_old import get_measurement
from utils import get_file, get_Hx, get_Hzz
from qiskit import QuantumCircuit, Aer

def get_statevector(n_qbts, var_params, backend):
    circ = Q_Circuit(n_qbts, var_params)
    circ.save_statevector()
    result = backend.run(circ).result()
    statevector = np.array(result.get_statevector(circ))
    return statevector

def expected_op(op, wf):
    return np.vdot(wf, np.matmul(op, wf)).real

def expected_op1_op2(op1, op2, wf):
    return np.vdot(wf, np.matmul(op1, np.matmul(op2, wf))).real

def get_noiseless_cov_mat(ops_l, wf):
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
    return cov_mat

def test1():
    #INPUT_DIR should be directory that contians the output of script: IBM_VQE_no_periodic_IMFIL.py, IBM_HR_dist_old.py, and IBM_HR_dist_fix.py (in this order)
    INPUT_DIR ="/root/research/HR/PAPER_FIGURES/1-D-TFIM/IBM_run/4_qubits_TFIM_X+0.5ZZ/run_shots_10000"
    params_l = get_file(INPUT_DIR, "angles_file.dat")
    N_qbt = 4
    idx = 13
    params = params_l[idx]
    #get noiseless cov_mat
    backend = Aer.get_backend("aer_simulator")
    state_vector =  get_statevector(N_qbt, params, backend)
    ops_l = [get_Hx(N_qbt), get_Hzz(N_qbt)]
    cov_mat = get_noiseless_cov_mat(ops_l, state_vector)
    print("Noiseless covariance matrix")
    print(cov_mat)

    #get covariance matrix with shot noise
    new_mnts_dicts_path = os.path.join(INPUT_DIR, "new_mnts_dicts", f"m_dict_{idx}.npy")
    m_dict = np.load(new_mnts_dicts_path, allow_pickle = True).item()
    print("Covariance matrix with shot noise")
    print(get_cov_mat(m_dict))

def test2():
    #INPUT_DIR should be directory that contians the output of script: IBM_VQE_no_periodic_IMFIL.py, IBM_HR_dist_old.py, and IBM_HR_dist_fix.py (in this order)
    INPUT_DIR ="/root/research/HR/PAPER_FIGURES/1-D-TFIM/IBM_run/4_qubits_TFIM_X+0.5ZZ/run_shots_10000"
    params_l = get_file(INPUT_DIR, "angles_file.dat")
    N_qbt = 4
    idx = 100
    shots = 100000
    params = params_l[idx]
    #get noiseless cov_mat
    backend = Aer.get_backend("aer_simulator")
    state_vector =  get_statevector(N_qbt, params, backend)
    ops_l = [get_Hx(N_qbt), get_Hzz(N_qbt)]
    cov_mat = get_noiseless_cov_mat(ops_l, state_vector)
    print("Noiseless cov_mat[0, 1]")
    print(cov_mat[0, 1])

    #get covariance matrix with shot noise
    m_dict = {}
    ops_l = ['xxxx', 'zzzz', 'xzzz', 'zxzz', 'zzxz', 'zzzx']
    for ops in ops_l:
        mnts_str = get_measurement(N_qbt, params, backend, "aer_simulator", shots, ops)
        mnts_num = []
        for mnt_str in mnts_str:
            mnt_num = list(map(lambda x: 1 if x=='0' else -1, mnt_str))
            mnt_num.reverse()
            mnts_num.append(mnt_num)
        m_dict[ops] = mnts_num
    print("Covariance matrix [0, 1] with shot noise")
    print(get_cov_mat(m_dict)[0, 1])

def main():
    test1()
    test2()

if __name__ == '__main__':
    main()
