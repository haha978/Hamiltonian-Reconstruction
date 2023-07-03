from qiskit import IBMQ, Aer, assemble, transpile
import qiskit
from qiskit import QuantumCircuit
import time
import numpy as np
from utils import get_exp_X, get_exp_ZZ, distanceVecFromSubspace, get_exp_cross, get_file, get_GST_E_and_wf, get_Hamiltonian
from qiskit.algorithms.optimizers import SPSA
import argparse
from functools import partial, reduce
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

def get_args(parser):
    parser.add_argument('--n_qbts',type = int, default = 4, help = "number of qubits(default: 4)")
    parser.add_argument('--load_dir', type = str, default = '.', help = "directory where the file exists")
    parser.add_argument('--load_param_l', type = str, default = './GND_E__-3.5067__angles_file.dat', help = "Loading list of parameters obtained from a VQE run")
    parser.add_argument('--J', type = float, default = 0.5, help = "coupling strength")
    parser.add_argument('--shots', type = int, default = 1000, help = "number of shots")
    parser.add_argument('--backend', type = str, default = 'aer_simulator', help = "Backend: 'aer_simulator', 'ibmq_manila', and 'ibm_oslo'")
    args = parser.parse_args()
    return args

HR_dist_hist = []

def Q_Circuit(N_qubits, var_params):
    #initialize actual circuit
    circ = QuantumCircuit(N_qubits)
    param_idx = 0
    for i in range(N_qubits):
        circ.ry(var_params[param_idx], i)
        param_idx = param_idx + 1
    circ.cx(0, 1)
    circ.cx(2, 3)
    circ.cx(1, 2)
    return circ

def get_measurement(N_qubits, var_params, backend, backend_nm, shots, ops):
    assert len(ops) == N_qubits, "number of qubits must match number of opss"
    circ = Q_Circuit(N_qubits, var_params)
    for i, op in enumerate(ops):
        if op == 'x':
            circ.h(i)
        elif op == 'y':
            circ.sdg(i)
            circ.h(i)
    circ.measure_all()
    if backend_nm == "ibm_oslo":
        q_map = [0,1,3,5]
    elif backend_nm == "ibmq_manila":
        q_map = [0,1,2,3]
    elif backend_nm == "aer_simulator":
        q_map = [0,1,2,3]
    else:
        RuntimeError("Unsupported backend device")
    circ = transpile(circ, backend, initial_layout = q_map)
    if backend_nm == 'aer_simulator':
        result = backend.run(circ, shots = shots, memory = True).result()
        memory = result.get_memory(circ)
    else:
        job = backend.run(circ, shots = shots, memory = True)
        print("Job created!")
        retrieved_job = backend.retrieve_job(job.job_id())
        start_time = time.time()
        print("Start counting time")
        result = retrieved_job.result()
        memory = result.get_memory(circ)
        end_time = time.time()
        print("Total time retrieving result tooked: ", end_time-start_time)
    return memory

def get_cov_mat(m_dict):
    cov_mat = np.zeros([2,2])
    exp_X, exp_ZZ = get_exp_X(m_dict['xxxx'], 1), get_exp_ZZ(m_dict['zzzz'], 1)
    cov_mat[0,0] = get_exp_X(m_dict['xxxx'], 2) - exp_X**2
    cov_mat[1,1] = get_exp_ZZ(m_dict['zzzz'], 2) - exp_ZZ**2
    #FINISH UP diagonal entries.
    cov_mat[0,1] = get_exp_cross(m_dict['xzzz'], [0,1,2]) + get_exp_cross(m_dict['xzzz'], [0,2,3]) + \
                    get_exp_cross(m_dict['zxzz'], [1,2,3]) + get_exp_cross(m_dict['zzxz'], [0,1,2]) + \
                    get_exp_cross(m_dict['zzzx'], [0,1,3]) + get_exp_cross(m_dict['zzzx'], [1,2,3]) - exp_X*exp_ZZ

    cov_mat[1,0] = cov_mat[0,1]
    return cov_mat

def main(args):
    global HR_dist_hist
    current_path = os.getcwd()
    J = args.J
    if not os.path.isdir(args.load_dir):
        ValueError("There is no load directory inputted")
    os.chdir(args.load_dir)
    E_file_l = [file for file in os.listdir(os.getcwd()) if "estimates.dat" in file]
    angles_l = [file for file in os.listdir(os.getcwd()) if "angles_file.dat" in file]
    assert len(E_file_l)==1 and len(angles_l)==1, "There needs to be one file that contains angles and one file that contains energy"
    estimates = get_file(os.getcwd(),E_file_l[0])
    angles = get_file(os.getcwd(),angles_l[0])
    N_qubits = args.n_qbts
    shots = args.shots
    title = "TFIM " + str(int(N_qubits)) + " spins"
    if args.backend =='aer_simulator':
        backend = Aer.get_backend('aer_simulator')
    else:
        IBMQ.save_account('9c02c0ee200f7c9e0de8ab0033a84d0b551bc86151f391920aee8acb1e63c3432e3a6084eedd54cd844d78794198eed88f798b055ce46aaaeefe5eae346f92b9', overwrite = True)
        provider = IBMQ.load_account()
        if args.backend == 'ibmq_manila':
            backend = provider.backend.ibmq_manila
        elif args.backend == 'ibm_oslo':
            backend = provider.backend.ibm_oslo
        else:
            raise ValueError("no backend device")
    if os.path.isfile("HR_dist_hist.pkl"):
        HR_dist_hist = get_file(os.getcwd(), "HR_dist_hist.pkl")
    if not os.path.isdir("mnts_dicts"):
        os.mkdir("mnts_dicts")
    for idx in range(len(HR_dist_hist),len(angles)):
        #breakpoint()
        params = angles[idx]
        m_dict_path = os.path.join("mnts_dicts",f"m_dict_{idx}.npy")
        if os.path.isfile(m_dict_path):
            m_dict = np.load(m_dict_path, allow_pickle = True).item()
        else:
            m_dict = {}
        ops_l = ['xxxx', 'zzzz', 'xzzz', 'zxzz', 'zzxz', 'zzzx']
        ops_l = [ops for ops in ops_l if ops not in m_dict.keys()]
        for ops in ops_l:
            mnts_str = get_measurement(N_qubits, params, backend, args.backend, shots, ops)
            mnts_num = []
            for mnt_str in mnts_str:
                mnts_num.append(list(map(lambda x: 1 if x=='0' else -1, mnt_str)))
            m_dict[ops] = mnts_num
            np.save(m_dict_path, m_dict)
        print(f"This is E = X + {J}ZZ: ",get_exp_X(m_dict['xxxx'], 1)+J*get_exp_ZZ(m_dict['zzzz'],1))
        #NOW get covariance matrix
        cov_mat =  get_cov_mat(m_dict)
        val, vec = np.linalg.eigh(cov_mat)
        argsort = np.argsort(val)
        val, vec = val[argsort], vec[:, argsort]
        print("This is smallest eigen value of the covariance matrix: ", val[0])
        #print("This is the eigenvector that has the smallest eigenvalue of covariance matrix: ",vec[:,:1])
        orig_H = np.array([1, J])
        orig_H = orig_H/np.linalg.norm(orig_H)
        HR_dist = distanceVecFromSubspace(orig_H, vec[:, :1])
        print("This is HR distance: ", HR_dist)
        HR_dist_hist.append(HR_dist)
        with open('HR_dist_hist.pkl', 'wb') as f:
            pickle.dump(HR_dist_hist, f)
    #NOW MAKE SOME PLOTS
    gst_E, gst_wf = get_GST_E_and_wf(get_Hamiltonian(args.n_qbts, J))
    plt.figure(figsize = (10,10), dpi = 300)
    plt.grid()
    fig, ax = plt.subplots()
    VQE_steps = np.array(list(range(len(estimates))))
    ax.scatter(VQE_steps, estimates, c = 'b', alpha = 0.8, marker = ".", label = "Energy")
    ax.set_xlabel('VQE Iterations')
    ax.set_ylabel("Energy")
    ax.legend(bbox_to_anchor=(1.28, 1.30), fontsize = 10)
    title = title = "VQE 1-D TFIM "+ str(args.n_qbts) +" spins" + "\n" + f"J: {J}, shots: {args.shots}" + '\n' + 'True Ground energy: ' + \
            str(round(gst_E, 3)) + '\n' + 'Estimated Ground Energy: '+ str(round(float(min(estimates)), 3)) + '\n'+ 'backend: ' + args.backend
    plt.title(title, fontdict = {'fontsize' : 15})
    ax2 = ax.twinx()
    ax2.scatter(VQE_steps, HR_dist_hist, c = 'r', alpha = 0.8, marker=".", label = "HR distance")
    ax2.set_ylabel("HR distance")
    ax2.legend(bbox_to_anchor=(1.28, 1.22), fontsize = 10)
    plt.savefig("VQE_HR_plot.png", dpi = 300, bbox_inches='tight')
    os.chdir(current_path)

if __name__== '__main__':
    parser = argparse.ArgumentParser(description = "HR distance measurement")
    args = get_args(parser)
    main(args)
