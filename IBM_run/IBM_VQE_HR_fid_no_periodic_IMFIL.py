from qiskit import IBMQ, Aer, assemble, transpile
import qiskit
from qiskit import QuantumCircuit
import time
import numpy as np
from qiskit.algorithms.optimizers import IMFIL
import argparse
from functools import partial
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description = "VQE runs for 4 qubits 1-D TFIM using IMFIL")
parser.add_argument('--n_qbts', type = int, default = 4, help = "number of qubits (default: 4)")
parser.add_argument('--backend', type = str, help = "Backend: 'aer_simulator', 'ibmq_manila', 'ibmq_bogota', 'ibmq_santiago', 'ibm_oslo', 'ibmq_belem','ibmq_quito'")
parser.add_argument('--shots', type = int, default = 10000, help ="Number of shots (default is 10000)")
parser.add_argument('--max_iter', type = int, default = 300, help = "Maximum number of function evaluations for IMFIL optimizer (default = 300)")
parser.add_argument('--J', type = float, default = 0.5, help = "J value that indicates nearest neighbor connection (default = 0.5)")
args = parser.parse_args()

history = []
estimates = []
HR_dist_hist = []
fidelities = []

def Q_Circuit(N_qubits, var_params, backend_nm):
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

def get_fidelity(wf, mat):
    """
    Get fidelity between a pure wave function and density matrix

    Args:
        wf (1-D numpy array): np array coresponding to wavefunciton
        mat (2-D numpy array): np array corresponding to density matrix

    Returns:
        OP_i (np 2d array): Operator's matrix representation

    """
    fid = np.matmul(np.conj(wf),np.matmul(mat, wf))
    return fid.real

def str_l_to_num_l(mts):
    num_l = []
    def alter_type(strnum):
        if strnum == '1':
            return -1
        else:
            return 1
    for mt in mts:
        num_mt= list(map(alter_type, mt))
        num_l.append(num_mt)
    return num_l

def get_exp_val(ops_idx_l, x_mts, y_mts, z_mts):
    num_mts = len(x_mts)
    total_sum = 0
    for shot_i in range(num_mts):
        mul = 1
        for qbt, ops_idx in enumerate(ops_idx_l):
            if ops_idx == 1:
                mul *= x_mts[shot_i][qbt]
            elif ops_idx == 2:
                mul *= y_mts[shot_i][qbt]
            elif ops_idx == 3:
                mul *= z_mts[shot_i][qbt]
        total_sum += mul
    return total_sum/num_mts

def get_density_matrix(N_qubits, x_mts, y_mts, z_mts):
    den_mat = np.zeros((2**N_qubits,2**N_qubits), dtype = np.complex128)
    I = np.eye(2)
    X = np.array([[0,1],[1,0]])
    Y = np.array([[0, -1j],[1j,0]])
    Z = np.array([[1,0],[0,-1]])
    op_lst = [I,X,Y,Z]
    for i1, op1 in enumerate(op_lst):
        for i2, op2 in enumerate(op_lst):
            for i3, op3 in enumerate(op_lst):
                for i4, op4 in enumerate(op_lst):
                    ops_idx_l = [i1,i2,i3,i4]
                    exp_val_basis = get_exp_val(ops_idx_l, x_mts, y_mts, z_mts)
                    basis = np.kron(np.kron(np.kron(op1,op2),op3),op4)
                    den_mat += (1/2**N_qubits)*exp_val_basis*basis
    return den_mat

def get_measurement(N_qubits, var_params, backend, backend_nm, shots, operator):
    circ = Q_Circuit(N_qubits, var_params, backend_nm)
    if operator == 'z':
        circ.measure_all()
    elif operator == 'x':
        for i in range(N_qubits):
            circ.h(i)
        circ.measure_all()
    elif operator == 'y':
        for i in range(N_qubits):
            circ.sdg(i)
            circ.h(i)
        circ.measure_all()
    if backend_nm == "ibm_oslo":
        q_map = [0,1,3,5]
    elif backend_nm == "ibmq_quito":
        q_map = [0,1,3,4]
    elif backend_nm == "ibmq_belem":
        q_map = [0,1,3,4]
    else:
        q_map = [0,1,2,3]
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

def alter_type(strnum):
    if strnum == '1':
        return -1
    else:
        return 1

def get_exp_X(X_vals, expo):
    exp_x = 0
    for X_val in X_vals:
        sum_x = np.sum(np.array(list(map(alter_type, X_val))))
        exp_x += (sum_x**expo)
    exp_x = exp_x/len(X_vals)
    return exp_x

def get_exp_ZZ(Z_vals, expo):
    exp_ZZ = 0
    for Z_val in Z_vals:
        Z_val = list(map(alter_type, Z_val))
        sum_zz = 0
        #non periodic boundary condition
        for i in range(len(Z_val)-1):
            sum_zz += Z_val[i]*Z_val[(i+1)]
        exp_ZZ += (sum_zz**expo)
    exp_ZZ = exp_ZZ/len(Z_vals)
    return exp_ZZ

def get_exp_XZZ(X_vals, Z_vals):
    exp_XZZ = 0
    for i in range(len(X_vals)):
        sum_i = 0
        X_val = list(map(alter_type, X_vals[i]))
        Z_val = list(map(alter_type, Z_vals[i]))
        ZZ_val = []
        for i in range(len(Z_val)-1):
            ZZ_val.append(Z_val[i]*Z_val[i+1])
        for idx1 in range(len(X_val)):
            #exclude when pauli-x and pauli-z operator overlaps because they anticommute.
            for idx2 in range(len(ZZ_val)):
                if not ((idx1 == idx2) or (idx1-1 == idx2)):
                    sum_i += X_val[idx1]*ZZ_val[idx2]
        exp_XZZ += sum_i
    exp_XZZ = exp_XZZ/len(X_vals)
    return exp_XZZ

def distanceVecFromSubspace(w, A):
    """
    Get L2 norm of distance from w to subspace spanned by columns of A

    Args:
        w (numpy 1d vector): vector of interest
        A (numpy 2d matrix): columns of A

    Return:
        L2 norm of distance from w to subspace spanned by columns of A
    """
    Q, _ = np.linalg.qr(A)
    r = np.zeros(w.shape)
    #len(Q[0]) is number of eigenvectors
    for i in range(len(Q[0])):
        r += np.dot(w, Q[:,i])*Q[:,i]
    return np.linalg.norm(r-w)

def get_HR_dist(X_vals, Z_vals, J):
    """
    X_vals, Z_vals: list of strings, which consist of 4 numbers
    """
    cov_mat = np.zeros([2,2])
    expected_X = get_exp_X(X_vals, 1)
    expected_ZZ = get_exp_ZZ(Z_vals, 1)
    #Z^2 in measurment
    cov_mat[0,0] = get_exp_X(X_vals, 2) - expected_X**2
    cov_mat[0,1] = get_exp_XZZ(X_vals, Z_vals) - expected_X*expected_ZZ
    cov_mat[1,0] = cov_mat[0,1]
    cov_mat[1,1] = get_exp_ZZ(Z_vals, 2) - expected_ZZ**2
    print("\nThis is covariance matrix")
    print(cov_mat)
    val, vec = np.linalg.eigh(cov_mat)
    argsort = np.argsort(val)
    val, vec = val[argsort], vec[:, argsort]
    print("This is smallest eigen value of the covariance matrix: ", val[0])
    print("This is the eigenvector that has the smallest eigenvalue of covariance matrix: ",vec[:,:1])
    orig_H = np.array([1, J])
    orig_H = orig_H/np.linalg.norm(orig_H)
    print("This is original hamiltonian: ", orig_H)
    HR_dist = distanceVecFromSubspace(orig_H, vec[:, :1])
    return HR_dist

def get_energy(var_params, backend, backend_nm, N_qubits, J, shots, folder, gst):
    """
    Returns the energy of the state, given a circuit and parameters of that circuit

    Args:
        var_params (np array): parameters of quatum circuit
        backend(qiskit backend): backend running a quantum circuit
        backend_nm(str): backend name
        N_qubits (int): number of qubits
        az_flag (str): type of ansatz(ALA, YY2, YY)
        J (float): coupling strength between neighboring qubits
        shots (int): number of shots for each circuit
        folder (str): folder that you are storing number of qubits
    """
    global history, estimates, HR_distance_hist, fidelities
    #get nearest neighbor coupling
    Z_vals = get_measurement(N_qubits, var_params, backend, backend_nm, shots, 'z')
    X_vals = get_measurement(N_qubits, var_params, backend, backend_nm, shots, 'x')
    Y_vals = get_measurement(N_qubits, var_params, backend, backend_nm, shots, 'y')
    ZZ = get_exp_ZZ(Z_vals, 1)
    X = get_exp_X(X_vals, 1)
    E_total = X + J * ZZ
    print("This is total Energy: ", E_total)
    estimates.append(E_total)
    history.append(var_params)
    #SAVE INTERMEDIATE DATA FOR LATER
    with open(folder + '/' +'E_estimates.pkl', 'wb') as f:
        pickle.dump(estimates, f)
    with open(folder + '/' +'param_history.pkl', 'wb') as f:
        pickle.dump(history, f)
    #Obtain HR distance
    HR_dist = get_HR_dist(X_vals, Z_vals, J)
    print("This is HR_distance: ", HR_dist)
    HR_dist_hist.append(HR_dist)
    with open(folder + '/' +'HR_dist_hist.pkl', 'wb') as f:
        pickle.dump(HR_dist_hist, f)
    den_mat = get_density_matrix(N_qubits, str_l_to_num_l(X_vals), str_l_to_num_l(Y_vals), str_l_to_num_l(Z_vals))
    fid = get_fidelity(gst, den_mat)
    print("This is fidelity: ", fid)
    fidelities.append(fid)
    return E_total

"""
Obtain ground state energy of 1-D TFIM without periodic boundary condition
"""

def get_Hamiltonian(N_qubits, J):
    sig_x = np.array([[0., 1.], [1., 0.]])
    sig_z = np.array([[1., 0.], [0., -1.]])
    Hx = 0
    for i in range(N_qubits):
        temp = [np.eye(2)]*N_qubits
        temp[i] = sig_x
        tempSum = temp[0]
        for j in range(1, N_qubits):
            tempSum = np.kron(tempSum, temp[j])
        Hx += tempSum
    Hz = 0
    for i in range(N_qubits-1):
        temp = [np.eye(2)]*N_qubits
        temp[i] = sig_z
        temp[(i+1)] = sig_z
        tempSum = temp[0]
        for j in range(1, N_qubits):
            tempSum = np.kron(tempSum, temp[j])
        Hz += tempSum
    return Hx + J*Hz

def get_GST_E_and_wf(Ham):
    val, vec = np.linalg.eigh(Ham)
    argsort = np.argsort(val)
    val, vec = val[argsort], vec[:, argsort]
    return val[0], vec[:, 0]


def main():
    global args, history, HR_dist_hist, estimates, fidelities
    N_qubits = args.n_qbts
    J = args.J
    num_shots = args.shots
    Nparams = 4
    assert J == 0.5, 'need 0.5 as a coupling constant'
    #AVAILABLE Device
    #ibmq_manila, ibmq_bogota, ibmq_santiago, can call different backends by
    #eg) backend = provider.backend.ibmq_manila
    if args.backend =='aer_simulator':
        backend = Aer.get_backend('aer_simulator')
    else:
        IBMQ.save_account('9c02c0ee200f7c9e0de8ab0033a84d0b551bc86151f391920aee8acb1e63c3432e3a6084eedd54cd844d78794198eed88f798b055ce46aaaeefe5eae346f92b9', overwrite = True)
        provider = IBMQ.load_account()
        if args.backend == 'ibmq_manila':
            backend = provider.backend.ibmq_manila
        elif args.backend == 'ibmq_bogota':
            backend = provider.backend.ibmq_bogota
        elif args.backend == 'ibmq_santiago':
            backend = provider.backend.ibmq_santiago
        elif args.backend == 'ibm_oslo':
            backend = provider.backend.ibm_oslo
        elif args.backend == 'ibmq_belem':
            backend = provider.backend.ibmq_belem
        elif args.backend == 'ibmq_quito':
            backend = provider.backend.ibmq_quito
        else:
            raise ValueError("no backend device")
    #Directory for storage
    STORE = './'+str(args.n_qbts)+'_qubits_TFIM_'+ 'X+'+ str(args.J) + 'ZZ'
    if os.path.isdir(STORE):
        pass
    else:
        os.mkdir(STORE)
    folder = STORE + '/run' + '_shots_' +str(num_shots)
    inc = 0
    init_folder = folder
    if os.path.isdir(folder):
    #if we make a folder that exists, add a number next to it to make the folder different.
        while os.path.isdir(folder) == True:
            folder = init_folder +'_' + str(inc)
            inc = inc + 1
        if init_folder != folder:
            os.mkdir(folder)
        else:
            pass
    else:
        os.mkdir(folder)
    #saving meta data
    Ground_E, gst = get_GST_E_and_wf(get_Hamiltonian(args.n_qbts, J))
    print(Ground_E)
    props_interim = {"system": "1-D TFIM",
                    "N_qubits":N_qubits,
                    "J":J,
                    "backend": args.backend,
                    "optimizer": 'IMFIL',
                    "maxiter":args.max_iter,
                    "Ground_energy":round(Ground_E, 6),
                    "num_shots": num_shots
                    }
    pickle.dump(props_interim, open(folder+ "/" + "props_interim.dat", "w+b"))
    #params = np.random.uniform(low = -np.pi, high = np.pi, size = Nparams)
    params = np.array([-0.00222395,  1.97775134, -0.65761327, -1.1641015 ])
    bounds = np.tile(np.array([-np.pi, np.pi]),(Nparams,1))
    imfil = IMFIL(maxiter = args.max_iter)
    get_energy_func = partial(get_energy, backend = backend, backend_nm = args.backend, N_qubits = N_qubits,\
                                J = J, shots = args.shots, folder = folder, gst = gst)
    #print(get_energy_func(params))
    result = imfil.minimize(get_energy_func, x0 = params, bounds = bounds)
    #save parameter history and their corresponding energy
    params = np.array(history)
    estimates = np.array(estimates)
    HR_dist_hist = np.array(HR_dist_hist)
    #final parameters that return the minimum energy value
    #CREATE HR vS ENERGY PLOT
    plt.figure(figsize=(10, 10), dpi=300)
    plt.grid()
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    VQE_steps = np.array(list(range(len(estimates))))
    ax.scatter(VQE_steps, HR_dist_hist, c = 'r', alpha = 0.8, marker=".", s = 7, label = "HR distance")
    ax2.scatter(VQE_steps, estimates, c = 'b', alpha = 0.5, marker=".", s= 7, label = "Energy(eV)")
    ax.set_xlabel('VQE Iterations')
    ax.set_ylabel('HR distance')
    ax2.set_ylabel("Energy(eV)")
    ax.legend(bbox_to_anchor=(1.28, 1.30), fontsize = 10)
    ax2.legend(bbox_to_anchor=(1.28, 1.22), fontsize = 10)
    final_params = params[np.argmin(estimates)]
    final_energy = estimates[np.argmin(estimates)]
    print("J : ", J)
    print("True ground energy : ", str(round(Ground_E, 6)))
    print("VQE ground energy: ", str(round(final_energy, 6)))

    props = {"system": "1-D TFIM",
               "N_qubits":N_qubits,
               "J":J,
               "backend": args.backend,
               "optimizer": 'SPSA(1st order)',
               "maxiter":args.max_iter,
               "Ground_energy":round(Ground_E, 6),
               "VQE Estimate":round(float(final_energy), 6),
               "num_shots": num_shots
              }

    prefix = "GND_E__"+ str(round(float(final_energy), 4)) + "_"
    prefix = folder + "/" + prefix
    pickle.dump(props, open(prefix + "_props.dat", "w+b"))
    pickle.dump(params, open(prefix +"_angles_file.dat", "w+b"))
    pickle.dump(final_params, open(prefix+"final_params.dat", "w+b"))
    pickle.dump(estimates, open(prefix+"estimates.dat", "w+b"))
    pickle.dump(HR_dist_hist, open(prefix+"HR_dist_hist.dat", "w+b"))
    title = "VQE on 1-D TFIM" + "\n" + "J: " + str(J) + '\n' + 'True Ground energy: ' + \
                    str(round(Ground_E, 6)) + '\n' + 'Estimated Ground Energy: '+ str(round(float(final_energy), 6)) + '\n' + \
                    'backend: ' + args.backend
    plt.title(title, fontdict = {'fontsize' : 15})
    plt.savefig(prefix+"HR_E_plot.png", dpi=300, bbox_inches='tight')

    #create HR distnace vs fidelity plot
    plt.figure(figsize=(10, 10), dpi=300)
    plt.grid()
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    VQE_steps = np.array(list(range(len(estimates))))
    ax.scatter(VQE_steps, HR_dist_hist, c = 'r', alpha = 0.8, marker=".", s = 7, label = "HR distance")
    ax2.scatter(VQE_steps, fidelities, c = 'g', alpha = 0.8, marker=".", s= 7, label = "Fidelity")
    ax.set_xlabel('VQE Iterations')
    ax.set_ylabel('HR distance')
    ax2.set_ylabel("Fidelity")
    ax.legend(bbox_to_anchor=(1.28, 1.30), fontsize = 10)
    ax2.legend(bbox_to_anchor=(1.28, 1.22), fontsize = 10)
    title = "VQE on 1-D TFIM" + "\n" + "J: " + str(J) + '\n' + 'True Ground energy: ' + \
                    str(round(Ground_E, 6)) + '\n' + 'Estimated Ground Energy: '+ str(round(float(final_energy), 6)) + '\n' + \
                    'backend: ' + args.backend
    plt.title(title, fontdict = {'fontsize' : 15})
    plt.savefig(prefix+"HR_fid_plot.png", dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    main()
