import numpy as np
from functools import reduce
import os
import pickle
#import cupy as cp

def get_Hx(N_qubits):
    sig_x = np.array([[0., 1.], [1., 0.]])
    Hx = 0
    for i in range(N_qubits):
        temp = [np.eye(2)]*N_qubits
        temp[i] = sig_x
        tempSum = temp[0]
        for j in range(1, N_qubits):
            tempSum = np.kron(temp[j], tempSum)
        Hx += tempSum
    return Hx

def get_Hzz(N_qubits):
    sig_z = np.array([[1., 0.], [0., -1.]])
    Hzz = 0
    for i in range(N_qubits-1):
        temp = [np.eye(2)]*N_qubits
        temp[i] = sig_z
        temp[(i+1)] = sig_z
        tempSum = temp[0]
        for j in range(1, N_qubits):
            tempSum = np.kron(temp[j], tempSum)
        Hzz += tempSum
    return Hzz

def get_Hamiltonian(N_qubits, J):
    return get_Hx(N_qubits) + J*get_Hzz(N_qubits)

def get_exp_X(X_vals, expo):
    exp_x = 0
    for X_val in X_vals:
        sum_x = np.sum(X_val)
        exp_x += (sum_x**expo)
    exp_x = exp_x/len(X_vals)
    return exp_x

def get_exp_ZZ(Z_vals, expo):
    exp_ZZ = 0
    for Z_val in Z_vals:
        sum_zz = 0
        #non periodic boundary condition
        for i in range(len(Z_val)-1):
            sum_zz += Z_val[i]*Z_val[(i+1)]
        exp_ZZ += (sum_zz**expo)
    exp_ZZ = exp_ZZ/len(Z_vals)
    return exp_ZZ

def mem_to_mem_num(memory):
    mem_num = []
    for mem in memory:
        #mem_num.append(list(map(lambda x: 1 if x == '0' else -1, mem)))
        mem_num_temp = list(map(lambda x: 1 if x == '0' else -1, mem))
        mem_num_temp.reverse()
        mem_num.append(mem_num_temp)
    return mem_num

def get_Hamiltonian_gpu(N_qubits, J):
    """ get Hamiltonian for 1-D TFIM

    Args:
        N_qubits(int): number of spins in 1-D TFIM
        J: coupling strength between nearest neighbor

    Return:
        Hamiltonian that corresponds to 1-D TFIM.
    """
    sig_x = cp.array([[0., 1.], [1., 0.]])
    sig_z = cp.array([[1., 0.], [0., -1.]])
    Hx = 0
    for i in range(N_qubits):
        temp = [cp.eye(2)]*N_qubits
        temp[i] = sig_x
        tempSum = temp[0]
        for j in range(1, N_qubits):
            tempSum = cp.kron(tempSum, temp[j])
        Hx += tempSum
    Hz = 0
    for i in range(N_qubits-1):
        temp = [cp.eye(2)]*N_qubits
        temp[i] = sig_z
        temp[(i+1)%N_qubits] = sig_z
        tempSum = temp[0]
        for j in range(1, N_qubits):
            tempSum = cp.kron(tempSum, temp[j])
        Hz += tempSum
    return Hx + J*Hz

# def get_fidelity(wf, mat):
#     """
#     Get fidelity between a pure wave function and density matrix

#     Args:
#         wf (1-D numpy array): np array coresponding to wavefunciton
#         mat (2-D numpy array): np array corresponding to density matrix

#     Returns:
#         OP_i (np 2d array): Operator's matrix representation

#     """
#     fid = np.matmul(np.conj(wf),np.matmul(mat, wf))
#     return fid.real

def get_GST_E_and_wf(Ham):
    val, vec = np.linalg.eigh(Ham)
    argsort = np.argsort(val)
    val, vec = val[argsort], vec[:, argsort]
    return val[0], vec[:, 0]

def get_first_excited_E_and_wf(Ham):
    val, vec = np.linalg.eigh(Ham)
    argsort = np.argsort(val)
    val, vec = val[argsort], vec[:, argsort]
    return val[1], vec[:, 1]

def get_GST_E_and_wf_gpu(Ham):
    val, vec = cp.linalg.eigh(Ham)
    argsort = cp.argsort(val)
    val, vec = val[argsort], vec[:, argsort]
    return val[0], vec[:, 0]

def get_xz_key(n_qbts, idx):
    key = ""
    str_list = ['z']*n_qbts
    str_list[idx] = 'x'
    key = key.join(str_list)
    return key

def get_fidelity(wf, mat):

    fid = np.matmul(np.conj(wf),np.matmul(mat, wf))
    return fid.real


def get_exp_cross(mts, q_indices):
    """
    Given a list of measurements mts,
    obtain the average of multiple of mt[indices in q_indices],
    where mt is an element in mts
    """
    sum_all = 0
    for mt in mts:
        mul = 1
        for q_i in q_indices:
            mul = mul * mt[q_i]
        sum_all += mul
    return sum_all/len(mts)

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

def get_file(dir_name, substring):
    """
    Returns a file in a directory that contains the given substring

    Args:
        dir_name (str): directory name
        substring (str)

    Returns:
        pickled file
    """
    filenames = os.listdir(dir_name)
    there_is_file = False
    file_path = dir_name
    for filename in filenames:
        if substring in filename:
            there_is_file = True
            file_path = os.path.join(file_path, filename)

    if there_is_file == False:
        raise ValueError('The directory input has no appropriate files')
    file = open(file_path, "rb")
    data = pickle.load(file)
    file.close()
    return data