import numpy as np
from functools import reduce

def get_exp_cross(cross_m, indices):
    tot_val, tot_count = 0, 0
    for cross_mt, m_count in cross_m.items():
        cross_mt = get_num_mt(cross_mt)
        mul = 1
        for ind in indices:
            mul = mul * cross_mt[ind]
        tot_val += (mul * m_count)
        tot_count += m_count
    return tot_val/tot_count

def get_exp_X(x_m, expo):
    tot_val, tot_count = 0, 0
    for x_mt, m_count in x_m.items():
        x_mt = get_num_mt(x_mt)
        tot_val += ((sum(x_mt)**expo)*m_count)
        tot_count += m_count
    exp_val = tot_val/tot_count
    return exp_val

def get_exp_ZZ(z_m, expo):
    tot_val, tot_count = 0, 0
    for z_mt, m_count in z_m.items():
        z_mt = get_num_mt(z_mt)
        sum_zz = 0
        for i in range(len(z_mt) - 1):
            sum_zz += z_mt[i]*z_mt[(i+1)]
        tot_val += ((sum_zz**expo) * m_count)
        tot_count += m_count
    exp_val = tot_val / tot_count
    return exp_val

def get_num_mt(mt):
    mt_l = list(map(lambda x: 1 if x == '0' else -1, mt))
    mt_l.reverse()
    return mt_l

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

def get_fidelity(wf, mat):
    fid = np.sqrt(np.matmul(np.conj(wf),np.matmul(mat, wf)))
    return fid.real

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
