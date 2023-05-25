import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import scipy
import math

def alter_type(strnum):
    if strnum == '1':
        return -1
    else:
        return 1

def expected_op(op, wf):
    """
    Returns expected value of operator op, given a wavefunction wf

    Args:
        op (2-D np array): matrix that corresponds to an operator
        wf (1-D np array): array that corresponds to wave vector
    """
    return np.vdot(np.matmul(op, wf), wf).real

def get_exp_X(X_vals, expo):
    sum_x_l = list(map(lambda x: np.sum(np.array(x)),X_vals))
    exp_x = reduce(lambda x, y: x+y**expo, sum_x_l, 0)/len(sum_x_l)
    return exp_x

def get_exp_ZZ(Z_vals, expo):
    def coupling(Z_val):
        sum_zz = 0
        for i in range(len(Z_val)-1):
            sum_zz += Z_val[i]*Z_val[(i+1)]
        return sum_zz
    Z_couplings = list(map(coupling, Z_vals))
    return reduce(lambda x,y: x+y**expo, Z_couplings, 0)/len(Z_couplings)

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

def X1_Z2_Z3(N_qubits):
    sig_x = np.array([[0., 1.], [1., 0.]])
    sig_z = np.array([[1., 0.], [0., -1.]])
    temp = [np.eye(2)]*N_qubits
    temp[0] = sig_x
    temp[1] = sig_z
    temp[2] = sig_z
    tempSum = temp[0]
    for j in range(1, N_qubits):
        tempSum = np.kron(tempSum, temp[j])
    return tempSum

def X2_Z3_Z4(N_qubits, state):
    sig_x = np.array([[0., 1.], [1., 0.]])
    sig_z = np.array([[1., 0.], [0., -1.]])
    temp = [np.eye(2)]*N_qubits
    temp[1] = sig_x
    temp[2] = sig_z
    temp[3] = sig_z
    tempSum = temp[0]
    for j in range(1, N_qubits):
        tempSum = np.kron(tempSum, temp[j])
    return tempSum

def get_Hz(N_qubits):
    sig_z = np.array([[1., 0.], [0., -1.]])
    Hz = 0
    temp = [sig_z]*N_qubits
    tempSum = temp[0]
    for j in range(1, N_qubits):
        tempSum = np.kron(tempSum, temp[j])
    return tempSum

def get_XZZZ(N_qubits):
    sig_z = np.array([[1., 0.], [0., -1.]])
    Hz = 0
    temp = [sig_z]*N_qubits
    temp[0] = np.array([[0,1], [1,0]])
    tempSum = temp[0]
    for j in range(1, N_qubits):
        tempSum = np.kron(tempSum, temp[j])
    return tempSum

def H_gate(N_qbts):
    return scipy.linalg.hadamard(16)

def apply_gate(state, N_qbts, ops):
    X_gate_i = 0
    for i in range(len(ops)):
        if ops[i] == 'x':
            X_gate_i = i
    ops_l = [np.eye(2)]*N_qbts
    H = (1/np.sqrt(2))*np.array([[1,1],[1,-1]])
    ops_l[X_gate_i] = H
    gate = ops_l[0]
    for i in range(N_qbts - 1):
        gate = np.kron(gate, ops_l[i+1])
    return np.matmul(gate, state)

def get_mnts(state, N_qbts, shots):
    prob = np.conj(state)*state
    prob = prob/np.sum(prob)
    mnts = np.random.choice(N_qbts**2, shots, p = prob)
    getbinary = lambda x: format(x, 'b').zfill(N_qbts)
    mnts = list(map(getbinary, mnts))
    mnts_l = []
    for mnt in mnts:
        mnts_l.append(list(map(lambda x: 1 if x=='0' else -1, mnt)))
    return mnts_l

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

def main():
    N_qbts = 4
    J = 0.5
    Ham = get_Hamiltonian(N_qbts, J)
    val, vec = np.linalg.eigh(Ham)
    argsort = np.argsort(val)
    val, vec = val[argsort], vec[:, argsort]
    #print("This is eigenvalue: ", val[0] )
    """
    write validation script
    """
    gst = np.random.randn(16)
    gst = gst/np.linalg.norm(gst)
    # APPLY GATE function is not functioning right.
    # state = apply_gate(gst, N_qbts,'xzzz')
    state = gst
    Hz = get_Hz(N_qbts)
    value1 = expected_op(Hz, state)
    mnts = get_mnts(state, N_qbts, shots= 10000)
    value2 = get_exp_cross(mnts, [0,1,2,3])
    print(f"expected: {value1}, actual: {value2}")

    XZZZ = get_XZZZ(N_qbts)
    value3 = expected_op(XZZZ, gst)
    state = apply_gate(gst, N_qbts, 'xzzz')
    mnts = get_mnts(state, N_qbts, shots = 10000)
    value4 = get_exp_cross(mnts, [0,1,2,3])
    print(f"expected: {value3}, actual: {value4}")

    XZZI = X1_Z2_Z3(N_qbts)
    value5 = expected_op(XZZI, gst)
    state = apply_gate(gst, N_qbts, 'xzzz')
    mnts = get_mnts(state, N_qbts, shots = 10000)
    value6 = get_exp_cross(mnts, [0,1,2])
    print(f"expected: {value5}, actual: {value6}")
    # ops_l = ['xzzz','zxzz','zzxz','zzzx']
    # state1 = apply_gate(gst, N_qbts, ops_l[0])
    # mnts1 = get_mnts(state1, N_qbts, shots = 100000)
    # result1 = get_exp_cross(mnts1, [0,1,2])
    # result1_1 = X1_Z2_Z3(N_qbts, state1)
    #
    # state2 = apply_gate(gst, N_qbts, ops_l[1])
    # mnts2 = get_mnts(state2, N_qbts, shots = 100000)
    # result2 = get_exp_cross(mnts2, [1,2,3])
    # result2_1 = X2_Z3_Z4(N_qbts, state2)
    # print("Result1: ", result1)
    # print("Result1-1: ", result1_1)
    # print("Result2: ", result2)
    # print("Result2-1: ", result2_1)
    # conj_gst = np.conj(gst)
    # prob = conj_gst*gst
    # prob = prob/np.sum(prob)
    # Z_vals_int = np.random.choice(N_qbts**2, 10000, p = prob)
    # getbinary = lambda x: format(x, 'b').zfill(N_qbts)
    # Z_vals = list(map(getbinary, Z_vals_int))
    # H = (1/np.sqrt(2))*np.array([[1,1],[1,-1]])



    # H_gate = H
    # I_gate = np.eye(2)
    # gate_l = [I_gate, H_gate, I_gate, I_gate]
    # gate = gate_l[0]
    # for i in range(N_qbts - 1):
    #     gate = np.kron(gate, gate_l[i+1])
    #
    # gst_H = np.matmul(gate, gst)
    # prob_X = np.conj(gst_H)*gst_H
    # prob_X = prob_X/np.sum(prob_X)
    # X_vals_int = np.random.choice(N_qbts**2, 10000, p = prob_X)
    # X_vals = list(map(getbinary, X_vals_int))
    # X_vals = list(map(lambda x: 1 if x=='0' else -1, X_vals))
    # E = get_exp_X(X_vals, 1)+J*get_exp_ZZ(Z_vals, 1)
    #
    #
    # print(get_HR_dist(X_vals, Z_vals, J, gst, HxHz))
    # var_H = get_exp_X(X_vals, 2) + (J**2) * get_exp_ZZ(Z_vals, 2) + 2*J*get_exp_XZZ(X_vals, Z_vals)  - E**2
    # print("This is variance of H: ", var_H)


if __name__ == '__main__':
    main()
