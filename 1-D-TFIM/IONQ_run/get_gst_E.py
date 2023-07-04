import numpy as np
import os

def get_Hamiltonian(N_qubits, J):
    """ get Hamiltonian for 1-D TFIM

    Args:
        N_qubits(int): number of spins in 1-D TFIM
        J: coupling strength between nearest neighbor

    Return:
        Hamiltonian that corresponds to 1-D TFIM.
    """
    sig_x = np.array([[0., 1.], [1., 0.]])
    sig_z = np.array([[1., 0.], [0., -1.]])
    Hx = 0
    for i in range(N_qubits):
        temp = [np.eye(2)]*N_qubits
        temp[i] = sig_x
        tempSum = temp[0]
        for j in range(1, N_qubits):
            tempSum = np.kron(temp[j], tempSum)
        Hx += tempSum
    Hz = 0
    for i in range(N_qubits - 1):
        temp = [np.eye(2)]*N_qubits
        temp[i] = sig_z
        temp[(i+1)%N_qubits] = sig_z
        tempSum = temp[0]
        for j in range(1, N_qubits):
            tempSum = np.kron(temp[j], tempSum)
        Hz += tempSum
    return Hx + J*Hz

def main():
    gst_E_dict = {}
    max_n_qbts = 11
    J = 1
    for i in range(3, max_n_qbts):
        n_qbts = i+1
        Ham = get_Hamiltonian(n_qbts, J)
        eig_vals, _ = np.linalg.eig(Ham)
        ground_energy = np.amin(eig_vals)
        gst_E_dict[n_qbts] = ground_energy
        np.save(f"gst_E_dict_J_{J}_no_periodic.npy", gst_E_dict)
    #print(np.load("gst_E_dict_J_0.5_no_periodic.npy", allow_pickle = True).item())

if __name__ == '__main__':
    main()
