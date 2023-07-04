import numpy as np
import os

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

def get_Hamiltonian_no_periodic(N_qubits, J):
    """ get Hamiltonian for 1-D TFIM

    Args:
        N_qubits(int): number of spins in 1-D TFIM
        J: coupling strength between nearest neighbor

    Return:
        Hamiltonian that corresponds to 1-D TFIM.
    """

    sig_z = np.array([[1., 0.], [0., -1.]])
    Hx = get_Hx(N_qubits)
    Hz = 0
    for i in range(N_qubits - 1):
        temp = [np.eye(2)]*N_qubits
        temp[i] = sig_z
        temp[i+1] = sig_z
        tempSum = temp[0]
        for j in range(1, N_qubits):
            tempSum = np.kron(temp[j], tempSum)
        Hz += tempSum
    return Hx + J*Hz

def get_Hamiltonian_periodic(N_qubits, J):
    """ get Hamiltonian for 1-D TFIM

    Args:
        N_qubits(int): number of spins in 1-D TFIM
        J: coupling strength between nearest neighbor

    Return:
        Hamiltonian that corresponds to 1-D TFIM.
    """
    sig_z = np.array([[1., 0.], [0., -1.]])
    Hx = get_Hx(N_qubits)
    Hz = 0
    for i in range(N_qubits):
        temp = [np.eye(2)]*N_qubits
        temp[i] = sig_z
        temp[(i+1)%N_qubits] = sig_z
        print(i, (i+1)%N_qubits)
        tempSum = temp[0]
        for j in range(1, N_qubits):
            tempSum = np.kron(temp[j], tempSum)
        Hz += tempSum
    return Hx + J*Hz

def main():
    gst_E_dict = {}
    max_n_qbts = 11
    J = 1.0
    periodic = True
    J = float(J)
    for i in range(3, max_n_qbts):
        n_qbts = i+1
        if periodic:
            Ham = get_Hamiltonian_periodic(n_qbts, J)
        else:
            Ham = get_Hamiltonian_no_periodic(n_qbts, J)
        eig_vals, _ = np.linalg.eig(Ham)
        ground_energy = np.amin(eig_vals)
        gst_E_dict[n_qbts] = ground_energy
        if periodic:
            np.save(f"gst_E_dict_J_{str(J)}_periodic.npy", gst_E_dict)
        else:
            np.save(f"gst_E_dict_J_{str(J)}_no_periodic.npy", gst_E_dict)

if __name__ == '__main__':
    main()
