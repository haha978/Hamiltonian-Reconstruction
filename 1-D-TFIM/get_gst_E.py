import numpy as np
import os
import argparse

def get_args(parser):
    parser.add_argument('--max_n_qbts', type = int, default = 11, help = "maximum number of qubits to find the ground state energy of")
    parser.add_argument('--J', type = float, default = 0.5, help = "neighbor coupling strength (default J: 0.5)")
    parser.add_argument('--periodic', action = 'store_true', help = "True if periodic boundary condition False if non-periodic boundary condition")
    args = parser.parse_args()
    return args

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
        tempSum = temp[0]
        for j in range(1, N_qubits):
            tempSum = np.kron(temp[j], tempSum)
        Hz += tempSum
    return Hx + J*Hz

def main(args):
    gst_E_dict = {}
    max_n_qbts = args.max_n_qbts
    J = args.J
    periodic = args.periodic
    J = float(J)
    for i in range(10, max_n_qbts):
        n_qbts = i+1
        if periodic:
            Ham = get_Hamiltonian_periodic(n_qbts, J)
        else:
            Ham = get_Hamiltonian_no_periodic(n_qbts, J)
        eig_vals, _ = np.linalg.eig(Ham)
        breakpoint()
        ground_energy = np.amin(eig_vals)
        gst_E_dict[n_qbts] = ground_energy.real
        if periodic:
            np.save(f"gst_E_dict_J_{str(J)}_periodic.npy", gst_E_dict)
        else:
            np.save(f"gst_E_dict_J_{str(J)}_no_periodic.npy", gst_E_dict)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "VQE for 1-D TFIM ground state energy dictionary creation")
    args = get_args(parser)
    main(args)
