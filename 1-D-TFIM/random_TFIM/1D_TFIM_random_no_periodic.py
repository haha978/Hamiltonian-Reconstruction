import numpy as np
import pickle
import os
import random
import argparse
import time
from utils import normalize, diagonalize, get_fidelity, expected_op, get_operator_cache, get_noisy_energy

parser = argparse.ArgumentParser(description = "Make set of perturbed ground state wavefunctions for VQE")
parser.add_argument('--save_dir', type = str, default = ".", help= "directory to save created wavefunctions (default: '.')")
parser.add_argument('--n_qbts', type = int, help = "number of qubits")
parser.add_argument('--J', type = float, help = "J value that indicates nearest neighbor connection")
args = parser.parse_args()

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

def main():
    global args
    N_qubits = args.n_qbts
    J = args.J
    Hamiltonian = get_Hamiltonian(N_qubits, J)
    eigen_values, eigen_vectors = diagonalize(Hamiltonian)
    Gnd_E = np.amin(eigen_values)
    index = np.argwhere(eigen_values == Gnd_E)[0][0]
    #obtain ground state's wavefunction
    ground_wf = eigen_vectors[:, index]
    ground_wf = (1./np.linalg.norm(ground_wf))*ground_wf
    print("J : ", J)
    print("True ground energy : ", round(Gnd_E, 6))
    indices = [ind for ind in range(len(ground_wf))]
    #for loop is from PRAVEEN's code block (modified slightly)
    maxFid = 0
    minFid = 100
    wf_count = 0
    folder = args.save_dir
    sub_folder = str(N_qubits)+"qbts_1D-TFIM_J_"+str(J)
    PATH = folder + '/' + sub_folder
    if not os.path.isdir(PATH):
        os.mkdir(PATH)
    props = {"system": "1-D TFIM",
             "N_qubits": N_qubits,
             "J":J,
             "Ground Energy": round(Gnd_E, 6)}
    pickle.dump(props, open(PATH+"/props.dat", "w+b"))
    while wf_count < 100:
        #copy the ground state wavefunction to obtain perturbed ground state
        wf = np.copy(ground_wf)
        shuffled_indices = indices.copy()
        #obtain random sequence of indices
        random.shuffle(shuffled_indices)
        indicesRange = random.randrange(1, len(wf))
        for ind in range(0, indicesRange, 2):
            r = random.random()
            k = shuffled_indices[ind]
            wf[k] =  (1-r) * wf[k]
        wf = normalize(wf)
        fidelity = 100*get_fidelity(wf, ground_wf)**2
        if fidelity >= 80:
            print("This is fidelity: ", fidelity)
            maxFid = max(fidelity, maxFid)
            minFid = min(fidelity, minFid)
            pickle.dump(np.array(wf),open(PATH+'/'+'wf_'+str(wf_count)+'_.dat', "w+b"))
            wf_count += 1
        else:
            pass
    print("This is maximum fidelity obtained: ", maxFid)
    print("This is minimum fidelity obtained: ", minFid)

if __name__ == '__main__':
    main()
