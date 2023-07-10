import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import pickle
import os
import sys
import random
import argparse

parser = argparse.ArgumentParser(description = "Get Hamiltonian Reconstruction Distance")
parser.add_argument('--n_qbts',type = int, help = "number of qubits")
parser.add_argument('--load_dir', type = str, help = "loading directory with random wave function of interest")
parser.add_argument('--num_eig', type = int, help = "number of eigen vectors during Hamiltonian reconstruction")
parser.add_argument('--ops', type = str, help = "operators used during reconstruction. Operators separated by spaces. Possible operators are: \
                                                        'x', 'y', 'z', 'xx', 'yy', 'zz', 'x_x', 'y_y', 'z_z'")
parser.add_argument('--gpu', type = int, default = -1, help = " -1 when not using GPU. Otherwise, value of this argument indicates device id number")
parser.add_argument('--save_dir', type = str, help = "save directory")

args = parser.parse_args()

LOAD_DIR = args.load_dir

if args.gpu == -1:
    from utils import Si, SiSi, SiSi_NN, get_Hamiltonian, getExactGroundWf, get_fidelity, distanceVecFromSubspace, diagonalize, expected_op, cov_mat
else:
    import cupy as cp
    from utils_gpu import Si, SiSi, SiSi_NN, get_Hamiltonian, getExactGroundWf, get_fidelity, distanceVecFromSubspace, diagonalize, expected_op, cov_mat
    dev = cp.cuda.Device(args.gpu)
    dev.use()

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
    return pickle.load(open(file_path, "rb+"))

def match_op_name(op_name):
    """
    Returns operator that corresponds to op_name

    Args:
        op_name (str): string that corresponds to an operator (e.g 'x', 'zz')

    Returns:
        Operator(2-D np array)
    """
    if op_name == 'x' or op_name =='y' or op_name =='z':
        return Si(op_name, args.n_qbts)
    elif op_name == 'xx' or op_name =='yy' or op_name =='zz':
        return SiSi(op_name, args.n_qbts)
    elif op_name == 'x_x' or op_name == 'y_y' or op_name == 'z_z':
        return SiSi_NN(op_name, args.n_qbts)
    else:
        raise ValueError('wrong operator notations')

def main():
    title = "1-D TFIM " +str(int(args.n_qbts)) + " spins"
    #entries contain all the directories inside LOAD_DIR
    entries = list(os.scandir(LOAD_DIR))
    entries = [os.path.join(LOAD_DIR,entry.name) for entry in entries if (entry.name != "props.dat")]
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)
    # get list of operators' names
    op_name_l = list(args.ops.split(" "))
    #reading the first VQE entry to get metadata of VQE runs
    VQE_init_props = get_file(LOAD_DIR, "props.dat")
    J = VQE_init_props["J"]
    assert args.n_qbts == VQE_init_props["N_qubits"], "Number of qubits inputted has error!"
    print("This is coupling strength: ", J)
    true_gnd_wf = getExactGroundWf(args.n_qbts, J)
    true_ham_vec = []
    for op_name in op_name_l:
        #This is specific for 1-D TFIM
        if op_name == 'x':
            true_ham_vec.append(1)
        elif op_name == 'zz':
            true_ham_vec.append(J)
        else:
            true_ham_vec.append(0)
    # NORMALIZE TRUE HAMILTONIAN
    true_ham_vec = np.array(true_ham_vec)
    norm = np.linalg.norm(true_ham_vec)
    true_ham_vec = true_ham_vec/norm
    #get np.array of operators in matrix form
    if args.gpu == -1:
        operators = np.array(list(map(match_op_name, op_name_l)))
    else:
        operators = cp.array(list(map(match_op_name, op_name_l)))
        true_ham_vec = cp.asarray(true_ham_vec)
    dists, fidelities, energies = [], [], []
    GST_E = 0
    for idx, entry in enumerate(entries):
        state_f = pickle.load(open(entry, "rb+"))
        if args.gpu != -1:
            state_f = cp.asarray(state_f)
        #FROM PROPERTY FILE WE CHECK NUMBER OF QUBITS
        N = int(np.log2(len(state_f)))
        assert N == args.n_qbts, "number of qubits is wrong"
        GST_E = VQE_init_props["Ground Energy"]
        Ham = get_Hamiltonian(N, J)
        hr_variances, hr_eig_vecs,  = diagonalize(cov_mat(operators, state_f))
        #sys.stdout = default_stdout
        #SUBSPACE IS 6 DIMENSIONAL
        print("These are varainces of Q matrix", hr_variances)
        dist = distanceVecFromSubspace(true_ham_vec, hr_eig_vecs[:, :args.num_eig])
        fidelity = 100*get_fidelity(true_gnd_wf, state_f)**2
        energy = expected_op(Ham, state_f).real
        print("This is HR distance: ", dist)
        print("This is fidelity: ", fidelity)
        print("This is energy: ", energy)
        energies.append(energy)
        if args.gpu == -1:
            dists.append(dist)
            fidelities.append(fidelity)
        else:
            dists.append(dist.get())
            fidelities.append(fidelity.get())
    #NOW START PLOTTING
    colors = [[random.uniform(0, 1),random.uniform(0, 1),col] for col in np.linspace(0, 1, len(entries))]
    plt.figure(figsize=(10, 10), dpi=300)
    plt.grid()
    fig, ax = plt.subplots()
    for i, dist in enumerate(dists):
        fidelity = fidelities[i]
        ax.scatter(fidelity, dist, c= [colors[i]], s=15**2, marker=".")
    hr_variances, hr_eig_vecs,  = diagonalize(cov_mat(operators, true_gnd_wf))
    gst_dist = distanceVecFromSubspace(true_ham_vec, hr_eig_vecs[:, :args.num_eig])
    if args.gpu >=0:
        gst_dist = gst_dist.get()
        FID = 100*get_fidelity(true_gnd_wf, true_gnd_wf).get()**2
    else:
        FID = 100*get_fidelity(true_gnd_wf, true_gnd_wf)**2
    ax.scatter(FID, gst_dist, s=15**2, marker="*", color = 'red', label = "ground state")
    last_path = os.path.normpath(LOAD_DIR)
    load_dir = os.path.basename(last_path)

    #get noise kind and value
    plt.title(title +'\n' + 'number of eigenvectors: '+str(args.num_eig) + \
                    '\n' + 'operators: '+ args.ops + '\n' + 'J: '+str(VQE_init_props["J"]) +'\n' + 'Ground Energy = ' + str(round(GST_E, 4)), fontsize = 25)
    ax.legend(loc='lower left', fontsize = 15)
    fig.autofmt_xdate()
    plt.xlabel('Fidelity (%)', fontsize = 30)
    plt.xticks(fontsize=25)
    plt.ylabel('HR distance', fontsize = 30)
    plt.yticks(fontsize=25)
    plt.savefig(args.save_dir+'/'+ load_dir+"_fid_"+args.ops+"_"+ str(args.num_eig)+"_.png", dpi = 300, bbox_inches='tight')

    plt.figure(figsize=(10, 10), dpi=300)
    plt.grid()
    fig, ax = plt.subplots()
    for i, dist in enumerate(dists):
        energy = energies[i]
        if args.gpu >= 0:
            energy = energy.get()
        ax.scatter(energy, dist, c= [colors[i]], s=15**2, marker=".")
    ax.scatter(GST_E, gst_dist, s=15**2, marker="*", color ='red', label = "ground state")
    plt.title(title +'\n'+ 'number of eigenvectors: '+str(args.num_eig) + \
                    '\n' + 'operators: '+ args.ops + '\n' + 'J: '+str(VQE_init_props["J"]) +'\n' + 'Ground Energy = ' + str(round(GST_E, 4)), fontsize = 25)
    ax.legend(loc='lower right', fontsize = 15)
    plt.xlabel('Energy[eV]', fontsize = 30)
    fig.autofmt_xdate()
    plt.xticks(fontsize=25)
    plt.ylabel('HR distance', fontsize = 30)
    plt.yticks(fontsize=25)
    plt.savefig(args.save_dir+'/'+load_dir+"_E_"+args.ops+"_"+ str(args.num_eig) +"_.png", dpi = 300, bbox_inches='tight')

if __name__ == '__main__':
    main()
