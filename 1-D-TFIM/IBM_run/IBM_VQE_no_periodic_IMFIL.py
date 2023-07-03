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
from utils import get_exp_X, get_exp_ZZ, str_l_to_num_l

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

def alter_type(strnum):
    if strnum == '1':
        return -1
    else:
        return 1

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
    memory = str_l_to_num_l(memory)
    return memory

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
    ZZ = get_exp_ZZ(Z_vals, 1)
    X = get_exp_X(X_vals, 1)
    E_total = X + J * ZZ
    print("This is total Energy: ", E_total)
    estimates.append(E_total)
    history.append(var_params)
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
            tempSum = np.kron(temp[j], tempSum)
        Hx += tempSum
    Hz = 0
    for i in range(N_qubits-1):
        temp = [np.eye(2)]*N_qubits
        temp[i] = sig_z
        temp[(i+1)] = sig_z
        tempSum = temp[0]
        for j in range(1, N_qubits):
            tempSum = np.kron(temp[j], tempSum)
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
    #final parameters that return the minimum energy value
    #CREATE HR vS ENERGY PLOT
    plt.figure(figsize=(10, 10), dpi=300)
    plt.grid()
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    VQE_steps = np.array(list(range(len(estimates))))
    ax.scatter(VQE_steps, estimates, c = 'b', alpha = 0.5, marker=".", s= 15, label = "Energy(eV)")
    ax.set_xlabel('VQE Iterations')
    ax.set_ylabel("Energy(eV)")
    ax.legend(bbox_to_anchor=(1.28, 1.30), fontsize = 10)
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
    title = "VQE on 1-D TFIM" + "\n" + "J: " + str(J) + '\n' + 'True Ground energy: ' + \
                    str(round(Ground_E, 3)) + '\n' + 'Estimated Ground Energy: '+ str(round(float(final_energy), 6)) + '\n' + \
                    'backend: ' + args.backend
    plt.title(title, fontdict = {'fontsize' : 15})
    plt.savefig(prefix+"E_plot.png", dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    main()
