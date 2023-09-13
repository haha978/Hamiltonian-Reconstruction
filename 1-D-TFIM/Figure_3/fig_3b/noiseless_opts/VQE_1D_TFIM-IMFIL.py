import numpy as np
import time
import cirq
import qsimcirq
import matplotlib.pyplot as plt
from skquant.opt import minimize
from qiskit.algorithms.optimizers import SPSA
import os
import pickle
import argparse

parser = argparse.ArgumentParser(description = "Make ground state wavefunctions for VQE")
#parser.add_argument('--az_flag', type = str, help = "type 1 (ALA, YY, YY2)")
parser.add_argument('--budget', type = int, default = 20000, help = "Budget for IMFILL optimizers (default: 20000)")
parser.add_argument('--n_qbts', type = int, help = "number of qubits")
parser.add_argument('--n_layers', type = int, help = "Number layers of ansatz type from ansatz flag")
parser.add_argument('--gpu', type = int, default = -1, help = " -1 when not using GPU. Otherwise, value of this argument indicates device id number")
parser.add_argument('--J', type = float, help = "J value that indicates nearest neighbor connection")
parser.add_argument('--opt', type = float, default = -1, help = "default -1 for IMFIL, 1 for SPSA")
parser.add_argument('--angles', type = str, default = "NONE", help = "angles path")

args = parser.parse_args()

param_hist = []
E_hist = []

if args.gpu >= 0:
    import cupy as cp
    from utils_gpu import get_Hamiltonian, expected_op, get_operator_cache, get_noisy_energy
    print("I am using GPU")
    dev = cp.cuda.Device(args.gpu)
    dev.use()
else:
    from utils import get_Hamiltonian, expected_op, get_operator_cache, get_noisy_energy
# STORE = './'+str(args.n_qubits)+'_qubits_TFIM_p1_'+str(args.p_1)+'_p2_'+str(args.p_2) + 'X + '+ str(args.J) + 'ZZ'

def Hadamard(qubit_index, qbts) :
    return cirq.H(qbts[qubit_index])

def CNOT(control_bit_index, bit_index, qbts) :
    return cirq.CNOT(qbts[control_bit_index], qbts[bit_index])

def R(theta, qubit_index, axis, qbts) :
    if axis == 'x' :
        rotation = cirq.rx(theta)
    if axis == 'y' :
        rotation = cirq.ry(theta)
    if axis == 'z' :
        rotation = cirq.rz(theta)
    return rotation(qbts[qubit_index])

# ## Test ALA
def Q_Circuit(N_qubits, var_params):
    crct = cirq.Circuit()
    qbts = [cirq.GridQubit(i, j) for i in range(N_qubits) for j in range (1)]
    param_idx = 0
    crct.append([Hadamard(i, qbts) for i in range (N_qubits)])
    if N_qubits % 2 == 0:
        for layer in range(args.n_layers):
            if layer % 2 == 0:
                for i in range(0, N_qubits, 2):
                    crct.append(CNOT(i, (i+1), qbts))
                for i in range(N_qubits):
                    crct.append(R(var_params[param_idx], i, 'y', qbts))
                    param_idx += 1
            else:
                for i in range(1, N_qubits-1, 2):
                    crct.append(CNOT(i, (i+1), qbts))
                for i in range(1, N_qubits-1):
                    crct.append(R(var_params[param_idx], i, 'y', qbts))
                    param_idx += 1
    else:
        for layer in range(args.n_layers):
            if layer % 2 == 0:
                for i in range(0, N_qubits-1, 2):
                    crct.append(CNOT(i, (i+1), qbts))
                for i in range(N_qubits-1):
                    crct.append(R(var_params[param_idx], i, 'y', qbts))
                    param_idx += 1
            else:
                for i in range(1, N_qubits, 2):
                    crct.append(CNOT(i, (i+1), qbts))
                for i in range(1, N_qubits):
                    crct.append(R(var_params[param_idx], i, 'y', qbts))
                    param_idx += 1
    return qbts, crct


## YY
# def Q_Circuit(N_qubits, var_params):
#     global args
#     N_layers = args.n_layers
#     #Defining the circuit
#     crct = cirq.Circuit()
#     #Defining qubits
#     qbts = [cirq.GridQubit(i, j) for i in range(N_qubits) for j in range (1)]
#     param_idx = 0
#     crct.append([Hadamard(i, qbts) for i in range (N_qubits)])

#     for layer_idx in range(N_layers):
#         for i in range(N_qubits-1):
#             crct.append(CNOT(i, (i+1), qbts))
#             crct.append(R(var_params[param_idx], (i+1), 'y', qbts))
#             crct.append(CNOT(i, (i+1), qbts))
#             param_idx += 1
#         crct.append(R(var_params[param_idx], 0, 'y', qbts))
#         param_idx += 1
        
#     return qbts, crct

def get_fidelity(wf1, wf2):
    """
    Returns fidelity between wf1 and wf2 |<wf1|wf>2|

    Args:
        wf1 (np array): wavefunction 1 of interest
        wf2 (np array): wavefunction 2 of interest
    """
    fid = np.abs(np.sum(np.conj(wf1)*wf2))
    return fid.real

def get_energy(simulator, var_params, N_qubits, Ham, shots, J, X_i_cache, ZZ_i_cache):
    """
    Returns energy of the state, given a circuit and parameters of that circuit

    Args:
        simulator: quantum circuit simulator
        var_params (np array): parameters of quatum circuit
        N_qubits (int): number of qubits
        Ham (2d np array): Hamiltonian of interest
        shots (int): number of shots
        J (float): coupling strength between neighboring qubits
    """
    global param_hist, E_hist
    qubits, quantum_circuit = Q_Circuit(N_qubits, var_params)
    final_state = simulator.simulate(quantum_circuit)
    state_f = final_state.final_state_vector
    if args.gpu >= 0:
        #using gpu
        state_f = cp.asarray(state_f)
        if shots == -1:
            #using shots
            energy = expected_op(Ham, state_f).get().item()
        else:
            energy = get_noisy_energy(state_f, X_i_cache, ZZ_i_cache, shots, J).item()
    else:
        #using cpu
        if shots == -1:
            energy = expected_op(Ham, state_f).item()
        else:
            energy = get_noisy_energy(state_f, X_i_cache, ZZ_i_cache, shots, J).item()

    print("This is current energy in VQE iterations", energy)
    E_hist.append(energy)
    param_hist.append(var_params)
    #print(var_params)
    #print(param_hist)
    return energy

def evaluation(simulator, final_params, N_qubits, Ham, ground_wf):
    """
    Get fidelity, energy, and final state vector from final parameters

    Args:
        simulator: quantum circuit simulator.
        final_params (np array): parameters of quatum circuit.
        N_qubits (int): number of qubits.
        Ham (2d np array): Hamiltonian of interest.
        ground_wf (np array): numpy array that indicates ground wavefunction.
    """
    qubits, quantum_circuit = Q_Circuit(N_qubits, final_params)
    result = simulator.simulate(quantum_circuit)
    state_f = result.final_state_vector
    fidelity = get_fidelity(ground_wf, state_f)
    energy = np.matmul(np.matmul(np.conj(state_f), Ham), state_f).real
    print("This is energy", energy)
    print("This is fidelity", get_fidelity(ground_wf, state_f))
    return energy, fidelity, state_f

def main():
    global args, E_hist, param_hist
    N_layers = args.n_layers
    N_qubits = args.n_qbts
    simulator = qsimcirq.QSimSimulator({'t': 1})
    J = args.J
    num_shots = -1
    Hamiltonian = get_Hamiltonian(N_qubits, J)
    start_time = time.time()
    if num_shots == -1:
        X_i_cache, ZZ_i_cache = [], []
    else:
        X_i_cache, ZZ_i_cache = get_operator_cache(N_qubits)
    end_time = time.time()
    print("This is time took to create operator cache: ",(end_time - start_time))
    #Number of parameters to vary (for this example 2 rotation angles per qubit)
    Nparams = 0
    if args.n_qbts % 2 == 0:
        for i in range(args.n_layers):
            if i % 2 == 0:
                Nparams += args.n_qbts
            else:
                Nparams += (args.n_qbts - 2)
    else:
        for i in range(args.n_layers):
            Nparams += (args.n_qbts - 1)
   
    # Nparams = 0
    # for i in range(args.n_layers):
    #     Nparams += args.n_qbts

    print("N_qubits : ", N_qubits)
    print("Nparams : ", Nparams)
    #INITIALIZE ANGLES
    if args.angles != "NONE":
        with open(args.angles, "rb") as angle_f:
            angles = pickle.load(angle_f)
    else:
        angles = np.random.uniform(low = -np.pi, high = np.pi, size = Nparams)
    bounds = np.tile(np.array([-np.pi, np.pi]),(Nparams,1))
    if args.opt == -1:
        minimize(lambda params: get_energy(simulator, params, N_qubits, Hamiltonian, num_shots, J, X_i_cache, ZZ_i_cache), angles, bounds, args.budget, method='imfil')
    elif args.opt == 1:
        spsa = SPSA(maxiter=3000) #,learning_rate=0.03, perturbation=0.05)

        spsa.optimize(Nparams, lambda params: get_energy(simulator, params, N_qubits, Hamiltonian, num_shots, J, X_i_cache, ZZ_i_cache), initial_point = angles)
    else: 
        assert (args.opt == 1), "Incorrect Optimizer Command"
        
    #estimates are energy values during optimization
    estimates = np.array(E_hist)
    #history of parameters during optimization
    params = param_hist
    #final paramters that returns the minimum energy value
    final_params = params[np.argmin(estimates)]
    #Directory for storage --> needs to be different for when there is output noise compared to when there isn't an output noise
    if num_shots == -1:
        STORE = './'+str(args.n_qbts)+'_qubits_TFIM_' + 'X+'+ str(args.J) + 'ZZ'
    else:
        STORE = './'+str(args.n_qbts)+'_qubits_TFIM_'+ 'X+'+ str(args.J) + 'ZZ_'+ 'shots_' +str(num_shots)
    #create storage directory if there is no such directory
    if os.path.isdir(STORE):
        pass
    else:
        os.mkdir(STORE)
    #get exact solution use only NUMPY from here
    if args.gpu >= 0:
        Hamiltonian = cp.asnumpy(Hamiltonian)
    eigen_values, eigen_vectors = np.linalg.eig(Hamiltonian)
    Ground_energy = np.amin(eigen_values)
    index = np.argwhere(eigen_values == Ground_energy)[0][0]
    #Obtain ground state's wavefunction
    ground_wf = eigen_vectors[:, index]
    ground_wf = (1./np.linalg.norm(ground_wf))*ground_wf
    #get energy, fidelity, and final state from
    final_energy, final_fidelity, final_state = evaluation(simulator, final_params, N_qubits, Hamiltonian, ground_wf)
    #when printing numpy values, set precision to 3 decimal points
    np.set_printoptions(precision=3)
    print("J : ", J)
    print("True ground energy : ", round(Ground_energy.real, 6))
    print("VQE ground energy : ", round(final_energy, 6))
    print("Fidelity : ", round(abs(final_fidelity), 6)*100, "%")

    props = {"system": "1-D TFIM",
               "N_qubits":N_qubits,
               "J":J,
               "Nlayers":N_layers,
               "Nparams":Nparams,
               "budget":args.budget,
               "Ground_energy":round(Ground_energy.real, 6),
               "VQE Estimate":round(float(final_energy), 6),
               "fidelity":final_fidelity,
              }
    prefix = "GND_E__"+ str(round(final_energy, 4)) + "_"
    folder = STORE+"/" + str(N_qubits) + "qubits_" + str(Nparams) + "params"
    inc = 0
    init_folder = folder
    if os.path.isdir(folder):
    #if we use ansatz with sam number of qubits and parameters then go into this loop
        while os.path.isdir(folder) == True:
            folder = init_folder +'_' + str(inc)
            inc = inc + 1
        if init_folder != folder:
            os.mkdir(folder)
        else:
            pass
    else:
        os.mkdir(folder)
    prefix = folder + "/" + prefix
    pickle.dump(params, open(prefix +"_angles_file.dat", "w+b"))
    pickle.dump(final_params, open(prefix+"final_params.dat", "w+b"))
    pickle.dump(final_state, open(prefix+"final_state.dat", "w+b"))
    pickle.dump(ground_wf, open(prefix+"ground_wf.dat", "w+b"))
    pickle.dump(estimates, open(prefix+"estimates.dat", "w+b"))
    pickle.dump(props, open(prefix+"props.dat", "w+b"))
    # Convergence plot
    plt.figure(figsize = (10,10), dpi = 300)
    plt.grid()
    fig, ax = plt.subplots()
    VQE_steps = np.array(list(range(len(estimates))))
    title =  "VQE 1-D "+ str(args.n_qbts) +" qubits TFIM" + "\n" + f"J: {args.J}" + '\n' + 'True Ground energy: ' + \
            str(round(Ground_energy.real, 3)) + '\n' + 'Estimated Ground Energy: '+ str(round(float(final_energy), 3))
    plt.title(title, fontdict = {'fontsize' : 15})
    ax.set_xlabel('VQE Iterations')
    ax.set_ylabel("Energy")
    ax.scatter(VQE_steps, estimates, c= 'b', alpha = 0.8, marker = ".", label = "Energy")
    ax.legend(bbox_to_anchor=(1.28, 1.30), fontsize = 10)
    plt.tight_layout()
    plt.savefig(prefix+"plot.png", dpi=300)
    #plt.show()
    pickle.dump(props, open(prefix+"props.dat", "w+b"))
    with open(prefix+"circuit.txt", "w") as external_file:
        print(Q_Circuit(N_qubits,final_params)[1], file = external_file)
        external_file.close()


if __name__ == '__main__':
    main()