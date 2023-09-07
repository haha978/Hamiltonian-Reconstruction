import numpy as np

def Si(ops, N_qbts):
    """ Single Spin Operator

    Args:
        ops (str): can be 'x','y','z'
        N_qbts (int): number of qubits

    Returns:
        OP_i (np 2d array): Operator's matrix representation
    """
    sig_i = np.eye(2)
    if ops == 'x':
        sig_i = np.array([[0., 1.], [1., 0.]])
    elif ops == 'y':
        sig_i = np.array([[0., -1j], [1j, 0.]])
    elif ops == 'z':
        sig_i = np.array([[1., 0.], [0., -1.]])
    else:
        raise ValueError("ops value invalid")
    OP_i = 0
    for i in range(N_qbts):
        temp = [np.eye(2)]*N_qbts
        temp[i] = sig_i
        tempSum = temp[0]
        for j in range(1, N_qbts):
            tempSum = np.kron(tempSum, temp[j])
        OP_i += tempSum
    return OP_i

def SiSi(ops, N_qbts):
    """ Neighboring Spin Operator

    Args:
        ops (str): can be 'xx','yy','zz'
        N_qbts (int): number of qubits

    Returns:
        OP_i (np 2d array): Operator's matrix representation
    """
    sig_i = np.eye(2)
    if ops == 'xx':
        sig_i = np.array([[0., 1.], [1., 0.]])
    elif ops == 'yy':
        sig_i = np.array([[0., -1j], [1j, 0.]])
    elif ops == 'zz':
        sig_i = np.array([[1., 0.], [0., -1.]])
    else:
        raise ValueError("ops value invalid")
    OP_i = 0
    for i in range(N_qbts):
        temp = [np.eye(2)]*N_qbts
        temp[i] = sig_i
        temp[(i+1)%N_qbts] = sig_i
        tempSum = temp[0]
        for j in range(1, N_qbts):
            tempSum = np.kron(tempSum, temp[j])
        OP_i += tempSum
    return OP_i

def SiSi_NN(ops, N_qbts):
    """ Next Neighboring Spin Operator

    Args:
        ops (str): can be 'x_x','y_y','z_z'
        N_qbts (int): number of qubits

    Returns:
        OP_i (np 2d array): Operator's matrix representation
    """
    sig_i = np.eye(2)
    if ops == 'x_x':
        sig_i = np.array([[0., 1.], [1., 0.]])
    elif ops == 'y_y':
        sig_i = np.array([[0., -1j], [1j, 0.]])
    elif ops == 'z_z':
        sig_i = np.array([[1., 0.], [0., -1.]])
    else:
        raise ValueError("ops value invalid")
    #OP_i is will be returned as an array
    OP_i = 0
    if N_qbts >= 5:
        for i in range(N_qbts):
            temp = [np.eye(2)]*N_qbts
            temp[i] = sig_i
            temp[(i+2)%N_qbts] = sig_i
            tempSum = temp[0]
            for j in range(1, N_qbts):
                tempSum = np.kron(tempSum, temp[j])
            OP_i += tempSum
    elif N_qbts == 4:
        # need to consider 4 spins case seperately
        for i in range(int(N_qbts/2)):
            temp = [np.eye(2)]*N_qbts
            temp[i] = sig_i
            temp[(i+2)%N_qbts] = sig_i
            tempSum = temp[0]
            for j in range(1, N_qbts):
                tempSum = np.kron(tempSum, temp[j])
            OP_i += tempSum
    else:
        raise ValueError("Need more than 4 qubits to consider next-nearest neighbor")
    return OP_i

def getExactGroundWf(N_qubits, J):
    eigen_values, eigen_vecs = np.linalg.eig(get_Hamiltonian(N_qubits, J))
    return eigen_vecs[:, np.argmin(eigen_values)]

def get_fidelity(wf1, wf2):
    """
    Get fidelity between a pure wave function and density matrix

    Args:
        wf1 (1-D numpy array): np array coresponding to wavefunciton 1
        wf2 (2-D numpy array): np array coresponding to wavefunciton 2

    Returns:
        OP_i (np 2d array): Operator's matrix representation

    """
    fid = np.matmul(np.conj(wf1), wf2)
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

def diagonalize(mat):
    """
    diagonalize matrix
    return sorted eigenvalues and eigen vectors
    """
    val, vec = np.linalg.eigh(mat)
    argsort = np.argsort(val)
    return val[argsort], vec[:, argsort]

def expected_op(op, wf):
    """
    Returns expected value of operator op, given a wavefunction wf

    Args:
        op (2-D np array): matrix that corresponds to an operator
        wf (1-D np array): array that corresponds to wave vector
    """
    return np.vdot(np.matmul(op, wf), wf).real

def noisy_partial_energy(E_i_exact, shots):
    """
    From exact energy obtain energy with read-out noise, using number of shots

    Args:
        shots (int): number of shots
        E_i (exact): Exact energy of some partial Hamiltonian
    """
    rng = np.random.default_rng()
    p_i_hat = (E_i_exact + 1)/2.0
    if p_i_hat > 1:
        print("probability cannot be higher than one!")
        p_i_hat = 1
    elif p_i_hat < 0:
        print("probability cannot be lower than zero!")
        p_i_hat = 0
    result = rng.binomial(1, p_i_hat, shots)
    p_i = np.sum(result)/shots
    E_i_noise = 2*p_i - 1
    return E_i_noise

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
        temp[(i+1)%N_qubits] = sig_z
        tempSum = temp[0]
        for j in range(1, N_qubits):
            tempSum = np.kron(tempSum, temp[j])
        Hz += tempSum
    return Hx + J*Hz

def get_operator_cache(N_qubits):
    """
    Get operators cache to speed up obtaining the energy

    Args:
        N_qubits (int): number of qubits
    """
    sig_x = np.array([[0., 1.], [1., 0.]])
    sig_z = np.array([[1., 0.], [0., -1.]])
    X_i_cache = []
    ZZ_i_cache = []
    for i in range(N_qubits):
        temp = [np.eye(2)]*N_qubits
        temp[i] = sig_x
        X_i = temp[0]
        for j in range(1, N_qubits):
            X_i = np.kron(X_i, temp[j])
        X_i_cache.append(X_i)
    for i in range(N_qubits-1):
        temp = [np.eye(2)]*N_qubits
        temp[i] = sig_z
        temp[(i+1)%N_qubits] = sig_z
        ZZ_i = temp[0]
        for j in range(1, N_qubits):
            ZZ_i = np.kron(ZZ_i, temp[j])
        ZZ_i_cache.append(ZZ_i)
    return X_i_cache, ZZ_i_cache

def cov_mat(ops_l, wf):
    """
    ops_l: list of operators
    wf: wave function used to calculate expected value of operators in ops_l
    """
    ops_n = len(ops_l)
    #intialize covariance matrix, with all its entries being zeros.
    Q = np.zeros((ops_n, ops_n), dtype=float)
    for i1 in range(ops_n):
        for i2 in range(i1, ops_n):
            O1_O2 = expected_op(np.matmul(ops_l[i1], ops_l[i2]), wf)
            O1 = expected_op(ops_l[i1], wf)
            O2 = expected_op(ops_l[i2], wf)
            Q[i1, i2] = O1_O2 - O1*O2
            Q[i2, i1] = Q[i1, i2]
    return Q


def get_noisy_energy(state_f, X_i_cache, ZZ_i_cache, shots, J):
    """
    Returns energy of state_f, with given number of shots.

    Args:
        state_f (np array): state_f from circuit
        N_qubits (int): number of qubits
        shots (int): number of shots
        J (float): coupling strength
    """
    E_noise = 0
    for X_i in X_i_cache:
        X_i_exact = expected_op(X_i, state_f)
        X_i_noise = noisy_partial_energy(X_i_exact, shots)
        E_noise += X_i_noise
    for ZZ_i in ZZ_i_cache:
        ZZ_i_exact = expected_op(ZZ_i, state_f)
        ZZ_i_noise = noisy_partial_energy(ZZ_i_exact, shots)
        E_noise += J*ZZ_i_noise
    return E_noise
