"""
Programmer: Vincent Lostanlen
Purpose: A first pass implementation of Haar scattering
The scheme is such that for an NxN matrix, there are
NxN coefficients
"""
import numpy as np
import matplotlib.pyplot as plt

def get_haarscattering(Xmat):
    N = Xmat.shape[0]
    assert(N&(N-1) == 0)
    J = int(np.log(N)/np.log(2))

    hypercube_shape = (2,) * 2 * J
    X = np.reshape(Xmat, hypercube_shape)

    Xs = [X]
    for j in range(J):
        X = np.expand_dims(np.expand_dims(Xs[-1], -1), -1)

        X_phi = np.sum(X, axis=(-3-j), keepdims=True)
        X_phi_phi = np.sum(X_phi, axis=(-J-3-j), keepdims=True)
        X_phi_psi = np.diff(X_phi, axis=(-J-3-j))

        X_psi = np.diff(X, axis=(-3-j))
        X_psi_phi = np.sum(X_psi, axis=(-J-3-j), keepdims=True)
        X_psi_psi = np.diff(X_psi, axis=(-J-3-j))

        Y = np.squeeze(np.block(
            [[X_phi_phi, X_phi_psi], [X_psi_phi, X_psi_psi]]))

        Xs.append(np.abs(Y))
    H = np.reshape(np.transpose(np.log1p(Xs[7]),
                        list(-np.arange(1, 2*J+1, 2)) + 
                        list(-np.arange(2, 2*J+1, 2))), (N, N))
    return H


if __name__ == '__main__':
    J = 8
    N = 2**J
    Xmat = np.random.rand(N, N)
    Xmat = Xmat + np.transpose(Xmat)
    H = get_haarscattering(Xmat)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(Xmat)
    plt.subplot(122)
    plt.imshow(H)
    plt.show()