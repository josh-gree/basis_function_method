from numpy import exp, vstack, zeros, outer
from projections import multi_mono_prj, poly_projection


def gradient(X, I, y, Nx, Ny, Np, Nd, Ne, fwd, bwd):

    E = exp(-multi_mono_prj(fwd, X))
    if Ne == 1:
        E = E.flatten()
    else:
        E = vstack([E[i, ...].flatten() for i in range(Ne)]).T

    Yx = poly_projection(fwd, X, I)
    Yx = Yx.flatten()
    y = y.flatten()

    if Ne == 1:
        e = (Yx - y) / E * I
        outer_prod = e * I * E
    else:
        e = (Yx - y) / E.dot(I)
        outer_prod = outer(e, I) * E

    if Ne == 1:
        gradient = zeros((Nx, Ny))
        gradient = bwd(outer_prod.reshape(Np, Nd)).reshape(Nx, Ny)
    else:
        gradient = zeros((Ne, Nx, Ny))
        for i in range(Ne):
            gradient[i, ...] = bwd(
                outer_prod[:, i].reshape(Np, Nd)).reshape(Nx, Ny)

    return gradient


def gradient_A(df, M, scales=None):
    if scales:
        g = (df.reshape(M.shape[1], -1).T.dot(M.T)
             ).T.reshape(-1, df.shape[1], df.shape[2])
        return g * scales[..., np.newaxis, np.newaxis]
    else:
        g = (df.reshape(M.shape[1], -1).T.dot(M.T)
             ).T.reshape(-1, df.shape[1], df.shape[2])
        return g * np.ones(M.shape[0])[..., np.newaxis, np.newaxis]
