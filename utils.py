# imports
import numpy as np
import odl

from template import template_array
from poly_phantom import poly_phantom
from helper_functions import spectrum, compton, photo_electric, StoX
from projections import poly_projection
from derivatives import gradient, gradient_A
from xraylib_np import CS_Energy

# set up projectors
Nx, Ny = 256, 256
Np, Nd = 1024, 512
Ne = 10
Es, Is = spectrum(Ne, 1e7)

space = odl.uniform_discr([-1, -1], [1, 1], [Nx, Ny], dtype='float32')
detector = odl.uniform_partition(-1.2, 1.2, Nd)
angle = odl.uniform_partition(0, 2 * np.pi, Np)
geometry = odl.tomo.Parallel2dGeometry(angle, detector)
operator = odl.tomo.RayTransform(
    space, geometry, impl="astra_cuda", use_cache=True)
fbp_op = odl.tomo.fbp_op(operator)


def fwd(X): return operator(X).asarray()


def bwd(X): return operator.adjoint(X).asarray()


def solve(S0, M,  N1_iters, N2_iters, stepsz_1, stepsz_2, fwd_op, grad, phantom):

    objs = []
    dists = []

    print('Coeff optimisation')
    print('------------------')
    for i in range(N1_iters):
        curr_x = StoX(S0, M)

        S0 = S0 - stepsz_1 * gradient_A(grad(curr_x))

        S0[S0 < 0] = 0

        obj_val = obj(curr_x, Is, sino, fwd_op)
        dist_val = np.linalg.norm(curr_x - phantom)

        objs.append(obj_val)
        dists.append(dist_val)

        print('[iter {}] Objective val : {}, Distance to true : {}'.format(
            i, obj_val, dist_val))

    x0 = StoX(S0, M)

    print('Full optimisation')
    print('------------------')
    for j in range(N2_iters):
        x0 = x0 - stepsz_2 * grad(x0)

        x0[x0 < 0] = 0

        obj_val = obj(x0, Is, sino, fwd_op)
        dist_val = np.linalg.norm(x0 - phantom)

        objs.append(obj_val)
        dists.append(dist_val)

        print('[iter {}] Objective val : {}, Distance to true : {}'.format(
            j, obj_val, dist_val))

    return x0, objs, dists


if __name__ == '__main__':

    def fwd(X): return operator(X).asarray()

    def bwd(X): return operator.adjoint(X).asarray()

    # make data
    phantom = poly_phantom(template_array())
    sino = poly_projection(fwd, phantom, Is)
    fbp = fbp_op(sino).asarray()

    # gradient
    def grad(X): return gradient(X, Is, sino, Nx, Ny, Np, Nd, Ne, fwd, bwd)

    # M
    materials = [10, 20, 30, 45, 50]
    material_profile = CS_Energy(np.array(materials), np.array(Es))

    M = material_profile

    S0 = np.ones((5, Nx, Ny))

    out = solve(S0, M,  100, 200, 1, 1, fwd, grad, phantom)
