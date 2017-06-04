# imports
import numpy as np
import odl

from template import template_array
from poly_phantom import poly_phantom
from helper_functions import spectrum, compton, photo_electric, StoX
from projections import poly_projection
from derivatives import gradient, gradient_A
from xraylib_np import CS_Energy
from solve import solve

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

    out = solve(S0, M,  100, 200, 1, 1, fwd, grad, phantom, Is)
