# imports
import numpy as np
import odl
import deepdish as dd

from template import template_array
# need to be able to control number of energies!
from poly_phantom import poly_phantom
from helper_functions import spectrum, compton, photo_electric
from projections import poly_projection
from derivatives import gradient
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


def fwd(X): return operator(X).asarray()


def bwd(X): return operator.adjoint(X).asarray()


# make data
phantom = poly_phantom(template_array())
sino = poly_projection(fwd, phantom, Is)
fbp = fbp_op(sino).asarray()

# gradient


def grad(X): return gradient(X, Is, sino, Nx, Ny, Np, Nd, Ne, fwd, bwd)


# # M - CO + PE
CO = compton(Es)
CO /= np.linalg.norm(CO)
PE = photo_electric(Es)
PE /= np.linalg.norm(PE)

M = np.vstack([CO, PE])

S0 = np.ones((2, Nx, Ny))
sol, objs, dists = solve(S0, M,  2, 3, .01, .1,
                         fwd, grad, phantom, Is, sino)

dd.io.save('COPE.h5', {'sol': sol, 'objs': objs, 'dists': dists})
# # M - known materials
materials = [14, 35, 40, 45, 50]
M = CS_Energy(np.array(materials), np.array(Es))

S0 = np.ones((5, Nx, Ny))
sol, objs, dists = solve(S0, M,  100, 400, .01, .1,
                         fwd, grad, phantom, Is, sino)

dd.io.save('known_ms.h5', {'sol': sol, 'objs': objs, 'dists': dists})

# # M - some materials, not in phantom
materials = [10, 23, 32, 41, 53]
M = CS_Energy(np.array(materials), np.array(Es))

S0 = np.ones((5, Nx, Ny))
sol, objs, dists = solve(S0, M,  60, 440, .01, .1,
                         fwd, grad, phantom, Is, sino)

dd.io.save('rand_ms.h5', {'sol': sol, 'objs': objs, 'dists': dists})


# M - lots of materials
materials = list(range(10, 60, 4))
M = CS_Energy(np.array(materials), np.array(Es))

S0 = np.ones((len(materials), Nx, Ny))
sol, objs, dists = solve(S0, M,  120, 380, .01, .1,
                         fwd, grad, phantom, Is, sino)

dd.io.save('lots_ms.h5', {'sol': sol, 'objs': objs, 'dists': dists})
