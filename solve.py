import numpy as np
from derivatives import gradient_A
from helper_functions import StoX


def solve(S0, M,  N1_iters, N2_iters, stepsz_1, stepsz_2, fwd_op, grad, phantom):

    objs = []
    dists = []

    print('Coeff optimisation')
    print('------------------')
    for i in range(N1_iters):
        curr_x = StoX(S0, M)

        S0 = S0 - stepsz_1 * gradient_A(grad(curr_x), M)

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
