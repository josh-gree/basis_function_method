from projections import poly_projection
from numpy.linalg import norm


def residual(X, I, y, fwd):
    '''
    difference between model and reality

    r(X) = -log(exp(-AX)I) - y

    3D Matrix -> 2D Matrix
    '''
    return poly_projection(fwd, X, I) - y


def obj(X, I, y, fwd):
    '''
    L2 objective (1/2)||r(X)||^2_2

    sum of squared deviation from reality...

    3D Matrix -> scalar
    '''
    r = residual(X, I, y, fwd)
    return norm(r)**2
