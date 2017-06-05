import matplotlib
matplotlib.use('Agg')

from helper_functions import spectrum

import numpy as np
import matplotlib.pyplot as plt
import deepdish as dd
import seaborn as sb
import pathlib

Es, Is = spectrum(30, 1e7)


def ims(data_loc):

    p = pathlib.Path(data_loc)

    data = dd.io.load('data.h5')
    sino = data['sino']
    phantom = data['phantom']
    fbp = data['fbp']

    sol = dd.io.load(bytes(p))

    M = sol['M']
    dists = sol['dists']
    objs = sol['objs']
    S0 = sol['S0']
    sol = sol['sol']

    sb.set(font_scale=1.4)
    sb.set_style({'axes.grid': False, 'image.cmap': 'viridis'})
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), sharey=True)

    ax[0].imshow(fbp, extent=[-1, 1, -1, 1])
    ax[0].set_xlabel('$x$')
    ax[0].set_ylabel('$y$')
    ax[0].set_title('FBP reconstruction')
    ax[1].imshow(sol[5:, ...].mean(axis=0), extent=[-1, 1, -1, 1])
    ax[1].set_xlabel('$x$')
    ax[1].set_title('Solution - Average Energy')
    ax[2].imshow(phantom[5:, ...].mean(axis=0), extent=[-1, 1, -1, 1])
    ax[2].set_xlabel('$x$')
    ax[2].set_title('Phantom - Average Energy')

    plt.savefig('ims_' + p.name.split('.')[0] + '.png')


def energy_profiles(data_loc):

    p = pathlib.Path(data_loc)

    data = dd.io.load('data.h5')
    sino = data['sino']
    phantom = data['phantom']
    fbp = data['fbp']

    sol = dd.io.load(bytes(p))

    M = sol['M']
    dists = sol['dists']
    objs = sol['objs']
    S0 = sol['S0']
    sol = sol['sol']

    sb.set_style("whitegrid", {'axes.grid': True})

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 7))
    ax.plot(Es, sol[:, 125, 50], label='Solution')
    ax.plot(Es, phantom[:, 125, 50], label='Phantom')
    ax.set_xticks(Es[::2])
    ax.set_xlabel('Energy KeV', fontsize=15)
    ax.set_ylabel('Attenuation', fontsize=15)
    ax.tick_params(labelsize=15)
    plt.legend(loc=0, fontsize=15)
    plt.savefig('E_profile1_' + p.name.split('.')[0] + '.png')

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 7))
    ax.plot(Es, sol[:, 190, 160], label='Solution')
    ax.plot(Es, phantom[:, 190, 160], label='Phantom')
    ax.set_xticks(Es[::2])
    ax.set_xlabel('Energy KeV', fontsize=15)
    ax.set_ylabel('Attenuation', fontsize=15)
    ax.tick_params(labelsize=15)
    plt.legend(loc=0, fontsize=15)
    plt.savefig('E_profile2_' + p.name.split('.')[0] + '.png')


def line_profiles(data_loc):

    p = pathlib.Path(data_loc)

    data = dd.io.load('data.h5')
    sino = data['sino']
    phantom = data['phantom']
    fbp = data['fbp']

    sol = dd.io.load(bytes(p))

    M = sol['M']
    dists = sol['dists']
    objs = sol['objs']
    S0 = sol['S0']
    sol = sol['sol']

    for i in [0, 7, 15, 23, 29]:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
        ax.set_title('{:0.2f} KeV'.format(Es[i]))
        x = np.linspace(-1, 1, 256)
        ax.plot(x, phantom[i, ...][130, :], label='Phantom')
        ax.plot(x, fbp[130, :], label='fbp')
        ax.plot(x, sol[i, ...][130, :], label='Solution')
        ax.set_xlabel('$y$', fontsize=15)
        ax.set_ylabel('$Attenuation$', fontsize=15)
        plt.legend(loc=0)
        plt.savefig('profile_{:0.2f}KeV_'.format(
            Es[i]) + p.name.split('.')[0] + '.png')


def coeffs(data_loc, names=None):
    p = pathlib.Path(data_loc)

    data = dd.io.load('data.h5')
    sino = data['sino']
    phantom = data['phantom']
    fbp = data['fbp']

    sol = dd.io.load(bytes(p))

    M = sol['M']
    dists = sol['dists']
    objs = sol['objs']
    S0 = sol['S0']
    sol = sol['sol']

    if not names:
        names = [str(x) for x in range(M.shape[0])]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
    x = np.linspace(-1, 1, 256)
    for i in range(M.shape[0]):
        ax.plot(x, S0[:, 130, :][i, :],
                label='basis function : {}'.format(names[i]))
    ax.set_xlabel('$y$', fontsize=15)
    ax.set_ylabel('$Attenuation$', fontsize=15)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig('coeffs_' + p.name.split('.')[0] + '.png')
