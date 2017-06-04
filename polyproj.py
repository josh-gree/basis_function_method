from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
import pickle
import xraylib
import astra

#TODO -> Need to sort out better choice of energies, should be able to input specific energies...

def polyimage(Nx,Ny,els,Es):

    rectx1,recty1,rectx2,recty2 = 0.15,0.15,0.85,0.85
    rectx1 = rectx1*Nx
    recty1 = (1 - recty1)*Ny
    rectx2 = rectx2*Nx
    recty2 = (1 - recty2)*Ny

    x1,y1,r1 = 0.3,0.3,0.1

    x1 = x1*Nx
    y1 = (1 - y1)*Ny
    r1 = Nx*r1

    x2,y2,r2 = 0.3,0.7,0.1

    x2 = x2*Nx
    y2 = (1 - y2)*Ny
    r2 = Nx*r2

    x3,y3,r3 = 0.7,0.7,0.1

    x3 = x3*Nx
    y3 = (1 - y3)*Ny
    r3 = Nx*r3

    x4,y4,r4 = 0.7,0.3,0.1

    x4 = x4*Nx
    y4 = (1 - y4)*Ny
    r4 = Nx*r4

    base = Image.new('RGBA', (Nx,Ny), (0,0,0))

    d = ImageDraw.Draw(base)

    rectbbox = [(rectx1,recty1), (rectx2,recty2)]

    bbox1 = [(x1-r1,y1-r1), (x1+r1,y1+r1)]
    bbox2 = [(x2-r2,y2-r2), (x2+r2,y2+r2)]
    bbox3 = [(x3-r3,y3-r3), (x3+r3,y3+r3)]
    bbox4 = [(x4-r4,y4-r4), (x4+r4,y4+r4)]

    d.rectangle(rectbbox,fill=(1,0,0))
    d.ellipse(bbox1,fill=(2,0,0))
    d.ellipse(bbox2,fill=(3,0,0))
    d.ellipse(bbox3,fill=(4,0,0))
    d.ellipse(bbox4,fill=(5,0,0))

    base_array = np.array(base.getdata())
    material_array = base_array.reshape(Nx,Ny,4)[:,:,0]

    airidx = np.where(material_array==0)
    m1idx = np.where(material_array==1)
    m2idx = np.where(material_array==2)
    m3idx = np.where(material_array==3)
    m4idx = np.where(material_array==4)
    m5idx = np.where(material_array==5)

    atts = np.array([np.array([xraylib.CS_Total(el, e) for e in Es]) for el in els])
    atts = atts.T
    
    arrays = []
    for i in range(len(Es)):
        array = np.zeros_like(material_array,dtype=float)
        array[m1idx] = atts[i,0]
        array[m2idx] = atts[i,1]
        array[m3idx] = atts[i,2]
        array[m4idx] = atts[i,3]
        array[m5idx] = atts[i,4]
        array = array.flatten(order="F")
        arrays.append(array)
    
    return np.vstack(arrays).T

def poly_image(Nx,Ny,els = [20,47,40,47,40]):
    '''
    Default elements surrounds Ca (ie bone like), 2 circles of Ag (silver), 2 circles of Zr (Zirconium)
    '''
    spectra = pickle.load(open("/home/josh/nine_month/data/E_spectra.p","rb"))
    es = spectra[::25,:][:,0] #steps of 25 give 3 energy bins
    fes = spectra[::25,:][:,1]
    Es = [(x[0]+x[1])/2 for x in zip(es[:-1],es[1:])]
    FES = [(x[0]+x[1])/2 for x in zip(fes[:-1],fes[1:])]
    ESminus = [x[1]-x[0] for x in zip(es[:-1],es[1:])]
    Is = [x[0]*x[1] for x in zip(FES,ESminus)]
    Is = np.array(Is)
    Is = Is/Is.sum()

    #scale number of photons
    I = Is*1e5
    I = I[:,np.newaxis] #make 2d 
    return polyimage(Nx,Ny,els,Es),I