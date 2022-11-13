from mpld3 import display
import numpy as np
from vaja1.main import loadImage
from vaja3.skripta3 import displayImage


def scaleImage(iImage, iA, iB):
    return iA*np.array(iImage, dtype=float) + iB

def windowImage(iImage, iC, iW):
    #uporabno za medicinske slike
    oImage = np.array(iImage, dtype=float)
    # najprej splošna linearna preslikava
    oImage = 255 / iW * (oImage-(iC - iW/2))
    oImage[iImage < (iC-iW/2)] = 0
    oImage[iImage > (iC+iW/2)] = 255
    return oImage

def sectionalScaleImage(iImage, iS, oS):

    oImage = np.array(iImage, dtype=float)

    for i in range(len(iS)-1):
        sp_meja_vhd = iS[i]
        zg_meja_vhd = iS[i+1]

        sp_meja_izh = oS[i]
        zg_meja_izh = oS[i+1]

        index = np.logical_and(iImage >= sp_meja_vhd, iImage <= zg_meja_vhd)
        oImage[index] = (zg_meja_izh - sp_meja_izh) / (zg_meja_vhd - sp_meja_vhd) * (iImage[index] - sp_meja_vhd) + sp_meja_izh


    return oImage

def gammaImage(iImage, iG):
    return 255**(1-iG)*np.array(iImage, dtype=float)**iG

if __name__ == '__main__':
    orig_size = [512,512]
    image = loadImage(r'./vaja5/data/image-512x512-16bit.raw', orig_size, np.int16)
    displayImage(image, 'slika')

    sI = scaleImage(image, iA=-0.125,iB=256)
    displayImage(sI, 'Slika po splosni lin. preslikavi')
    print(image.min(),image.max(),sI.min(),sI.max())
    
    wI = windowImage(sI, iC=1000, iW=500)
    displayImage(wI, 'Slika po linearnem oknenju')
    print(image.min(),image.max(),wI.min(),wI.max())

    s_vhd = [0,85,170,255]
    s_izh = [85,0,255,170]
    ssI = sectionalScaleImage(wI, iS=s_vhd, oS=s_izh)
    displayImage(ssI, 'Slika po odsekoma lin. preslikavi')

    gI = gammaImage(wI, 0.5)
    displayImage(gI, 'Slika po gamma preslikavi')
    