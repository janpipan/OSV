import numpy as np
import matplotlib.pyplot as plt
from vaja3.skripta3 import displayImage


def loadImage3D(iPath, iSize, iType):

    fid = open(iPath, 'rb')
    buffer = fid.read()
    

    oShape = [iSize[1], iSize[0], iSize[2]]

    oImage = np.ndarray(oShape, dtype=iType, buffer=buffer, order='F')
    fid.close()
    return oImage

def getPlanarCrossSection(iImage, iDim, iNormVec, iLoc):

    Y, X, Z = iImage.shape
    dx, dy, dz, = iDim

    if iNormVec == [1,0,0]:
        oCS = iImage[:,iLoc,:].T
        oV = np.arange(Z) * dz
        oH = np.arange(Y) * dy
    elif iNormVec == [0,1,0]:
        oCS = iImage[iLoc,:,:].T
        oV = np.arange(Z) * dz
        oH = np.arange(X) * dx
    elif iNormVec == [0,0,1]:
        oCS = iImage[:,:,iLoc]
        oV = np.arange(Y) * dy
        oH = np.arange(X) * dx
    else:
        print("Napacen iNormVec")

    return np.copy(oCS), oH, oV


def displayImage(iImage, iTitle, iGridX, iGridY):
    plt.figure()
    plt.title(iTitle)
    plt.imshow(iImage, cmap='gray',aspect='equal',extent=(iGridX[0], iGridX[-1], iGridY[0], iGridY[-1]))

def getPlannarProjection(iImage, iDim, iNormVec, iFunc):

    Y, X, Z = iImage.shape
    dx, dy, dz, = iDim

    # axis 0 = vrstice, y smer
    # axis 1 = stolpci, x smer
    # axis 2 = rezine, z smer

    if iNormVec == [1,0,0]:
        oP = iFunc(iImage, axis=1).T
        oV = np.arange(Z) * dz
        oH = np.arange(Y) * dy
    elif iNormVec == [0,1,0]:
        oP = iFunc(iImage, axis=0).T
        oV = np.arange(Z) * dz
        oH = np.arange(X) * dx
    elif iNormVec == [0,0,1]:
        oP = iFunc(iImage, axis=2).T
        oV = np.arange(Y) * dy
        oH = np.arange(X) * dx
    else:
        raise NotImplementedError('Neznani iNormVec')
        
    return oP, oH, oV


if __name__ == '__main__':
    imSize = [512, 58, 907]
    vxDim = [0.597656, 3, 0.597656]
    I = loadImage3D('./vaja4/data/spine-512x058x907-08bit.raw', iSize=imSize, iType=np.uint8)
    print(I.shape)

    xI, xH, xV = getPlanarCrossSection(I, iDim=vxDim, iNormVec=[1,0,0], iLoc=250)
    displayImage(xI, "Stranski prerez", iGridX=xH, iGridY=xV)
    xI, xH, xV = getPlanarCrossSection(I, iDim=vxDim, iNormVec=[0,1,0], iLoc=45)
    displayImage(xI, "Celni prerez", iGridX=xH, iGridY=xV)
    xI, xH, xV = getPlanarCrossSection(I, iDim=vxDim, iNormVec=[0,0,1], iLoc=250)
    displayImage(xI, "Prečni prerez", iGridX=xH, iGridY=xV)

    func = np.average
    PxI, PxH, PxV = getPlannarProjection(I, iDim=vxDim, iNormVec=[1,0,0],iFunc=func)
    displayImage(PxI,"Stranska projekcija", PxH, PxV)
    PxI, PxH, PxV = getPlannarProjection(I, iDim=vxDim, iNormVec=[0,1,0],iFunc=func)
    displayImage(PxI,"Stranska projekcija", PxH, PxV)
    PxI, PxH, PxV = getPlannarProjection(I, iDim=vxDim, iNormVec=[0,0,1],iFunc=func)
    displayImage(PxI,"Stranska projekcija", PxH, PxV)


## odštej center, prištej center