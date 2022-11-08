import numpy as np
import matplotlib.pyplot as plt
from vaja3.skripta3 import displayImage, saveImage
import time 


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

    # image height width and depth
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
    elif iNormVec[2] == 0:
        phi = np.arctan(iNormVec[1]/iNormVec[0])
        #phi = 30*np.pi/180
        cos, sin = np.cos(phi), np.sin(phi)
        multMat = [[cos, -sin],[sin, cos]]
        oP = np.zeros(iImage.shape)
        # center points of old picture
        y_center, x_center = Y / 2, X / 2
        """ for z in range(Z):
            for y in range(Y):
                for x in range(X):
                    x_pos = x - cX
                    y_pos = y - cY
                    new_x = int(np.round(x_pos*cos - y_pos*sin)) + cX
                    new_y = int(np.round(x_pos*sin + y_pos*cos)) + cY
                    if new_x >= 0 and new_x < X and new_y >= 0 and new_y < Y:
                        oP[new_y][new_x][z] = iImage[y][x][z]  """ 

        """ for z in range(Z):
            for y in range(Y):
                for x in range(X):
                    y_pos = Y - 1 - y - cY
                    x_pos = X - 1 - x - cX
                    new_y=round(-x_pos*sin+y_pos*cos)
                    new_x=round(x_pos*cos+y_pos*sin)
                    new_y= cY - new_y
                    new_x = cX - new_x

                    if new_x >= 0 and new_y >= 0 and new_x < X and new_y < Y:
                        oP[new_y,new_x,z] = iImage[new_y,new_x,z] """
        """ # creates one array for x indexes and one array for y indexes
        y_index, x_index = np.indices((Y,X))
        y_index, x_index = y_index.flatten(), x_index.flatten()

        # calculates the values in coordiante system
        y_val, x_val = (y_index - y_center) * dy, (x_index - x_center) * dx
        
        # create coordinate array and multiply it with rotation matrix
        # coordinate array has 
        coordinates = np.vstack((x_val,y_val))
        x_new_val, y_new_val = np.matmul(multMat, coordinates)

        # translate new values back to index values
        x_new_index = np.round((x_new_val / dx) + x_center)
        y_new_index = np.round((y_new_val / dy) + y_center)
        x_new_index = x_new_index.astype(int)
        y_new_index = y_new_index.astype(int)


        for z in range(Z):
            # for every point in image
            for i in range(len(x_new_index)):
                # if point is inside the image assign new add value to output image
                if x_new_index[i] >= 0 and x_new_index[i] < X and y_new_index[i] >= 0 and y_new_index[i] < Y:
                    #oP[new_y[i]-1][new_x[i]-1][z] = iImage[yr[i]][xr[i]][z]
                    oP[y_index[i]][x_index[i]][z] = iImage[y_new_index[i]][x_new_index[i]][z]
         """
        for z in range(Z):
            for y in range(Y):
                for x in range(X):
                    y_val, x_val = (y - y_center) * dy, (x - x_center) * dx
                    y_new_val, x_new_val = x_val*sin + y_val*cos, x_val*cos - y_val*sin
                    x_new_index = np.round((x_new_val / dx) + x_center)
                    y_new_index = np.round((y_new_val / dy) + y_center)
                    x_new_index = x_new_index.astype(int)
                    y_new_index = y_new_index.astype(int)
                    if x_new_index >= 0 and x_new_index < X and y_new_index >= 0 and y_new_index < Y:
                        #oP[new_y[i]-1][new_x[i]-1][z] = iImage[yr[i]][xr[i]][z]
                        oP[y][x][z] = iImage[y_new_index][x_new_index][z]

        oP = iFunc(oP,axis=0).T
        oV = np.arange(Z) * dz
        oH = np.arange(X) * dx
    else:
        raise NotImplementedError('Neznani iNormVec')
        
    return np.copy(oP), oH, oV


if __name__ == '__main__':
    imSize = [512, 58, 907]
    vxDim = [0.597656, 3, 0.597656]
    I = loadImage3D('./vaja4/data/spine-512x058x907-08bit.raw', iSize=imSize, iType=np.uint8)
    print(I.shape)

    """ xI, xH, xV = getPlanarCrossSection(I, iDim=vxDim, iNormVec=[1,0,0], iLoc=250)
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
    displayImage(PxI,"Stranska projekcija", PxH, PxV) """

    # 1. naloga
    # Priložite slike naslednjih pravokotnih ravninskih prerezov dane 3D slike:
    # - stranski prerez na položaju x_c = 256 slikovnih elementov.
    # - čelni preprez na položaju y_c = 35 slikovnih elementov
    # - prečni prerez na položaju z_c = 467 slikovnih elementov
    # Priložene slike naj bodo v dimenzijskem sorazmerju z velikostjo slikovnega elementa dane 3D slike

    xI, xH, xV = getPlanarCrossSection(I, iDim=vxDim, iNormVec=[1,0,0], iLoc=256)
    displayImage(xI, "Stranski prerez", iGridX=xH, iGridY=xV)
    xI, xH, xV = getPlanarCrossSection(I, iDim=vxDim, iNormVec=[0,1,0], iLoc=35)
    displayImage(xI, "Čelni prerez", iGridX=xH, iGridY=xV)
    xI, xH, xV = getPlanarCrossSection(I, iDim=vxDim, iNormVec=[0,0,1], iLoc=467)
    displayImage(xI, "Prečni prerez", iGridX=xH, iGridY=xV)

    # 2. naloga
    # Priložite slike stranske, čelne in prečne pravokotne ravninske projekcije dane 3D slike, pri 
    # čemer za funkcijo točk PF uporabite:
    # - makismalno vrednost,
    # - povprečno vrednost.
    # Priložene slike naj bodo v dimenzijskem sorazmerju z velikostjo slikovnega elementa dane 3D slike
    
    # maksimalna vrednost
    func = np.max
    PxI, PxH, PxV = getPlannarProjection(I, iDim=vxDim, iNormVec=[1,0,0],iFunc=func)
    displayImage(PxI,"Stranska projekcija z maksimalno vrednostjo", PxH, PxV)
    PxI, PxH, PxV = getPlannarProjection(I, iDim=vxDim, iNormVec=[0,1,0],iFunc=func)
    displayImage(PxI,"Čelna projekcija z maksimalno vrednostjo", PxH, PxV)
    PxI, PxH, PxV = getPlannarProjection(I, iDim=vxDim, iNormVec=[0,0,1],iFunc=func)
    displayImage(PxI,"Prečna projekcija z maksimalno vrednostjo", PxH, PxV) 

    # povprečna vrednost
    func = np.average
    PxI, PxH, PxV = getPlannarProjection(I, iDim=vxDim, iNormVec=[1,0,0],iFunc=func)
    displayImage(PxI,"Stranska projekcija s povprečno vrednostjo", PxH, PxV)
    PxI, PxH, PxV = getPlannarProjection(I, iDim=vxDim, iNormVec=[0,1,0],iFunc=func)
    displayImage(PxI,"Čelna projekcija s povprečno vrednostjo", PxH, PxV)
    PxI, PxH, PxV = getPlannarProjection(I, iDim=vxDim, iNormVec=[0,0,1],iFunc=func)
    displayImage(PxI,"Prečna projekcija s povprečno vrednostjo", PxH, PxV) 

    # 3. naloga
    # Katere vrste projekcij, pri katerih za funkcijo točk uporabite makismalno vrednost, minimalno
    # vrednost, povprečno vrednost, vrednost mediane, vrednost standardnega odklona oz. vrednost 
    # variance, je v primeru prikazovanja CT slik človeškega telesa sploh smilseno računati?
    # Obrazložite odgovor.

    # 4. naloga
    func = np.max
    start = time.time()
    PxI, PxH1, PxV1 = getPlannarProjection(I, iDim=vxDim, iNormVec=[3.83,9.24,0],iFunc=func)
    displayImage(PxI,"Prečna projekcija s povprečno vrednostjo", PxH1, PxV1) 
    end = time.time()
    print(f"{end - start}")
    #saveImage(PxI, "./vaja4/data/image.raw", np.uint8)
    #PxI, PxH1, PxV1 = getPlannarProjection(I, iDim=vxDim, iNormVec=[1,1,0],iFunc=func)
    #displayImage(PxI,"Prečna projekcija s povprečno vrednostjo", PxH1, PxV1) 
    #saveImage(PxI, "./vaja4/data/image.raw", np.uint8)
    #PxI, PxH1, PxV1 = getPlannarProjection(I, iDim=vxDim, iNormVec=[9.24,3.83,0],iFunc=func)
    #displayImage(PxI,"Prečna projekcija s povprečno vrednostjo", PxH1, PxV1) 
    #saveImage(PxI, "./vaja4/data/image.raw", np.uint8)

    


## odštej center, prištej center