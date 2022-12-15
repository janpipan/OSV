import numpy as np
import matplotlib.pyplot as plt
from vaja6.skripta6 import transformImage, getParameters
from vaja3.skripta3 import loadImage, displayImage
from matplotlib import cm
import time

def exhaustiveRegistration(iImageA, iImageB, iTx, iTy):

    # set start, stop and step parameters
    tx_start = iTx[0]
    tx_stop = iTx[1]
    tx_step = iTx[2]
    tx_range = int((tx_stop-tx_start)/tx_step)+1

    ty_start = iTy[0]
    ty_stop = iTy[1]
    ty_step = iTy[2]
    ty_range = int((ty_stop-ty_start)/ty_step)+1

    # init output arrays
    oMap = np.zeros((ty_range,tx_range))
    oTx = np.arange(tx_start,tx_stop+1,tx_step)
    oTy = np.arange(ty_start,ty_stop+1,ty_step)

    for y in range(0,ty_range):
        for x in range(0,tx_range):
            # transform image
            params = getParameters(iType='affine', rot=0, scale=[1,1], trans=[(x+tx_start)*tx_step,(y+ty_start)*ty_step], shear=[0,0])
            transformedImage = transformImage(iType='affine', iImage=np.copy(iImageB), iDim=[1,1], iP=np.linalg.inv(params), iInterp=1)
            #calculate MSE
            MSE = 0
            for i in range(iImageA.shape[0]):
                for j in range(iImageA.shape[1]):
                        MSE += (iImageA[i, j] - transformedImage[i, j])**2
        
            MSE /= (iImageA.shape[0]* iImageA.shape[1])
            # add MSE result to output map
            oMap[y,x] = MSE
    
    return oMap, oTx, oTy


if __name__ == '__main__':
    size = [83,100]
    imageRef = loadImage('vaja9/vaja09/head-T2-083x100-08bit.raw',size, np.uint8)
    image1 = loadImage('vaja9/vaja09/head-SD-083x100-08bit.raw',size, np.uint8)
    image2 = loadImage('vaja9/vaja09/head-T1-083x100-08bit.raw',size, np.uint8)

    tx = [-15,15,1]
    ty = [-10,20,1]

    displayImage(imageRef, 'Referenčna slika')
    displayImage(image1, 'Slika SD')
    displayImage(image2, 'Slika T1')

    m, x, y = exhaustiveRegistration(imageRef, image1, tx, ty)
    X, Y = np.meshgrid(x,y)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_title("Head-SD")
    ax.set_xlabel("Premik po x-osi")
    ax.set_ylabel("Premik po y-osi")
    ax.set_zlabel("Premik po z-osi")
    ax.plot_surface(X, Y, m, rstride=1, cstride=1)

    

    m1, x1, y1 = exhaustiveRegistration(imageRef, image2, tx, ty)
    X1, Y1 = np.meshgrid(x1,y1)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_title("Head-T1")
    ax.set_xlabel("Premik po x-osi")
    ax.set_ylabel("Premik po y-osi")
    ax.set_zlabel("Premik po z-osi")
    ax.plot_surface(X1, Y1, m1, rstride=1, cstride=1)

    #calculate best 
    min = m[0][0]
    minPos = [0,0]

    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if m[i,j] < min:
                min = m[i,j]
                minPos = [i,j]

    print(x[minPos[1]],y[minPos[0]])

    min = m[0][0]
    minPos = [0,0]

    for i in range(m1.shape[0]):
        for j in range(m1.shape[1]):
            if m1[i,j] < min:
                min = m1[i,j]
                minPos = [i,j]

    print(x1[minPos[1]],y1[minPos[0]])
    
