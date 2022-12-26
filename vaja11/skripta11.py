import numpy as np
import matplotlib.pyplot as plt
import cv2

def displayImage(iImage, iTitle, iGridX=None, iGridY=None, points=None):
    fig = plt.figure()
    plt.title(iTitle)
    if iGridX is None or iGridY is None:
        plt.imshow(iImage, cmap='gray',aspect='equal',vmin=0, vmax=255)
    else:
        plt.imshow(iImage, cmap='gray',aspect='equal',extent=[iGridX[0], iGridX[-1], iGridY[0], iGridY[-1]],vmin=0, vmax=255)
    #if points == None:
        #plt.show()
    return fig

def loadFrame(iVideo, iK):
    
    iVideo.set(1, iK-1)

    ret, oFrame = iVideo.read()

    oFrame = oFrame[:,:,0].astype(float)
    
    return oFrame

def framePrediction(iF, iMV):
    iMV = np.copy(iMV).astype(int)
    oF = np.roll(iF, [iMV[1], iMV[0]], axis=(0,1))
    
    if iMV[0] >= 0:
        oF[:, :iMV[0]] = -1
    else:
        oF[:, iMV[0] :] = -1

    if iMV[1] >= 0:
        oF[: iMV[1], :] = -1
    else:
        oF[iMV[1] :, :] = -1
    
    return oF 


def blockMatching(iF1, iF2, iSize, iSearchSize):
    
    
    Y, X = iF1.shape

    Bx, By = iSize

    M = int(X/Bx)# st stolpcev
    N = int(Y/By) # st vrstic

    oMF = np.zeros((N, M, 2), dtype=float)
    oCP = np.zeros((N, M, 2), dtype=float)
    Err = np.ones((N, M), dtype=float) * 255

    P = int((iSearchSize - 1)/2)
    PTS = np.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [1, -1], [-1, 1], [1, 1]])

    for n in range(N):
        ymin = n * By
        ymax = (n + 1) * By
        y = np.arange(ymin, ymax)

        for m in range(M):
            xmin = m * Bx
            xmax = (m + 1) * Bx
            x = np.arange(xmin, xmax)

            oCP[n, m, 0] = x.mean()
            oCP[n, m, 1] = y.mean()

            B1 = iF1[ymin : ymax, xmin : xmax]
            B2 = iF2[ymin : ymax, xmin : xmax]

            for i in range(1,4):
                Pi = (P+1)/2 ** i
                PTSi = PTS * Pi
                d0 = oMF[n, m, :] #vektor premika (x,y)
                
                for p in range(PTSi.shape[0]):
                    # trenutni vektor premika
                    d = d0 + PTSi[p,:]
                    pF2 = framePrediction(iF1, d) # napoved F2 na podlagi F1 in trenutnega d
                    pB2 = pF2[ymin:ymax, xmin:xmax]
                    # izracun napake napovedi, ampak samo za vrednosti, ki so večje od -1
                    idx = np.logical_and(B2 >= 0, pB2 >= 0)
                    bErr = np.sum(np.abs(B2[idx] - pB2[idx]))/idx.sum()

                    if bErr < Err[n, m]:
                        Err[n, m] = bErr
                        oMF[n, m, :] = d
    return oMF, oCP

def displayMotionField(iMF, iCP, iTitle, iImage=None):

    if iImage is None:
        fig = plt.figure()
        plt.gca().invert_yaxis()
        plt.gca().set_aspect('equal')
        plt.title(iTitle)
    else: 
        fig = displayImage(iImage=iImage, iTitle=iTitle)

    plt.quiver(
        iCP[:,:,0],
        iCP[:,:,1],
        iMF[:,:,0],
        iMF[:,:,1],
        color='r',
        scale=0.5,
        units='xy',
        angles='xy'
    )
    plt.show()

    #return fig


if __name__ == '__main__':
    cap = cv2.VideoCapture('vaja11/data/simple-video.avi')
    frame30 = loadFrame(cap, iK = 30)
    frame31 = loadFrame(cap, iK = 31)
    displayImage(frame30, 'Frame 1')
    displayImage(frame31, 'Frame 2')
    bSize = [8, 8]
    searchSize = 15 # 2**4-1
    mf, cp = blockMatching(frame30, frame31, iSize=bSize, iSearchSize=searchSize)
    displayMotionField(mf, cp, 'Vektorji premika', iImage=frame30)