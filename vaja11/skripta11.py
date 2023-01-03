import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import io


def displayImage(iImage, iTitle, iGridX=None, iGridY=None, points=None):
    fig = plt.figure()
    plt.title(iTitle)
    if iGridX is None or iGridY is None:
        plt.imshow(iImage, cmap=plt.cm.gray, vmin=0, vmax=255, aspect="equal")
    else:
        plt.imshow(
            iImage,
            cmap=plt.cm.gray,
            vmin=0,
            vmax=255,
            aspect="equal",
            extent=[iGridX[0], iGridX[-1], iGridY[0], iGridY[-1]],
        )
    return fig


def loadFrame(iVideo, iK):
    """
    Funkcija za nalaganje slike iz videa
    """

    iVideo.set(1, iK - 1)
    ret, oFrame = iVideo.read()
    oFrame = oFrame[:, :, 0].astype(float)

    return oFrame

def framePrediction(iF, iMV):
    iMV = np.array(iMV).astype(int)
    oF = np.roll(iF, [iMV[1], iMV[0]], axis=(0, 1))

    if iMV[0] >= 0:
        oF[:, : iMV[0]] = -1
    else:
        oF[:, iMV[0] :] = -1

    if iMV[1] >= 0:
        oF[: iMV[1], :] = -1
    else:
        oF[iMV[1] :, :] = -1

    return oF


def blockMatching(iF1, iF2, iSize, iSearchSize):
    """
    Funkcija za dolocanje polja vektorjev premika z blocnim
    ujemanjem
    """

    Y, X = iF1.shape
    Bx, By = iSize

    M = int(X / Bx)
    N = int(Y / By)

    oMF = np.zeros((N, M, 2), dtype=float)
    oCP = np.zeros((N, M, 2), dtype=float)
    Err = np.ones((N, M), dtype=float) * 255

    P = int((iSearchSize - 1) / 2)
    PTS = np.array(
        [[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [1, -1], [-1, 1], [1, 1]]
    )

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

            B1 = iF1[ymin:ymax, xmin:xmax]
            B2 = iF2[ymin:ymax, xmin:xmax]

            for i in range(1, 4):
                Pi = (P + 1) / (2 ** i)
                PTSi = PTS * Pi
                d0 = oMF[n, m]

                for p in range(PTSi.shape[0]):
                    d = d0 + PTSi[p, :]
                    # napoved slike 2 na podlagi slike 1 in vektorja d
                    pF2 = framePrediction(iF1, d)
                    pB2 = pF2[ymin:ymax, xmin:xmax]

                    idx = np.logical_and(B2 >= 0, pB2 >= 0)
                    bErr = np.sum(np.abs(B2[idx] - pB2[idx])) / idx.sum()

                    if bErr < Err[n, m]:
                        oMF[n, m, :] = d
                        Err[n, m] = bErr

    return oMF, oCP

def displayMotionField(iMF, iCP, iTitle, iImage=None):
    """
    Funkcija za prikaz polja vektorjev premika
    """

    if iImage is None:
        fig = plt.figure()
        plt.gca().invert_yaxis()
        plt.gca().set_aspect("equal")
        plt.title(iTitle)
    else:
        fig = displayImage(iImage=iImage, iTitle=iTitle)

    plt.quiver(
        iCP[:, :, 0],
        iCP[:, :, 1],
        iMF[:, :, 0],
        iMF[:, :, 1],
        color="r",
        scale=0.5,
        units="xy",
        angles="xy",
    )
    return fig


def predictImage(iImage, iMF, iCP, iSize):

    oImage = np.zeros(iImage.shape).astype(int)
    yCP, xCP, zCp = iCP.shape
    Bx, By = iSize
    
    for n in range(yCP):
        for m in range(xCP):
            ymin = n * By
            ymax = (n + 1) * By
            xmin = m * Bx
            xmax = (m + 1) * Bx

            oImage[int(ymin+iMF[n,m,1]):int(ymax+iMF[n,m,1]),int(xmin+iMF[n,m,0]):int(xmax+iMF[n,m,0])] = iImage[ymin:ymax,xmin:xmax]
            
    return oImage


def fig2img(fig):
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

def motionFieldGIF(iPath, vLen, iSize,iSearchSize):
    oFrames = []
    cap = cv2.VideoCapture(iPath)
    for i in range(vLen - 1):
        frame1 = loadFrame(cap, iK = i)
        frame2 = loadFrame(cap, iK = i + 1)
        mf, cp = blockMatching(frame1, frame2, iSize=iSize, iSearchSize=iSearchSize)
        oFrames.append(fig2img(displayMotionField(mf,cp,'Vektorji premika', iImage=frame1)))
    return oFrames


if __name__ == '__main__':
    cap = cv2.VideoCapture('vaja11/data/simple-video.avi')
    frame30 = loadFrame(cap, iK = 30)
    frame31 = loadFrame(cap, iK = 31)
    displayImage(frame30, 'Frame 1')
    displayImage(frame31, 'Frame 2')
    bSize = [8, 8]
    searchSize = 15 # 2**4-1
    mf, cp = blockMatching(frame30, frame31, iSize=bSize, iSearchSize=searchSize)
    displayMotionField(mf, cp, 'Vektorji premika')
    displayMotionField(mf, cp, 'Vektorji premika', iImage=frame30)

    # 1. naloga
    # Izračunajte napoved slike pri času t2 > t1 na podlagi slike pri času t1 ter
    # ter pripadajočo sliko razlik. Priložite programsko kodo in izirse obeh slik.

    # 2. naloga 
    # Sestavite video polja vektorjev premika za vsak par zaporednih slik danega videa
    # in ga shranite v datoteko tipa gif, tako da bo frekvenca novega videa enaka frekvenci
    # originalnega video posnetka.
    frames = motionFieldGIF('vaja11/data/simple-video.avi', 153, bSize, searchSize)
    frames[0].save(
        "motionField.gif",
        duration = 153 ,
        loop =0 ,
        save_all = True ,
        optimize = False ,
        append_images = frames [1:] ,

    )
    # 3. naloga
    # Preizkusite delovanje algoritam bločnega ujemanja na realnem videu real-video.avi, ki 
    # je sestavljen iz K = 138 zaporednih sivinskih slik velikosti X x Y = 256 x 144 
    # slikovnih elementov in zapisan v nezgoščeni obliki s frekvenco 25 Hz znotraj zalogovnika
    # AVI. Prilžite programsk kodo in izrise slik (polje vektorjev ter polje vektrojev 
    # premika, superponirano na sliko) za poljubno izbrani primer slik iz tega videa.

    imagePredict = predictImage(frame30, mf, cp, bSize)
    displayImage(imagePredict, "Predicted image")
    displayMotionField(mf,cp, 'Vektorji premika', iImage=imagePredict)
