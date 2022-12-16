import numpy as np
import matplotlib.pyplot as plt
from vaja1.main import loadImage
from vaja3.skripta3 import displayImage


def computeDFT2(iMatrix, iDir="forward"):


    N, M = iMatrix.shape
    n = np.arange(N).reshape(1, -1)
    m = np.arange(M).reshape(1, -1)

    WN = 1 / np.sqrt(N) * np.exp(-1j * 2 * np.pi / N) ** (n.T @ n)
    WM = 1 / np.sqrt(M) * np.exp(-1j * 2 * np.pi / M) ** (m.T @ m)
    
    if iDir == 'inverse':
        WN = np.conjugate(WN)
        WM = np.conjugate(WM)
    
    oMatrix = WN @ iMatrix @ WM

    return oMatrix


def analyzeDFT2(iMatrix, iOperations, iTitle=""):

    oMatrix = np.array(iMatrix)

    for operation in iOperations:
        if operation == 'amplitude':
            oMatrix = np.abs(oMatrix)
        elif operation == 'phase':
            oMatrix = np.unwrap(np.angle(oMatrix))
        elif operation == 'ln':
            oMatrix = np.log(oMatrix + 1e-10) # pristejemo da se izognemo napakam 
        elif operation == 'log':
            oMatrix = np.log10(oMatrix + 1e-10)
        elif operation == 'scale':
            oMatrix -= oMatrix.min() 
            oMatrix /= oMatrix.max()
            oMatrix *= 255
            oMatrix = oMatrix.astype(np.uint8)
        elif operation == 'center':
            N, M = oMatrix.shape
            center = np.array([(N - 1) / 2, (M - 1) / 2]).astype(int)
            A = oMatrix[: center[0], : center[1]]
            B = oMatrix[: center[0], center[1] :]
            C = oMatrix[center[0] :, : center[1]]
            D = oMatrix[center[0] :, center[1] :]
            upper = np.hstack((D, C))
            lower = np.hstack((B, A))
            oMatrix = np.vstack((upper, lower))
        elif operation == 'display':
            plt.figure()
            plt.imshow(oMatrix, aspect='equal',cmap='gray')
            plt.title(iTitle)
        else:
            raise NotImplementedError


    return oMatrix


def getFilterSpectrum(iMatrix, iD0, iType):

    oMatrix = np.zeros_like(iMatrix, dtype=float)
    N, M = iMatrix.shape

    n_center = (N - 1) / 2
    m_center = (M - 1) / 2

    if iType[0] == 'I':
        for n in range(N):
            for m in range(M):
                D = np.sqrt((n - n_center) ** 2 + (m - m_center) ** 2)
                if D <= iD0:
                    oMatrix[n, m] = 1

    
    elif iType[0] == 'B':
        pass

    if iType[1:] == 'HPF':
        oMatrix = 1 - oMatrix

    return oMatrix


if __name__ == '__main__':
    image = loadImage('vaja10/data/pattern-236x330-08bit.raw', [236,330], np.uint8)
    displayImage(image,"pattern")


    G = computeDFT2(image,'forward')
    gR = computeDFT2(G, 'inverse')

    displayImage(gR.real,"reconstructed")



    analyzeDFT2(G,['amplitude', 'center', 'log', 'scale', 'display'], 'Amplitude spectrum')
    analyzeDFT2(G,['phase', 'scale', 'display'], 'Phase spectrum')



    H_16_ILPH = getFilterSpectrum(G, min(G.shape)/4, 'IHPF')

    analyzeDFT2(H_16_ILPH, iOperations=['scale','display'], iTitle='filter spectrum')

    Gf = G * analyzeDFT2(H_16_ILPH, iOperations=['center'])

    gf = computeDFT2(Gf, iDir='inverse')
    analyzeDFT2(gf, iOperations=['amplitude','display'], iTitle='filtered spectrum')
