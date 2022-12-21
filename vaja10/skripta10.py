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
        q = 2
        for n in range(N):
            for m in range(M):
                D = np.sqrt((n - n_center) ** 2 + (m - m_center) ** 2)
                oMatrix[n, m] = 1/(1+(D/iD0) ** (2*q))

    if iType[1:] == 'HPF':
        oMatrix = 1 - oMatrix

    return oMatrix


if __name__ == '__main__':
    image = loadImage('vaja10/data/pattern-236x330-08bit.raw', [236,330], np.uint8)
    displayImage(image,"pattern")


    G = computeDFT2(image,'forward')
    gR = computeDFT2(G, 'inverse')

    """ displayImage(gR.real,"reconstructed")



    analyzeDFT2(G,['amplitude', 'center', 'log', 'scale', 'display'], 'Amplitude spectrum')
    analyzeDFT2(G,['phase', 'scale', 'display'], 'Phase spectrum')



    H_16_ILPH = getFilterSpectrum(G, min(G.shape)/4, 'IHPF')

    analyzeDFT2(H_16_ILPH, iOperations=['scale','display'], iTitle='filter spectrum')

    Gf = G * analyzeDFT2(H_16_ILPH, iOperations=['center'])

    gf = computeDFT2(Gf, iDir='inverse')
    analyzeDFT2(gf, iOperations=['amplitude','display'], iTitle='filtered spectrum')


    butter16 = getFilterSpectrum(G, min(G.shape)/16, 'BHPF')
    analyzeDFT2(butter16, iOperations=['scale','display'], iTitle='filter spectrum')

    Gf16 = G * analyzeDFT2(butter16, iOperations=['center'])
    gf16 = computeDFT2(Gf16, iDir='inverse')
    displayImage(gf16.real, 'image')

    butter4 = getFilterSpectrum(G, min(G.shape)/4, 'BHPF')
    analyzeDFT2(butter4, iOperations=['scale','display'], iTitle='filter spectrum')

    Gf4 = G * analyzeDFT2(butter4, iOperations=['center'])
    gf4 = computeDFT2(Gf4, iDir='inverse')
    displayImage(gf4.real, 'image')

    butter3 = getFilterSpectrum(G, min(G.shape)/3, 'BHPF')
    analyzeDFT2(butter3, iOperations=['scale','display'], iTitle='filter spectrum')

    Gf3 = G * analyzeDFT2(butter3, iOperations=['center'])
    gf3 = computeDFT2(Gf3, iDir='inverse')
    displayImage(gf3.real, 'image') """

    BL16 = getFilterSpectrum(G, min(G.shape)/16, 'BLPF')
    analyzeDFT2(BL16, iOperations=['scale','display'], iTitle='filter spectrum')
    BL4 = getFilterSpectrum(G, min(G.shape)/4, 'BLPF')
    analyzeDFT2(BL4, iOperations=['scale','display'], iTitle='filter spectrum')
    BL3 = getFilterSpectrum(G, min(G.shape)/3, 'BLPF')
    analyzeDFT2(BL3, iOperations=['scale','display'], iTitle='filter spectrum')
    BH16 = getFilterSpectrum(G, min(G.shape)/16, 'BHPF')
    analyzeDFT2(BH16, iOperations=['scale','display'], iTitle='filter spectrum')
    BH4 = getFilterSpectrum(G, min(G.shape)/4, 'BHPF')
    analyzeDFT2(BH4, iOperations=['scale','display'], iTitle='filter spectrum')
    BH3 = getFilterSpectrum(G, min(G.shape)/3, 'BHPF')
    analyzeDFT2(BH3, iOperations=['scale','display'], iTitle='filter spectrum')

    GfL16 = G * analyzeDFT2(BL16, iOperations=['center'])
    gfL16 = computeDFT2(GfL16, iDir='inverse')
    analyzeDFT2(gfL16, iOperations=['amplitude','display'], iTitle='filter spectrum')

    GfL4 = G * analyzeDFT2(BL4, iOperations=['center'])
    gfL4 = computeDFT2(GfL4, iDir='inverse')
    analyzeDFT2(gfL4, iOperations=['amplitude','display'], iTitle='filter spectrum')

    GfL3 = G * analyzeDFT2(BL3, iOperations=['center'])
    gfL3 = computeDFT2(GfL3, iDir='inverse')
    analyzeDFT2(gfL3, iOperations=['amplitude','display'], iTitle='filter spectrum')

    GfH16 = G * analyzeDFT2(BH16, iOperations=['center'])
    gfH16 = computeDFT2(GfH16, iDir='inverse')
    analyzeDFT2(gfH16, iOperations=['amplitude','display'], iTitle='filter spectrum')

    GfH4 = G * analyzeDFT2(BH4, iOperations=['center'])
    gfH4 = computeDFT2(GfH4, iDir='inverse')
    analyzeDFT2(gfH4, iOperations=['amplitude','display'], iTitle='filter spectrum')

    GfH3 = G * analyzeDFT2(BH3, iOperations=['center'])
    gfH3 = computeDFT2(GfH3, iDir='inverse')
    analyzeDFT2(gfH3, iOperations=['amplitude','display'], iTitle='filter spectrum')


    pumpkin_image = loadImage('vaja10/data/pumpkin-200x152-08bit.raw', [200,152], np.uint8)
    cameraman_image = loadImage('vaja10/data/cameraman-256x256-08bit.raw', [256,256], np.uint8)
    

    pumpkinDFT = computeDFT2(pumpkin_image, 'forward')
    cameramanDFT = computeDFT2(cameraman_image, 'forward')
    print(f"Povprečna vrednost pattern: {np.average(image)}")

    print(f"Povprečna vrednost pumpkin: {np.average(pumpkin_image)}")

    print(f"Povprečna vrednost cameraman: {np.average(cameraman_image)}")