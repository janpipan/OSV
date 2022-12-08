import numpy as np
from vaja7.skripta7 import sobelAmplitudePhase, spatialFiltering
from vaja3.skripta3 import loadImage, displayImage
from vaja5.skripat5 import thresholdImage


def getCenterPoint(iImage, iRadius):

    oAcc = np.zeros((iImage.shape[0], iImage.shape[1]), dtype=np.uint8)
    max = 0
    oCenter = (0,0)
    for i in range(iImage.shape[0]):
        for j in range(iImage.shape[1]):
            if iImage[i, j]:
                for phi in range(360):
                    x0 = int(np.round(j + iRadius * np.cos(phi)))
                    y0 = int(np.round(i + iRadius * np.sin(phi)))
                    if x0 < iImage.shape[1] and x0 >= 0 and y0 >= 0 and y0 < iImage.shape[0]:
                        oAcc[y0, x0] += 1
                        if oAcc[y0, x0] > max:
                            max = oAcc[y0,x0]
                            oCenter = (x0,y0)

    return oCenter, oAcc


if __name__ == "__main__":
    orig_size = [160,160]
    image = loadImage(f"./vaja8/circles-160x160-08bit.raw",orig_size, np.uint8)

    displayImage(image,"Originalna slika")
    sobelFilterX = np.array([[-1,0,1], [-2,0,2], [-1,0,1]])
    sobelFilterY = np.array([[-1,-2,-1], [0,0,0], [1,2,1]])
    sobelXImage = spatialFiltering(iType='kernel',iImage=image, iFilter=sobelFilterX)
    sobelYImage = spatialFiltering(iType='kernel',iImage=image, iFilter=sobelFilterY)
    ampImage, phaseImage = sobelAmplitudePhase(sobelXImage,sobelYImage)
    displayImage(ampImage, "Amplitudni odziv ")

    ampImageThreshold = thresholdImage(ampImage, 220)

    displayImage(ampImageThreshold, "Upragovljena slika amplitudnega odziva")

    center, acc = getCenterPoint(ampImageThreshold, 39)
    print('(x,y) =', center)
    displayImage(acc, "Akumulator Houghove preslikave")