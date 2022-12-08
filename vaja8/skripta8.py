import numpy as np
from vaja7.skripta7 import sobelAmplitudePhase, spatialFiltering
from vaja3.skripta3 import loadImage, displayImage
from vaja5.skripat5 import thresholdImage


def getCenterPoint(iImage, iRadius):

    oAcc = np.zeros((iImage.shape[0], iImage.shape[1]), dtype=np.uint8)
    for i in range(iImage.shape[0]):
        for j in range(iImage.shape[1]):
            if iImage[i, j] > 0:
                for theta in range(360):
                    a = i + iRadius * np.cos(theta)
                    b = j + iRadius * np.sin(theta)
                    if a < iImage.shape[0] and b < iImage.shape[1]:
                        oAcc[int(a), int(b)] += 1

    oCenter = (np.unravel_index(oAcc.argmax(), oAcc.shape))
    return oCenter, oAcc

    #return oCenter, oAcc

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
    displayImage(phaseImage, "Fazni odziv ")
    sobelXImage = (sobelXImage - sobelXImage.min()) / (np.abs(sobelXImage.min()) + sobelXImage.max()) * 255
    sobelYImage = (sobelYImage - sobelYImage.min()) / (np.abs(sobelYImage.min()) + sobelYImage.max()) * 255

    ampImageThreshold = thresholdImage(ampImage, 220)

    displayImage(ampImageThreshold, "Upragovljena slika amplitudnega odziva")

    coords, hough_space = getCenterPoint(ampImageThreshold, 39)
    print(coords)
    displayImage(hough_space, "Hough space accumulator")