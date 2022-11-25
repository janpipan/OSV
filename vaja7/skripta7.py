import numpy as np
from vaja3.skripta3 import loadImage, displayImage

def changeSpatialDomain(iType, iImage, iX, iY, iMode=None, iBgr=0):
    
    Y, X = iImage.shape
    
    if iType == 'enlarge':
        
        if iMode is None:
            oImage = np.zeros((Y+2*iY,X+2*iX), dtype=float)
            oImage[iY:Y+iY,iX:X+iX] = iImage

    elif iType == 'reduce':

        oImage = np.copy(iImage[iY:Y-iY,iX:X-iX])


    return oImage


def spatialFiltering(iType, iImage, iFilter, iStatFunc=None, iMorphOp=None):

    # N = rows, M = cols    
    N, M = iFilter.shape

    n = int((N - 1) / 2)
    m = int((M - 1) / 2)

    iImage = changeSpatialDomain(iType='enlarge', iImage=iImage, iX=m, iY=n)

    Y, X = iImage.shape
    oImage = np.zeros((Y, X), dtype=float)

    for y in range(n, Y-n):
        for x in range(m, X-m):
            patch = iImage[y-n:y+n+1,x-n:x+n+1]
            if iType == 'kernel':
                oImage[y,x] = (patch * iFilter).sum() 
            elif iType == 'statistical':
                oImage[y,x] = iStatFunc(patch)
            elif iType == 'morpholoigcal':
                R = patch[iFilter != 0]
                if iMorphOp == 'erosion':
                    oImage[y,x] = R.min()
                elif iMorphOp == 'dilation':
                    oImage[y,x] = R.max()

    oImage = changeSpatialDomain(iType='reduce', iImage=oImage, iX=m, iY=n)

    return oImage












if __name__ == '__main__':
    orig_size = [256,256]

    image = loadImage(f"./vaja7/data/cameraman-256x256-08bit.raw",orig_size, np.uint8)

    displayImage(image,"Camera Man")

    K = 1/16*np.array([[1,2,1],[2,4,2],[1,2,1]])
    SE = np.array(
        [
            [0,0,1,0,0],
            [0,1,1,1,0],
            [1,1,1,1,1],
            [0,1,1,1,0],
            [0,0,1,0,0],
        ]
    )

    kI = spatialFiltering(iType='kernel', iImage=image, iFilter=K)
    displayImage(kI, "Slika po filtriranju z jedrom")

    sI = spatialFiltering(iType='statistical', iImage=image, iFilter=np.zeros((3,3)), iStatFunc=np.median)
    displayImage(sI, "Slika po statisticnem filtriranju")

    mI = spatialFiltering(iType='morpholoigcal', iImage=image, iFilter=SE, iMorphOp='erosion')
    displayImage(mI, "Slika po morfoloskem filtriranju")


