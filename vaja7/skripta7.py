import numpy as np
import math
from vaja3.skripta3 import loadImage, displayImage

def changeSpatialDomain(iType, iImage, iX, iY, iMode=None, iBgr=0):
    
    Y, X = iImage.shape
    
    if iType == 'enlarge':
        
        if iMode is None:
            oImage = np.zeros((Y+2*iY,X+2*iX), dtype=float)
            oImage[iY:Y+iY,iX:X+iX] = iImage
        
        elif iMode == 'constant':
            oImage = np.ones((Y+2*iY,X+2*iX), dtype=float) * iBgr
            oImage[iY:Y+iY,iX:X+iX] = iImage

        elif iMode == 'extrapolation':
            oImage = np.zeros((Y+2*iY,X+2*iX), dtype=float)
            oImage[iY:Y+iY,iX:X+iX] = iImage
            # top side
            oImage[:iY,iX:iX+X] = np.tile(iImage[:1,:X], (iY,1))
            # bottom side
            oImage[Y+iY:,iX:iX+X] = np.tile(iImage[Y-1:,:X], (iY,1))
            # left side
            oImage[iY:Y+iY,:iX] = np.tile(iImage[:Y,:1].T, (iX,1)).T
            # right side
            oImage[iY:Y+iY,iX+X:] = np.tile(iImage[:Y,X-1:].T, (iX,1)).T 
            # top left corner
            oImage[:iY,:iX] = np.ones((iY,iX)) * iImage[0][0]
            # top right corner
            oImage[:iY,iX+X:] = np.ones((iY,iX)) * iImage[0][X-1]
            # bottom left corner
            oImage[iY+Y:,:iX] = np.ones((iY,iX)) * iImage[Y-1][0]
            # bottom right corner
            oImage[iY+Y:,iX+X:] = np.ones((iY,iX)) * iImage[Y-1][X-1]

        elif iMode == 'reflection':
            numPicsX = math.ceil(iX/X)
            numPicsY = math.ceil(iY/Y)
            tImgSize = (Y*(2*numPicsY+1),X*(2*numPicsX+1))
            tImage = np.zeros(tImgSize, dtype=float)
            """ # top side
            oImage[:iY,iX:iX+X] = np.flipud(iImage[:iY,:X])
            # bottom side
            oImage[Y+iY:,iX:iX+X] = np.flipud(iImage[Y-iY:,:X])
            # left side
            oImage[iY:Y+iY,:iX] = np.fliplr(iImage[:Y,:iX])
            # right side
            oImage[iY:Y+iY,iX+X:] = np.fliplr(iImage[:Y,X-iX:])
            # top left corner
            oImage[:iY,:iX] = np.flip(iImage[:iY,:iX])
            # top right corner
            oImage[:iY,iX+X:] = np.flip(iImage[:iY,X-iX:])
            # bottom left corner
            oImage[iY+Y:,:iX] = np.flip(iImage[Y-iY:,:iX])
            # bottom right corner
            oImage[iY+Y:,iX+X:] = np.flip(iImage[Y-iY:,X-iX:]) """
            # set flipped images
            image1 = np.copy(iImage)
            image2 = np.copy(iImage[:,::-1])
            image3 = np.copy(iImage[::-1,:])
            image4 = np.copy(iImage[::-1,::-1])
            startX, indexY = 0,0
            # determine which image starts in top left corner 
            # set how to flip image 
            if (numPicsX % 2 == 1):
                imageSelectorX = 4
                startX = 1
            elif (numPicsX % 2 == 0):
                imageSelectorX = 8
            if (numPicsY % 2 == 1):
                imageSelectorY = 1
                indexY = 1
            elif (numPicsY % 2 == 0):
                imageSelectorY = 2
            imageSelector = imageSelectorY + imageSelectorX
            image = iImage
            # add pictures to temp image
            for y in range(2*numPicsY+1):
                # set x flipping  
                indexX = startX + (y%2)
                for x in range(2*numPicsX+1):
                    # set which image will be placed to the range
                    if imageSelector == 10:
                        image = image3
                    elif imageSelector == 9:
                        image = image4
                    elif imageSelector == 6:
                        image = image2
                    elif imageSelector == 5:
                        image = image1
                    # add image to the range
                    tImage[Y*y:Y*(y+1),X*x:X*(x+1)] = image
                    # add or subtract to image selector
                    imageSelector += (-1)**(indexX)
                    indexX += 1
                imageSelector += ((-1)**(indexY))*4
                indexY += 1
            # set output image
            oImgSize = (Y+2*iY,X+2*iX)
            oImage = np.zeros(oImgSize, dtype=float)
            tImgExtra = (int((tImgSize[0]-oImgSize[0])/2)), int((tImgSize[1]-oImgSize[1])/2)
            
            oImage = tImage[tImgExtra[0]:oImgSize[0]+tImgExtra[0],tImgExtra[1]:oImgSize[1]+tImgExtra[1]]
              

        elif iMode == 'period':
            oImage = np.zeros((Y+2*iY,X+2*iX), dtype=float)
            oImage[iY:Y+iY,iX:X+iX] = iImage
            """ # top side
            oImage[:iY,iX:iX+X] = iImage[Y-iY:,:X]
            # bottom side
            oImage[Y+iY:,iX:iX+X] = iImage[:iY,:X]
            # left side
            oImage[iY:Y+iY,:iX] = iImage[:Y,X-iX:]
            # right side
            oImage[iY:Y+iY,iX+X:] = iImage[:Y,:iX]
            # top left corner
            oImage[:iY,:iX] = iImage[Y-iY:,X-iX:]
            # top right corner
            oImage[:iY,iX+X:] = iImage[Y-iY:,:iX]
            # bottom left corner
            oImage[iY+Y:,:iX] = iImage[:iY,X-iX:]
            # bottom right corner
            oImage[iY+Y:,iX+X:] = iImage[:iY,:iX] """
            imageIndexX , imageIndexY = iX % X, iY % Y
            for y in range(Y+2*iY):
                for x in range(X+2*iX):
                    oImage[y][x] = iImage[imageIndexY][imageIndexX]
                    imageIndexX += 1
                    if imageIndexX == X:
                        imageIndexX = 0
                imageIndexY += 1
                if imageIndexY == Y:
                    imageIndexY = 0
                    

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


def weightedAverageFilter(iM, iN, iValue):

    oFilter = np.ones((iN, iM))

    x_weight, y_weight = 0,0
    for y in range(iN):
        x_weight = 0
        for x in range(iM):
            val = iValue ** (x_weight + y_weight)
            oFilter[y][x] = val
            x_weight = x_weight + 1 if x < int(iM / 2) else x_weight - 1   
        y_weight = y_weight + 1 if y < int(iN / 2) else y_weight - 1
    
    return oFilter / oFilter.sum()

def sobelAmplitudePhase(g_x, g_y):
    # calculate amplitude response
    oAmplitude = np.sqrt(g_x**2+g_y**2)
    # normalize values and multiply by 255 to get values in range 0..255
    oAmplitude = oAmplitude / oAmplitude.max() * 255
    g_x[g_x==0]=0.00000001

    oPhase = np.arctan(g_y/g_x)
    oPhase = oPhase / np.pi * 255
    print(oPhase)

    return oAmplitude, oPhase

def sharpenImage(iImage, filter, c):
    fImage = spatialFiltering(iType='kernel', iImage=iImage, iFilter=filter)
    oMask = iImage-fImage
    oMaskRange = (oMask - oMask.min()) / (np.abs(oMask.min()) + oMask.max()) * 255
    oImage = iImage + c * oMask
    return oImage, oMaskRange


if __name__ == '__main__':
    orig_size = [256,256]

    image = loadImage(f"./vaja7/data/cameraman-256x256-08bit.raw",orig_size, np.uint8)

    displayImage(image,"Originalna slika")

    # Naloga 1. Doma:
    # Priložite izrise slik, ki jih pridobite s filtriranjem dane slike v točki 2 iz navodil (filtriranje z jedrom: glajenje s podanim jedrom uteženega povprečja velikosti 3x3; statistično filtriranje: prostorska domena 3x3 ter statistična operacije mediane; morfološko filtriranje: erozija s strukturnim elementom velikosti 5x5)
    print("Priložite izrise slik, ki jih pridobite s filtriranjem dane slike v točki 2 iz navodil.")
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

    # Naloga 2. Doma:
    # Napišite funkcijo za izračun koeficientov jedra filtra za uteženo povprečenje:
    print("Kako se imenuje filter, za katerega izberemo iValue=1")
    print("Filter za katerega izberemo iValue=1 se imenuje filter z artimetičnim povprečjem")

    # Naloga 3. Doma:
    # Filtrirajte dano sliko z Sobelivoma operatorjema, ki sta podana v navodilih. Izračunajte tudi amplitudni in fazni odziv dobljenega gradientnega vektorskega polja.
    sobelFilterX = np.array([[-1,0,1], [-2,0,2], [-1,0,1]])
    sobelFilterY = np.array([[-1,-2,-1], [0,0,0], [1,2,1]])
    sobelXImage = spatialFiltering(iType='kernel',iImage=image, iFilter=sobelFilterX)
    sobelXImage = sobelXImage / sobelXImage.max() * 255
    sobelYImage = spatialFiltering(iType='kernel',iImage=image, iFilter=sobelFilterY)
    sobelYImage = sobelYImage / sobelYImage.max() * 255
    displayImage(sobelXImage, "Filtriranje s sobelovim operatorjem v x smeri")
    displayImage(sobelYImage, "Filtriranje s sobelovim operatorjem v y smeri")

    ampImage, phaseImage = sobelAmplitudePhase(sobelXImage,sobelYImage)
    displayImage(ampImage, "Amplitude image")
    displayImage(phaseImage, "Fazni odziv ")
    # Naloga 4. Doma:
    # Izostrite dano sliko s pomočjo t. i. maskiranja neostrih področij, pri čemer za pridobivanje maske uporabite glajenje z Gaussovim povprečenjem (jedro filtra velikosti 3x3 je podano v navodilih), za stopnjo ostrenja pa izbrete c = 2.

    gaussianFilter = np.array([[0.01,0.08,0.01],[0.08,0.64,0.08],[0.01,0.08,0.01]])
    sharpImage, sharpenMask = sharpenImage(image,gaussianFilter,2)
    displayImage(sharpenMask, "Maska neostrih področij")
    displayImage(sharpImage, "Slika izostrena z Gaussovim povprečenjem")

    # Naloga 5. Doma:
    # Dopolnite funkicjo za spremnijanje prostorske domene slike changeSpatialDomain() tako, da bo omogočala več načinov spreminjanja, kjer dodaten vhodni argument iMode predstavlja način spreminjanja prostorske domene, in sicer:
    # - s konstantno sivinsko vrednostjo: 'constant', pri čemer je poljubna konstantna sivinska vrenost podana v vhodnem arugmetnu iBgr.
    # - z ekstrapolacijo sivinskih vrednosti: 'extrapolation'
    # - z zrcaljenjem sivinskih vrenodsti: 'reflection'
    # - s periodičnim ponavljanjem sivinskih vrednosti: 'period'
    # Priložite programsko kodo dopolnjene funkcije changeSpatialdomain() ter izrise dane slike
    # z razširjeno prostorsko domeno, pri čemer uporabite vsakega od zgoraj navedenih načinov razširitve tako, 
    # da sliko v vsako x koordianatno smer razširite za 128 slikovnih elementov, v vsako y koordinatno smer 
    # pa za 384 slikovnih elementov (pri razširitvi s konstantno sivinsko vrednostjo izbreite vrednost 127)

    constantSpatialDomain = changeSpatialDomain(iType='enlarge', iImage=image, iX=128, iY=384, iMode='constant', iBgr=127)
    displayImage(constantSpatialDomain, "Spremenjena prostorska domena s konstantno sivinsko vrednostjo")
    extrapolationSpatialDomain = changeSpatialDomain(iType='enlarge', iImage=image, iX=128, iY=384, iMode='extrapolation')
    displayImage(extrapolationSpatialDomain, "Spremenjena prostorska domena z ekstrapolacijo sivinskih vrednosti")
    reflectionSpatialDomain = changeSpatialDomain(iType='enlarge', iImage=image, iX=128, iY=384, iMode='reflection')
    displayImage(reflectionSpatialDomain, "Spremenjena prostorska domena z zrcaljenjem sivinskih vrednosti")
    periodSpatialDomain = changeSpatialDomain(iType='enlarge', iImage=image, iX=128, iY=384, iMode='period')
    displayImage(periodSpatialDomain, "Spremenjena prostorska domena s periodičnim ponavljanjem sivinskih vrednosti")

    displayImage(image,"image")
    imagetest = image[:,::-1]
    displayImage(imagetest,"testing")
    imagetest = image[::-1,:]
    displayImage(imagetest, "testing")
    imagetest = image[::-1,::-1]
    displayImage(imagetest,"testing")
    # Naloga 6. Doma:
    # Primerjajte rezultate, ki jih pridobite z različnimi vrstami filtriranja nad vhodno sliko, pri čemer vhodni sliki na 
    # različen način (glej vprašanje 5) spremenite prostorsko domeno.
    # Kako vpliva način razširitve prostorske domene slike na rezultate filtriranja? Utemeljite odgovor.
    




