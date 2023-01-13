import numpy as np
import matplotlib.pyplot as plt
from vaja1.main import loadImage
from vaja3.skripta3 import displayImage
from vaja7.skripta7 import changeSpatialDomain


# 1.Naloga

def distancePoint2Line(iL, iP):

    a, b, c = iL[0], -1, iL[1]

    oD = np.abs(a*iP[0]+b*iP[1]+c)/np.sqrt(a**2+b**2)


    return oD


if __name__ == '__main__':
    image = loadImage('zagovor_lab_vaj/data/train-400x240-08bit.raw',[400,240],np.uint8)
    displayImage(image,"")
    iL = [0.22,100]
    print(distancePoint2Line(iL, [399,0]))


# 2.Naloga

def weightedGaussianFilter(iS, iWR, iStdR, iW):
    M,N = iS
    m = int((iS[0] - 1) / 2)
    n = int((iS[1] - 1) / 2)

    k1 = (iStdR[1] - iStdR[0])/(iWR[1]-iWR[0])

    n1 = (iStdR[0] * iWR[1]  - iStdR[1] * iWR[0]) / (iWR[1]-iWR[0])

    sigma = k1 * iW + n1

    oK = np.zeros((N,M))

    for ix, x in enumerate(range(-m,m+1,1)):
        for iy, y in enumerate(range(-n,n+1,1)):
            oK[iy, ix] = np.exp(-((x**2 + y**2)/(2 * sigma**2))) / (2 * np.pi * sigma**2)

    
    oStd = sigma

    # seštevek filtra mora biti skupaj 1
    oK = oK / oK.sum()


    return oK, oStd


if __name__ == '__main__':
    iS = [7,7]
    iWR = [0,10]
    iStdR = [0.1,10]

    print(weightedGaussianFilter(iS, iWR, iStdR, 0))


# 3. Naloga

def imitateMiniature(iImage, iS, iStdR, iL, iD0):
    oImage = np.copy(iImage)

    Y, X = iImage.shape
    d1 = distancePoint2Line(iL, [0, 0])
    d2 = distancePoint2Line(iL, [0, X-1])
    d3 = distancePoint2Line(iL, [Y-1, 0])
    d4 = distancePoint2Line(iL, [Y-1, X-1])

    dmax = np.max([d1,d2,d3,d4])

    M,N = iS
    m = int((iS[0] - 1) / 2)
    n = int((iS[1] - 1) / 2)

    iImage = changeSpatialDomain('enlarge',iImage,m,n,'extrapolation')

    Y, X = iImage.shape
    oImage = np.zeros((Y,X),dtype=float)
    oVal = []

    for y in range(n, Y-n):
        for x in range(m, X-m):
            d = distancePoint2Line(iL, (x-m,y-n))
            if d > iD0:
                gK, std = weightedGaussianFilter(iS, [iD0,dmax],iStdR,d)

                patch = iImage[y-n:y+n+1,x-n:x+n+1]
                oImage[y,x] = (patch*gK).sum()

                oVal.append([d,std])
            else:
                oImage[y,x] = iImage[y,x]



    oImage = changeSpatialDomain('reduce',oImage,m,n)
    oVal = np.array(oVal)
    


    return oImage, oVal



if __name__ == '__main__':
    imitateImage,oVal = imitateMiniature(image,iS, iStdR,iL,25)

    displayImage(imitateImage, "transformed")

    plt.figure()
    plt.plot(oVal[:,0],oVal[:,1],'.')
    plt.xlabel('razdalja')
    plt.ylabel('sigma')