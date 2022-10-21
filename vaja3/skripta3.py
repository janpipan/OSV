import numpy as np
import matplotlib.pyplot as plt
from vaja1.main import loadImage, displayImage

def interpolateImage(iImage, iSize, iOrder):
    iOrder = int(iOrder)

    cols, rows = iSize 

    oImage = np.zeros((rows,cols)) # alternativa np.zeros(iSize[::-1])

    # step: [dx,dy]
    step = [(iImage.shape[1]-1)/(cols-1),(iImage.shape[0]-1)/(rows-1)]
    
    for y in range(rows):
        for x in range(cols):
            # pt = point, koordinata trenutnega pixla (x,y) v originalnem koordinatnem sistemu
            pt = np.array([x,y]) * np.array(step)

            # velja za interpolacijo prvega reda
            if iOrder == 0:
                px = np.round(pt).astype(int)
                s = iImage[px[1],px[0]] #sivinska vrednost
            
            elif iOrder == 1:
                px = np.floor(pt).astype(int)

                a = abs(pt[0]-(px[0]+1))*abs(pt[1]-px[1])
                b = abs(pt[0]-px[0])*abs(pt[1]-px[1])
                c = abs(pt[0]-(px[0]+1))*abs(pt[1]-(px[1]+1))
                d = abs(pt[0]-px[0])*abs(pt[1]-(px[1]+1))

                sa = iImage[px[1], min(px[0]+1,iImage.shape[1]-1)]
                sb = iImage[min(px[1]+1,iImage.shape[0]-1), min(px[0]+1,iImage.shape[1]-1)]
                sc = iImage[px[1], px[0]]
                sd = iImage[min(px[1]+1,iImage.shape[0]-1), px[0]]

                s = sa*a + sb*b + sc*c + sd*d
            oImage[y,x] = s


    return oImage


if __name__ == "__main__":
    orig_size = [200,152]
    image = loadImage('./vaja3/data/pumpkin-200x152-08bit.raw',orig_size,np.uint8)

    displayImage(image,"Pumpkin")

    interpolateZero = interpolateImage(image,2*np.array(orig_size), 0)

    displayImage(interpolateZero, "Pumpkin interpolate zero")

    interpolateFirst = interpolateImage(image,2*np.array(orig_size), 1)

    displayImage(interpolateFirst, "Pumpkin iterpolate first")