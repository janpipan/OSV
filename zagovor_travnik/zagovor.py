from re import S
import numpy as np
import matplotlib.pyplot as plt
import cv2
from vaja3.skripta3 import loadImage, displayImage

# 1. Naloga

def normalize_image(iImage):

    oImage = np.copy(iImage).astype(float)

    oImage[:,:,:] = oImage / 255

    return oImage

if __name__ == '__main__':
    #size = []
    image1 = cv2.imread(r'./zagovor_travnik/data/travnik-uint8.jpeg')
    image = cv2.cvtColor(image1,cv2.COLOR_BGR2RGB)
    
    #image = loadImage(r'./zagovor/data/', iSize=size, iType=np.uint8)

    displayImage(image, 'Zacetna slika')
    

    img_normalized = normalize_image(image)

    displayImage(img_normalized, 'Normalizirana slika')


# 2. Naloga

def rgb2hsv(iRGB):
    
    Y, X, Z = iRGB.shape

    oHSV = np.zeros((Y,X,Z))

    for y in range(Y):
        for x in range(X):
            r, g, b = iRGB[y,x,0],iRGB[y,x,1],iRGB[y,x,2]

            v = max(r, g, b)
            c = max(r, g, b) - min(r, g, b)
            l = v - (c/2)

            if c == 0:
                h = 0
            elif v == r:
                h = 60 * ((g-b)/c)
            elif v == g:
                h = 60 * (2+(b-r)/c)
            elif v == b:
                h = 60 * (4+(r-g)/c)
            
            s = 0
            if v == 0:
                s = 0
            else:
                s = c/v
            oHSV[y,x,0], oHSV[y,x,1], oHSV[y,x,2] = h, s, v  

   
    return oHSV

if __name__ == '__main__':
    img_hsv = rgb2hsv(img_normalized)

    print(img_normalized.max())

    displayImage(img_hsv, 'HSV image')


# 3. Naloga

if __name__ == '__main__':
    h_slice = np.copy(img_hsv[:,:,0])
    #h_slice = np.where(h_slice < 100, h_slice / 2, h_slice)

    h_slice[h_slice<100] /=   2
    img_hsv_transformed = np.copy(img_hsv)
    img_hsv_transformed[:,:,0] = h_slice
    displayImage(img_hsv, 'HSV transformed')


# 4. Naloga

def hsv2rgb(iHSV):
    
    Y, X, Z = iHSV.shape

    oRGB = np.zeros((Y,X,Z))

    for y in range(Y):
        for x in range(X):
            h, s, v = iHSV[y,x,0], iHSV[y,x,1], iHSV[y,x,2]

            c = v * s
            h1 = h/60
            x1 = c * (1-abs(h1%2 -1))

            rgb1 = [0,0,0]

            if h1 >= 0 and h1 < 1:
                rgb1 = [c,x1,0]
            elif h1 >= 1 and h1 < 2:
                rgb1 = [x1,c,0]
            elif h1 >= 2 and h1 < 3:
                rgb1 = [0,c,x1]
            elif h1 >= 3 and h1 < 4:
                rgb1 = [0,x1,c]
            elif h1 >= 4 and h1 < 5:
                rgb1 = [x1,0,c]
            elif h1 >= 5 and h1 < 6:
                rgb1 = [c,0,x1]
            m = v - c

            rgb = (rgb1[0] + m, rgb1[1] + m, rgb1[2] + m)

            oRGB[y,x,:] = rgb

    return oRGB

if __name__ == '__main__':

    img_hsv_rgb = hsv2rgb(img_hsv)
    displayImage(img_hsv_rgb, 'HSV -> RGB')

    #img_hsv_rgb2 = hsv2rgb(image2)
    #displayImage(img_hsv_rgb2, 'HSV -> RGB')

    img_hsv_transformed_rgb = hsv2rgb(img_hsv_transformed)
    displayImage(img_hsv_transformed_rgb, 'HSV -> RGB transformed')
    