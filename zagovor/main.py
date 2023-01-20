import numpy as np
import matplotlib.pyplot as plt
import cv2
from vaja3.skripta3 import loadImage, displayImage

# 1. Naloga

def std_channels(iImage, x_R, y_R, x_G, y_G, x_B, y_B):

    Y,X,Z = iImage.shape
    
    oImage = np.copy(iImage).astype(float)

    for y in range(Y):
        for x in range(X):
            oImage[y,x,0] = (iImage[y,x,0] - x_R) / y_R
            oImage[y,x,1] = (iImage[y,x,1] - x_G) / y_G
            oImage[y,x,2] = (iImage[y,x,2] - x_B) / y_B

    return oImage

if __name__ == '__main__':
    image = cv2.imread(r'./zagovor/data/planina-509x339-08bit.jpeg')
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    displayImage(image, 'Zacetna slika')

    img_std = std_channels(image,100,50,60,30,40,100)

    displayImage(img_std,'Standardizacija intenzitet')


# 2. Naloga

def displayImage(iImage, iTitle, iGridX=None, iGridY=None, points=None):
    plt.figure()
    plt.title(iTitle)
    if iGridX is None or iGridY is None:
        plt.imshow(iImage, cmap='jet',aspect='equal',vmin=0, vmax=255)
    else:
        plt.imshow(iImage, cmap='jet',aspect='equal',extent=[iGridX[0], iGridX[-1], iGridY[0], iGridY[-1]],vmin=0, vmax=255)
    if points == None:
        plt.show()

def image2pieces(X,Y,N):

    oLabelImage = np.zeros((Y,X))

    oPts = []

    for j in range(N):
        x = np.random.choice(X,1)
        y = np.random.choice(Y,1)
        cpt = [x,y]
        if cpt not in oPts:
            oPts.append([x,y])
        # ce je tocka ze v arrayu odstej j in ponovno generiraj tocko
        else:
            j -= 1

    oPts = np.array(oPts)

    for y in range(Y):
        for x in range(X):
            distances = []
            for coord in oPts:
                x_1, y_1 = coord
                distance = np.sqrt(((x_1-x)**2) + ((y_1-y)**2))
                distances.append(distance)
            distance = np.array(distances).min()
            distance_index = distances.index(distance)
            oLabelImage[y,x] = distance_index



            

    return oLabelImage, oPts

if __name__ == '__main__':
    X,Y = 509, 339
    N = 255
    labelImage, pts = image2pieces(X,Y,N)

    displayImage(labelImage,'Label image')



# 3. Naloga


def displayImage(iImage, iTitle, iGridX=None, iGridY=None, points=None):
    plt.figure()
    plt.title(iTitle)
    if iGridX is None or iGridY is None:
        plt.imshow(iImage, cmap='gray',aspect='equal',vmin=0, vmax=255)
    else:
        plt.imshow(iImage, cmap='gray',aspect='equal',extent=[iGridX[0], iGridX[-1], iGridY[0], iGridY[-1]],vmin=0, vmax=255)
    if points == None:
        plt.show()

def img2collage(iImage, N):

    Y,X,Z = iImage.shape

    oLabelImage, pts = image2pieces(X,Y,N)

    oImage = np.zeros(iImage.shape)

    #intensities = np.zeros(pts.shape[0])

    intensities_r = []
    intensities_g = []
    intensities_b = []
    for i in range(pts.shape[0]):
        intensities_r.append([])
        intensities_g.append([])
        intensities_b.append([])


    for y in range(Y):
        for x in range(X):
            intensities_r[int(oLabelImage[y,x])].append(iImage[y,x,0])
            intensities_g[int(oLabelImage[y,x])].append(iImage[y,x,1])
            intensities_b[int(oLabelImage[y,x])].append(iImage[y,x,2])

    for i in range(pts.shape[0]):
        intensities_r[i] = np.mean(intensities_r[i])
        intensities_g[i] = np.mean(intensities_g[i])
        intensities_b[i] = np.mean(intensities_b[i])

    for y in range(Y):
        for x in range(X):
            oImage[y,x,0] = intensities_r[int(oLabelImage[y,x])]
            oImage[y,x,1] = intensities_g[int(oLabelImage[y,x])]
            oImage[y,x,2] = intensities_b[int(oLabelImage[y,x])]

    return oImage, oLabelImage

if __name__ == '__main__':
    img_final_10, label_final_10 = img2collage(img_std, 10)
    displayImage(img_final_10,'Vizualizacija območij (N=10)')
    displayImage(label_final_10,'Prikaz predloge območij (N=10)')


    img_final_255, label_final_255 = img2collage(img_std, 255)
    displayImage(img_final_255,'Vizualizacija območij (N=255)')
    displayImage(label_final_255,'Prikaz predloge območij (N=255)')






