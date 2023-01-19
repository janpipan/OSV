import numpy as np
import matplotlib.pyplot as plt
import cv2
from vaja3.skripta3 import loadImage, displayImage
from vaja7.skripta7 import spatialFiltering, sobelAmplitudePhase
from vaja6.skripta6 import displayPoints


if __name__ == '__main__':
    size = [693,340]
    image = cv2.imread(r'./zagovor_bled/data/bled-lake-decimated-uint8.jpeg')
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    #image = loadImage(r'./zagovor_bled/data/bled-lake-decimated-uint8.jpeg',iSize=size,iType=np.uint8)
    displayImage(image,'Začetna slika')


def get_blue_region(iImage, iThreshold):
    Y, X, _ = iImage.shape
    blueBrightMask = np.zeros((Y,X))
    brightAreaMask = np.zeros((Y,X))

    blueBrightMask[iImage[:,:,2] > iThreshold] = 255

    brightAreaMask[(iImage[:,:,0] > iThreshold) & (iImage[:,:,1] > iThreshold) & (iImage[:,:,2] > iThreshold)] = 255

    oImage = blueBrightMask - brightAreaMask
    return oImage

# 1. Naloga
if __name__ == '__main__':

    image2 = get_blue_region(image,235)
    displayImage(image2,'Začetna slika')


# 2. Naloga

if __name__ == '__main__':

    SE = np.array(
        [
            [0,0,1,0,0],
            [0,1,1,1,0],
            [1,1,1,1,1],
            [0,1,1,1,0],
            [0,0,1,0,0],
        ]
    )

    

    morphImageErosion = spatialFiltering(iType='morpholoigcal', iImage=image2, iFilter=SE, iMorphOp='erosion')
    displayImage(morphImageErosion, "Slika po morfoloski eroziji")

    lake_mask = spatialFiltering(iType='morpholoigcal', iImage=morphImageErosion, iFilter=SE, iMorphOp='dilation')
    displayImage(lake_mask, "Slika po morfoloski dilaciji")

    sobelFilterX = np.array([[-1,0,1], [-2,0,2], [-1,0,1]])
    sobelFilterY = np.array([[-1,-2,-1], [0,0,0], [1,2,1]])
    sobelXImage = spatialFiltering(iType='kernel',iImage=lake_mask, iFilter=sobelFilterX)
    sobelYImage = spatialFiltering(iType='kernel',iImage=lake_mask, iFilter=sobelFilterY)
    lake_edge_mask, _ = sobelAmplitudePhase(sobelXImage,sobelYImage)
    # prikaz na območju 0...255
    sobelXImage = (sobelXImage - sobelXImage.min()) / (np.abs(sobelXImage.min()) + sobelXImage.max()) * 255
    sobelYImage = (sobelYImage - sobelYImage.min()) / (np.abs(sobelYImage.min()) + sobelYImage.max()) * 255

    

    displayImage(lake_edge_mask, "Amplitude image")


# 3. Naloga

def find_edge_coordinates(iImage):

    Y, X = iImage.shape
    oEdges = []

    for y in range(Y):
        for x in range(X):
            if iImage[y,x]:
                oEdges.append([x,y])

    return np.array(oEdges)


if __name__ == '__main__':

    edgeCoordinates = find_edge_coordinates(lake_edge_mask)
    print(edgeCoordinates)

# 4. Naloga

def compute_distances(iImage, iMask = None):

    Y, X = iImage.shape

    edgeCoordinates = find_edge_coordinates(iImage)
    
    oImage = np.zeros((Y,X))

    if iMask is None:
        for y in range(Y):
            for x in range(X):
                distances = []
                for coord in edgeCoordinates:
                    x_1, y_1 = coord
                    if x_1 == x and y_1 == y:
                        continue
                    distance = np.sqrt(((x_1-x)**2) + ((y_1-y)**2))
                    distances.append(distance)
                
                distance = np.array(distances).min()
                oImage[y,x] = distance
    else:
        for y in range(Y):
            for x in range(X):
                if iMask[y,x]:
                    distances = []
                    for coord in edgeCoordinates:
                        x_1, y_1 = coord
                        if x_1 == x and y_1 == y:
                            continue
                        distance = np.sqrt(((x_1-x)**2) + ((y_1-y)**2))
                        distances.append(distance)
                    
                    distance = np.array(distances).min()
                    oImage[y,x] = distance

    

    return oImage


if __name__ == '__main__':
    imageDistances = compute_distances(lake_edge_mask, lake_mask)
    displayImage(imageDistances, 'finalImage')


    Y,X = imageDistances.shape

    maxVal = imageDistances.max()
    max_y, max_x = 0, 0

    for y in range(Y):
            for x in range(X): 
                if imageDistances[y,x] == maxVal:
                    max_y, max_x = y, x
    
    maxCoords = np.array([[max_x,max_y]])
    print(maxCoords)

    y,x =np.where(imageDistances == np.amax(imageDistances))
    maxCoords = np.array([[x,y]])
    print(maxCoords)

    imageNormalized = (imageDistances - imageDistances.min()) / (np.abs(imageDistances.min()) + imageDistances.max()) * 255
    displayImage(imageNormalized, 'finalImage')
    imageFinal = imageNormalized + lake_edge_mask
    displayImage(imageFinal, 'finalImage',points=True)
    displayPoints(maxCoords, "rx")

    print(f'Koordinate: ({max_x}, {max_y}), Oddaljenost: {maxVal}')

