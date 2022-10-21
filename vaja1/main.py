import matplotlib.pyplot as plt
import numpy as np


"""
image = open(raw_img, 'rb')
buffer = image.read()
image.close()

s1 = np.ndarray((512,410), dtype=np.uint8, buffer=buffer, order='F')

plt.figure()
plt.imshow(s1, cmap='gray') """

# Naloga 2
def loadImage(iPath, iSize, iType):
    fid = open(iPath, 'rb')
    buffer = fid.read()
    fid.close()

    len_buffer = len(np.frombuffer(buffer=buffer, dtype=iType))

    if len_buffer != np.prod(iSize):
        oShape = (iSize[1], iSize[0], 3)
    else:
        oShape = (iSize[1], iSize[0])
    oImage = np.ndarray(oShape, dtype=iType, buffer=buffer, order='F')
    return oImage


def displayImage(iImage, iTitle):
    plt.figure()
    plt.title(iTitle)
    plt.imshow(iImage, cmap='gray',aspect='equal')



# funkcija ki piše v raw format
def saveImage(iImage, iPath, iType):
    oFile = open(iPath, 'wb')
    oFile.write(iImage.tobytes(order="F"))
    oFile.close() 
    
if __name__ == '__main__':
    picture = plt.imread("./vaja1/data/lena-color.png")

    # Naloga 1
    #print(picture.shape)

    #plt.figure()

    #plt.imshow(picture)

    #plt.imsave('./vaja1/data/lena.jpeg', picture)


    raw_img = './vaja1/data/lena-gray-410x512-08bit.raw'

    lena_gray= loadImage(raw_img, (410,512), np.uint8)

    displayImage(lena_gray, 'Lena gray')

    saveImage(lena_gray,'./vaja1/data/lena-gray.raw', np.uint8)



    #plt.figure()
    #plt.imshow(lena_grey, cmap='gray')

    lena_color_path = './vaja1/data/lena-color-512x410-08bit.raw'

    lena_color = loadImage(lena_color_path, (512,410), np.uint8)

    displayImage(lena_color, 'Lena color')

    

    lena_gray_raw_path = './vaja1/data/lena-gray.raw'

    lena_gray_raw = loadImage(lena_gray_raw_path, (410,512), np.uint8)

    displayImage(lena_gray_raw,"lena gray raw")

