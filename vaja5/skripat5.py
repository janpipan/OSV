from mpld3 import display
import numpy as np
from vaja1.main import loadImage
from vaja3.skripta3 import displayImage
from vaja2.skripta2 import displayHistogram, computeHistogram


def scaleImage(iImage, iA, iB):
    return iA*np.array(iImage, dtype=float) + iB

def windowImage(iImage, iC, iW):
    #uporabno za medicinske slike
    oImage = np.array(iImage, dtype=float)
    # najprej splošna linearna preslikava
    oImage = 255 / iW * (oImage-(iC - iW/2))
    oImage[iImage < (iC-iW/2)] = 0
    oImage[iImage > (iC+iW/2)] = 255
    return oImage

def sectionalScaleImage(iImage, iS, oS):

    oImage = np.array(iImage, dtype=float)

    for i in range(len(iS)-1):
        sp_meja_vhd = iS[i]
        zg_meja_vhd = iS[i+1]

        sp_meja_izh = oS[i]
        zg_meja_izh = oS[i+1]

        index = np.logical_and(iImage >= sp_meja_vhd, iImage <= zg_meja_vhd)
        oImage[index] = (zg_meja_izh - sp_meja_izh) / (zg_meja_vhd - sp_meja_vhd) * (iImage[index] - sp_meja_vhd) + sp_meja_izh


    return oImage

def gammaImage(iImage, iG):
    return 255**(1-iG)*np.array(iImage, dtype=float)**iG

def thresholdImage(iImage, iT):
    oImage = np.copy(iImage)
    # if value in inputImage is less or equal to threshold value, set value to 0, if it is greater set value to 255 
    oImage[iImage <= iT] = 0
    oImage[iImage > iT] = 255
    return oImage

def threshRange(iImage):
    nBits = int(np.log2(iImage.max())) + 1
    oLevels = np.arange(0, 2**nBits, 1)
    values = np.zeros(len(oLevels))
    for i in range(256):
        values[i] = np.sum(thresholdImage(iImage,i) == 0)

    return values, oLevels
        

def nonLinearSectionalScaleImage(iImage, iS, oS):
    
    oImage = np.array(iImage, dtype=float)

    coefficients = []

    for i in range(len(iS)-2):

        sp_meja_vhd = iS[i]
        zg_meja_vhd = iS[i+2]

        # matrix created from input points 
        # [s_i^2     s_i        1]
        # [s_(i+1)^2 s_(i+1)    1]
        # [s_(i+2)^2 s_(i+2)    1]

        A = np.matrix(f'{iS[i]**2} {iS[i]} 1; {iS[i+1]**2} {iS[i+1]} 1; {iS[i+2]**2} {iS[i+2]} 1')

        # matrix created from output points
        # [q_i] 
        # [q_(i+1)]
        # [q_(i+2)]

        Y = np.matrix(f'{oS[i]}; {oS[i+1]}; {oS[i+2]}')

        # calculate matrix with coefficients X = A^-1*Y
        # result:
        # [A]
        # [B]
        # [C]
        X = np.matmul(np.linalg.inv(A),Y)

        if i == 0 or i == 2 or i == 4:
            coefficients.append(X)

        index = np.logical_and(iImage >= sp_meja_vhd, iImage <= zg_meja_vhd)

        oImage[index] = X.item(0,0)*iImage[index]**2 + X.item(1,0)*iImage[index] + X.item(2,0)
    
    return oImage, coefficients



if __name__ == '__main__':
    orig_size = [512,512]
    image = loadImage(r'./vaja5/data/image-512x512-16bit.raw', orig_size, np.int16)
    #displayImage(image, 'slika')

    sI = scaleImage(image, iA=-0.125,iB=256)
    #displayImage(sI, 'Slika po splosni lin. preslikavi')
    #print(image.min(),image.max(),sI.min(),sI.max())
    
    wI = windowImage(sI, iC=1000, iW=500)
    #displayImage(wI, 'Slika po linearnem oknenju')
    #print(image.min(),image.max(),wI.min(),wI.max())

    s_vhd = [0,85,170,255]
    s_izh = [85,0,255,170]
    ssI = sectionalScaleImage(wI, iS=s_vhd, oS=s_izh)
    #displayImage(ssI, 'Slika po odsekoma lin. preslikavi')

    gI = gammaImage(wI, 0.5)
    #displayImage(gI, 'Slika po gamma preslikavi')

    print("Naloga 1:")
    print("Zapišite najmanjšo in največjo slikovno (sivinsko) vrednost posameznih slik iz točk 1-5 v navodilih.")
    print(f"1. min: {image.min()}, max: {image.max()}")
    print(f"2. min: {sI.min()}, max: {sI.max()}")
    print(f"3. min: {wI.min()}, max: {wI.max()}")
    print(f"4. min: {ssI.min()}, max: {ssI.max()}")
    print(f"5. min: {gI.min()}, max: {gI.max()}")

    print("Naloga 2:")
    print("Napišite funkcijo za upragovanje sivinskih vrednosti. Priložite programsko kodo funkcije thresholdImage() in sliko, pridobljeno z upragovanjem sivinskih vrednosti linearno oknene slike (iz točke 3 navodil) pri vrednosti parametra t=127")
    tI = thresholdImage(wI,127)
    displayImage(tI, 'Slika po upragovanju sivinskih vrednosti')

    print("Naloga 3:")
    print("Priložite sliko, ki prikazuje potek števila slikovnih elementov upragovane slike s sivinsko vrednostjo s_g=0 v odvisnosti od parametra t, pri čemer t spremnijate čez celotno dinamično območje sivinskih vrednosti linearno oknene slike (iz točke 3 navodil) po koraku 1. Kako se imenuje prikazan potek? Priložite tudi pripadajočo programsko kodo.")
    val, l = threshRange(wI)
    displayHistogram(val, l, "Histogram")
    print("Na sliki dobimo porazdelitev sivinskih elementov v vhodni sliki. Če bi prikazan potek normalizirali s številom slikovnih elementov, bi prikazan potek predstavljal komulativno porazdelitev slikovnih elementov.")

    print("Naloga 4:")
    print("Priložite funkcijo za odsekoma nelinearno preslikavo na podlagi kontrolnih točk. Priložite programsko kodo funkcije nonLinearSectionalScaleImage() in sliko, pridobljeno z odsekoma nelinearno preslikavo sivinskih vrednosti dane slike pri kontrolnih točkah s_1=(0,0), s_2=(40,255), s_3=(80,80), s_4=(127,20), s_5=(167,167), s_6=(207,240) in s_7=(255,255). Zapišite tudi koeficiente A_i, B_i, C_i dobljenih kvadratnih funkcij q_i; i = 0, 2, 4")
    s_vhd = [0,40,80,127,167,207,255]
    s_izh = [0,255,80,20,167,240,255]
    nLI, coefficients = nonLinearSectionalScaleImage(wI,s_vhd,s_izh)
    displayImage(nLI, 'Slikea po odsekoma nelinearni preslikavi')
    print(f"i=0: A={coefficients[0].item(0,0)} B={coefficients[0].item(1,0)} C={coefficients[0].item(2,0)} ")
    print(f"i=2: A={coefficients[1].item(0,0)} B={coefficients[1].item(1,0)} C={coefficients[1].item(2,0)}")
    print(f"i=4 A={coefficients[2].item(0,0)} B={coefficients[2].item(1,0)} C={coefficients[2].item(2,0)}")