import numpy as np
import matplotlib.pyplot as plt
from vaja1.main import loadImage, displayImage, saveImage
from vaja2.skripta2 import computeHistogram, displayHistogram

def interpolateImage(iImage, iSize, iOrder):
    iOrder = int(iOrder)

    cols, rows = iSize 

    oImage = np.zeros((rows,cols),dtype=np.uint8) # alternativa np.zeros(iSize[::-1])

    # step: [dx,dy]
    step = [(iImage.shape[1]-1)/(cols-1),(iImage.shape[0]-1)/(rows-1)]
    
    for y in range(rows):
        for x in range(cols):
            # pt = point, koordinata trenutnega pixla (x,y) v originalnem koordinatnem sistemu
            pt = np.array([x,y]) * np.array(step)
           
            # velja za interpolacijo ničtega reda
            if iOrder == 0:
                px = np.round(pt).astype(int)
                s = iImage[px[1],px[0]] #sivinska vrednost 
            
            elif iOrder == 1:
                px = np.floor(pt).astype(int)

                a = abs(pt[0] - (px[0]+1)) * abs(pt[1] - (px[1]+1))
                b = abs(pt[0] - (px[0]+0)) * abs(pt[1] - (px[1]+1))
                c = abs(pt[0] - (px[0]+1)) * abs(pt[1] - (px[1]+0))
                d = abs(pt[0] - (px[0]+0)) * abs(pt[1] - (px[1]+0))
                
                sa = iImage[px[1], px[0]]
                sb = iImage[px[1], min(px[0]+1,iImage.shape[1]-1)]
                sc = iImage[min(px[1]+1,iImage.shape[0]-1), px[0]]
                sd = iImage[min(px[1]+1,iImage.shape[0]-1), min(px[0]+1,iImage.shape[1]-1)]

                s = sa*a + sb*b + sc*c + sd*d 
               
            
            oImage[y,x] = s


    return oImage





def displayImage(iImage, iTitle, iGridX, iGridY):
    plt.figure()
    plt.title(iTitle)
    plt.imshow(iImage, cmap='gray',aspect='equal',extent=(iGridX[0], iGridX[1], iGridY[0], iGridY[1]))


if __name__ == "__main__":
    orig_size = [200,152]
    image = loadImage('./vaja3/data/pumpkin-200x152-08bit.raw',orig_size,np.uint8)

    displayImage(image,"Pumpkin",(0,200),(0,152))

    interpolateZero = interpolateImage(image,2*np.array(orig_size), 0)

    displayImage(interpolateZero, "Pumpkin interpolate zero",(0,400),(0,304))

    interpolateFirst = interpolateImage(image,2*np.array(orig_size), 1)

    displayImage(interpolateFirst, "Pumpkin iterpolate first",(0,400),(0,304))

    # 1. Naloga
    # Določite interpolacijsko sliko kot območje velikosti X x Y = 65 x 50 slikovnih elementov v dani sliki, pri čemer se prvi slikovni element območja nahaja 
    # na lokaciji (x,y) = (75,30) slikovnih elementov v dani sliki (to je v 75-stolpcu in 30-ti vrstici slike).
    # Priložite sliko dobljenega interpolacijskega območja. Priložite tudi sliko histograma interpolacijskega območja ter zapišite minimalno, maksimalno in povprečno 
    # sivinsko vrednost območja.

    image_interpolate = np.zeros((50,65),dtype=int)
    start_y, start_x = 29,74
    for y in range(image_interpolate.shape[0]):
        for x in range(image_interpolate.shape[1]):
            image_interpolate[y][x] = image[start_y+y][start_x+x]
    
    displayImage(image_interpolate, "Interpolacijska slika",(0,65),(0,50))

    h, p, cdf, l = computeHistogram(image_interpolate)

    displayHistogram(h, l, "Histogram interpolacijske slike")
    


    # 2. Naloga
    # Priložite sliko interpoliranega območja velikosti X x Y = 600 x 300 slikovnih elemenotov, pridobljenega z interpolacijo ničtega reda interpolacijskega območja. Priložite tudi
    # sliko histograma interpoliranega območja ter zapišite minimalno, maksimalno in povprečno sivinsko vrednost interpoliranega območja.

    new_size = (600,300)
    image_interpolate_zero = interpolateImage(image_interpolate,new_size,0)

    displayImage(image_interpolate_zero, "Interpolacija ničtega reda",(0,65),(0,50))

    h0, p0, cdf0, l0 = computeHistogram(image_interpolate_zero)

    displayHistogram(h0, l0, "Histogram interpolacije ničtega reda")

    # 3. Naloga
    # Kaj so prednosti in slabosti interpolacije prvega reda? 
    print("Robovi na sliki so bolj gladki, vendar pa je celotna slika bolj zamegljena")
    # Priložite sliko interpoliranega območja velikosti X x Y = 600 x 300 slikovnih elemenotv, pridobljenega z interpolacijo prvega reda interpolacijskega območja. Priložite tudi 
    # sliko histograma interpoliranega območja ter zapišite minimalno, maksimalno in povprečno sivinsko vrednost interpoliranega območja.

    image_interpolate_one = interpolateImage(image_interpolate,new_size,1)

    displayImage(image_interpolate_one, "Interpolacija prvega reda",(0,65),(0,50))
    displayImage(image_interpolate_one, "Interpolacija prvega reda drugič",(0,600),(0,300))

    h1, p1, cdf1, l1 = computeHistogram(image_interpolate_one)

    displayHistogram(h1, l1, "Histogram interpolacije prvega reda")


    # 4. Naloga
    # Kaj dosežemo z interpolacijami višjih redov, npr. z interpolacijo tretjega reda?

    # 5. Naloga
    # Pri prikazovanju slik vzpostavite dimenzijsko sorazmerje med interpolacijskim in interpoliranim območjem (tj. območji morata biti enakih fizičnih dimenzij), in sicer tako, 
    # da spremenite obstoječo funkcijo displayImage z dodatnima vhodnima argumentom iGridX in iGridY, ki predstavljata vektorja položajev slikovnih elementov v x in y smeri, ki
    # omogočata izračun parametra extent v funkciji imshow iz modula matplotlib.pyplot (https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html). Priložite sliki
    # interpoliranih območij, pridobljenih z interpolacijo ničtega oz. prvega reda interpolacijskega območja, ki sta v dimenzijskem sorazmerju z interpolacijskim območjem. Priložite
    # tudi programsko kodo spremenjene funkcije displayImage.

