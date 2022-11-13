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





def displayImage(iImage, iTitle, iGridX=None, iGridY=None):
    plt.figure()
    plt.title(iTitle)
    if iGridX is None or iGridY is None:
        plt.imshow(iImage, cmap='gray',aspect='equal',vmin=0, vmax=255)
    else:
        plt.imshow(iImage, cmap='gray',aspect='equal',extent=[iGridX[0], iGridX[-1], iGridY[0], iGridY[-1]],vmin=0, vmax=255)
    plt.show()

""" def getMinMaxAverage(p):
    minimum, maximum, average = -1, 0, 0  
    for i in range(len(p)):
        average += p[i] * i
        if p[i] > 0 and minimum == -1:
            minimum = i
        elif p[i] > 0:
            maximum = i
    return minimum, maximum, average  """

if __name__ == "__main__":
    orig_size = [200,152]
    image = loadImage('./vaja3/data/pumpkin-200x152-08bit.raw',orig_size,np.uint8)

    #displayImage(image,"Pumpkin")

    #interpolateZero = interpolateImage(image,2*np.array(orig_size), 0)

    #displayImage(interpolateZero, "Pumpkin interpolate zero")

    #interpolateFirst = interpolateImage(image,2*np.array(orig_size), 1)

    #displayImage(interpolateFirst, "Pumpkin iterpolate first")

    # 1. Naloga
    # Določite interpolacijsko sliko kot območje velikosti X x Y = 65 x 50 slikovnih elementov v 
    # dani sliki, pri čemer se prvi slikovni element območja nahaja 
    # na lokaciji (x,y) = (75,30) slikovnih elementov v dani sliki (to je v 75-stolpcu in 30-ti 
    # vrstici slike).
    # Priložite sliko dobljenega interpolacijskega območja. Priložite tudi sliko histograma 
    # interpolacijskega območja ter zapišite minimalno, maksimalno in povprečno 
    # sivinsko vrednost območja.

    # slika, ki bo interpolirana
    image_interpolate = np.zeros((50,65),dtype=int)
    # začetne vrednosti v originalni sliki
    start_y, start_x = 29,74
    for y in range(image_interpolate.shape[0]):
        for x in range(image_interpolate.shape[1]):
            image_interpolate[y][x] = image[start_y+y][start_x+x]
    
    displayImage(image_interpolate, "Interpolacijska slika")

    h, p, cdf, l = computeHistogram(image_interpolate)

    displayHistogram(h, l, "Histogram interpolacijske slike")

    saveImage(image_interpolate, './vaja3/data/image.raw',np.uint8)

    print(f"Min: {image_interpolate.min()}, Max: {image_interpolate.max()}, Povprečna: {np.average(image_interpolate)}")


    # 2. Naloga
    # Priložite sliko interpoliranega območja velikosti X x Y = 600 x 300 slikovnih elemenotov, 
    # pridobljenega z interpolacijo ničtega reda interpolacijskega območja. Priložite tudi
    # sliko histograma interpoliranega območja ter zapišite minimalno, maksimalno in povprečno 
    # sivinsko vrednost interpoliranega območja.

    new_size = (600,300)
    image_interpolate_zero = interpolateImage(image_interpolate,new_size,0)

    displayImage(image_interpolate_zero, "Interpolacija ničtega reda")

    h0, p0, cdf0, l0 = computeHistogram(image_interpolate_zero)

    displayHistogram(h0, l0, "Histogram interpolacije ničtega reda")

    print(f"Min: {image_interpolate_zero.min()}, Max: {image_interpolate_zero.max()}, Povprečna: {np.average(image_interpolate_zero)}")
    print("Kaj so prednosti in kaj slabosti interpolacije ničtega reda?")
    print("Prednosti interpolacije ničtega reda so enostavna implementacija, hkrati pa se tudi ne spreminjajo vrednosti. Slabost interpolacije ničtega reda je, da so robovi bolj grobi.")
    # 3. Naloga
    # Kaj so prednosti in slabosti interpolacije prvega reda? 
    
    # Priložite sliko interpoliranega območja velikosti X x Y = 600 x 300 slikovnih elemenotv, 
    # pridobljenega z interpolacijo prvega reda interpolacijskega območja. Priložite tudi 
    # sliko histograma interpoliranega območja ter zapišite minimalno, maksimalno in povprečno 
    # sivinsko vrednost interpoliranega območja.

    image_interpolate_one = interpolateImage(image_interpolate,new_size,1)

    displayImage(image_interpolate_one, "Interpolacija prvega reda")

    h1, p1, cdf1, l1 = computeHistogram(image_interpolate_one)

    displayHistogram(h1, l1, "Histogram interpolacije prvega reda")

    print(f"Min: {image_interpolate_one.min()}, Max: {image_interpolate_one.max()}, Povprečna: {np.average(image_interpolate_one)}")
    print("Kaj so prednosti in kaj slabosti interpolacije prvega reda?")
    print("Robovi na sliki so bolj gladki, vendar pa je celotna slika bolj zamegljena. Bolj zapletena je tudi njena implemntacija.")
    
    # 4. Naloga
    # Kaj dosežemo z interpolacijami višjih redov, npr. z interpolacijo tretjega reda?
    print("Z interpolacijami višjega reda slika postane bolj meglena v primerjavi z interpolacijami nižjega reda, vendar pa tudi robovi postanejo bolj gladki (ne izgledajo tako pixlasto).")

    # 5. Naloga
    # Pri prikazovanju slik vzpostavite dimenzijsko sorazmerje med interpolacijskim in 
    # interpoliranim območjem (tj. območji morata biti enakih fizičnih dimenzij), in sicer tako, 
    # da spremenite obstoječo funkcijo displayImage z dodatnima vhodnima argumentom iGridX in iGridY, 
    # ki predstavljata vektorja položajev slikovnih elementov v x in y smeri, ki
    # omogočata izračun parametra extent v funkciji imshow iz modula matplotlib.pyplot 
    # (https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html). Priložite sliki
    # interpoliranih območij, pridobljenih z interpolacijo ničtega oz. prvega reda interpolacijskega 
    # območja, ki sta v dimenzijskem sorazmerju z interpolacijskim območjem. Priložite
    # tudi programsko kodo spremenjene funkcije displayImage.
    displayImage(image_interpolate, "Slika interpolacijskega območja")
    displayImage(image_interpolate_zero, "Interpolacija ničtega reda", iGridX=(0,image_interpolate.shape[1]), iGridY=(image_interpolate.shape[0],0))
    displayImage(image_interpolate_one, "Interpolacija prvega reda", iGridX=(0,image_interpolate.shape[1]), iGridY=(image_interpolate.shape[0],0))
