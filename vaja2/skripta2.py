from matplotlib import pyplot as plt
import numpy as np
from vaja1.main import displayImage, loadImage, saveImage




def computeHistogram(iImage):
    iImage = iImage.astype(int)
    # histogram
    """ oHist = [0] * 255
    oProb = 0
    oLevels =0
    oCDF = 0
    
    for i in range(len(iImage)):
        for j in range(len(iImage[i])):
           oHist[iImage[i][j]] += 1  """
    # število bitov uporabljenih za zapis intenzitete bita
    nBits = int(np.log2(iImage.max())) + 1

    oLevels = np.arange(0, 2**nBits, 1)

    oHist = np.zeros(len(oLevels))
    


    for y in range(iImage.shape[0]):
        for x in range(iImage.shape[1]):
            oHist[iImage[y][x]] += 1

    oProb = oHist / iImage.size
    
    oCDF = np.zeros_like(oHist, dtype=float)
    for i in range(len(oProb)):
        oCDF[i] = oProb[:i+1].sum() 

    return oHist, oProb, oCDF, oLevels

def equalizeHistogram(iImage):
    h, p, cdf, l = computeHistogram(iImage)
    nBits = int(np.log2(iImage.max())) + 1
    smax = 2**nBits - 1
    # T = np.floor(cdf*smax)
    oImage = np.zeros_like(iImage)

    for y in range(iImage.shape[0]):
        for x in range(iImage.shape[1]):
            s_i = iImage[y][x]
            oImage[y][x] = int(cdf[s_i] * smax)
            #oImage[y][x] = T[iImage[y,x]]

    return oImage

def equalizeHistogramDecimal(iImage):
    h, p, cdf, l = computeHistogram(iImage)
    nBits = int(np.log2(iImage.max())) + 1
    smax = 2**nBits - 1
    # T = np.floor(cdf*smax)
    oImage = np.zeros_like(iImage)

    counterList = list()
    counter = 0

    for y in range(iImage.shape[0]):
        for x in range(iImage.shape[1]):
            s_i = iImage[y][x]
            oImage[y][x] = cdf[s_i] * smax
            if cdf[s_i] * smax in counterList:
                counter += 1
            else:
                counterList.append(cdf[s_i]*smax)
            
    print(counter)
    return oImage

def displayHistogram(iHist, iLevels, iTitle):
    plt.figure()
    plt.title(iTitle)
    plt.bar(iLevels, iHist, width=1, edgecolor='darkred', color='red')
    plt.xlim((iLevels.min(), iLevels.max()))
    plt.ylim((0, 1.05*iHist.max()))
    plt.show()

def computeEntropy(iImage):
    oEntropy = 0
    h, p, cdf, l = computeHistogram(iImage)
    for i in range(len(p)):
        if p[i] != 0.0:
            oEntropy += p[i] * np.log2(p[i])
    return oEntropy * -1

    

def addNoise(iImage, iStd):

    pass

    

if __name__ == '__main__':
    """ image = loadImage("./vaja2/data/valley-1024x683-08bit.raw", (1024,683), np.uint8)
    h, p, cdf ,l = computeHistogram(image)
    displayHistogram(h,l,'Histogram')
    displayHistogram(p,l, 'Prob')
    displayHistogram(cdf, l, 'cdf')
    displayImage(image,'original')
    izravnan = equalizeHistogram(image)
    h1, p1, cdf1, l1 = computeHistogram(izravnan)
    displayHistogram(h1,l1,'Histogram izravnan')
    displayHistogram(p1,l1, 'Prob izravanan')
    displayHistogram(cdf1, l1, 'cdf izravnan')
    displayImage(equalizeHistogram(image), 'izravnan historgram') """
    image = loadImage("./vaja2/data/valley-1024x683-08bit.raw", (1024,683), np.uint8)
    # 1. Vprašanje
    h, p, cdf, l = computeHistogram(image)
    displayImage(image,'Začetna slika')
    displayHistogram(h, l, 'Histogram začetne slike')
    displayHistogram(p, l, 'Normaliziran histogram začetne slike')
    displayHistogram(cdf, l, 'Kumulativna porazdelitev verjetnosti sivinskih vrednosti začetne slike')
    print("Kakšne lastnosti ima histogram dane slike?")
    print("Odg: Sivinske vrednosti posameznih pixlov so v večini razporejene v območju med 120 in 180\n")

    # 2. Vprašanje
    equalizedImage = equalizeHistogram(image)
    h1, p1, cdf1, l1 = computeHistogram(equalizedImage)
    displayImage(equalizedImage, 'Slika z izravnanim histogramom')
    displayHistogram(h1, l1, 'Histogram slike z izravnanim histogramom')
    displayHistogram(p1, l1, 'Normaliziran histogram slike z izravnanim histogramom')
    displayHistogram(cdf1, l1, 'Kumulativna porazdelitev verjetnosti sivinskih vrednosti slike z izravnanim histogramom')
    print("Kakšne lastnosti ima histogram in kakšne kumulativna porazdelitev verjetnosti sivinskih vrednosti slike z izravnanim histogramom?")
    print("Odg: Histogram slike z izravnanim histogramom ima sivinske vrednosti razpršene po celotnem dinamičnem območju. Pri komulativni porazdelitvi verjetnosti sivinskih vrednosti pa lahko vidimo, da vrednosti skoraj linearno naraščajo. \n")


    # 3. Vprašanje
    entropyOriginal = computeEntropy(image)
    print(f"Entropija začetne slike: {entropyOriginal}")

    entropyEqualized = computeEntropy(equalizedImage)
    print(f"Entropija slike z izravnanim histogramom: {entropyEqualized}\n")

    print("Entropija katere slike je večja in zakaj?")
    print("Odg: Večja je entropija prve slike. V teoriji bi morala imeti večjo entropijo slika z izravnanim histogramom, vendar pa imajo pixli lahko le celoštevilsko vrednost, zaradi česar moramo pri izravnavi histograma vrednosti zaokrožiti. Zaradi zaokroževanja slika nosi manj informacije posledično pa je manjša tudi entropija.")







