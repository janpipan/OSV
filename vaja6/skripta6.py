from turtle import ontimer
import numpy as np
import matplotlib.pyplot as plt
from pyproj import transform
from vaja1.main import loadImage
from vaja3.skripta3 import displayImage

def getRadialValue(iXY, iCP):
    
    K = iCP.shape[0]
    oValue = np.zeros(K)

    x_i, y_i = iXY

    for k in range(K):
        x_k, y_k = iCP[k]
        r = np.sqrt((x_i-x_k)**2 + (y_i-y_k)**2)
        if r > 0:
            oValue[k] = -r**2*np.log(r)
    
    return oValue

def getParameters(iType, **kwargs):
    # default vrednosti scale = 1,1 vse ostalo 0
    if iType == 'affine':
        Tk = np.array([[kwargs['scale'][0], 0, 0], [0, kwargs['scale'][1], 0], [0, 0, 1]])
        Tt = np.array([[1, 0, kwargs['trans'][0]], [0, 1, kwargs['trans'][1]], [0, 0, 1]])
        phi = kwargs['rot']*np.pi/180
        cos, sin = np.cos(phi), np.sin(phi)
        Tr = np.array([[cos, -sin, 0],[sin, cos, 0], [0, 0, 1]])
        Tg = np.array([[1, kwargs['shear'][0], 0], [kwargs['shear'][1], 1, 0], [0, 0, 1]])
        # @ matrix multiplication
        oP = Tg @ Tr @ Tt @ Tk
    elif iType == 'radial':
        K = kwargs['orig_pts'].shape[0]
        UU = np.zeros((K,K),dtype=float)
        coef_matrix = np.zeros((K, 2), dtype=float)

        for k in range(K):
            rad_values = getRadialValue(kwargs['orig_pts'][k], kwargs['orig_pts'])
            UU[k, :] = rad_values
        
        UU_inv = np.linalg.inv(UU)
        alphas = UU_inv @ kwargs['mapped_pts'][:, 0]
        betas = UU_inv @ kwargs['mapped_pts'][:,1]

        coef_matrix[:,0] = alphas
        coef_matrix[:,1] = betas

        oP = {'pts': kwargs['orig_pts'], 'coef': coef_matrix}


    return oP

def transformImage(iType, iImage, iDim, iP, iBgr=0, iInterp=0, startPoint='index'):

    Y, X = iImage.shape

    xc, yc = (X - 1) / 2, (Y - 1) / 2

    oImage = np.ones((Y,X), dtype=float) * iBgr

    for y in range(Y):
        for x in range(X):
            if startPoint == 'index':
                pt = np.array([x,y]) * iDim
            elif startPoint == 'center':
                pt = np.array([x-xc, y-yc]) * iDim
            if iType == 'affine':
                pt = np.append(pt, 1)
                pt = iP @ pt
                pt = pt[:2]
            elif iType == 'radial':
                U = getRadialValue(pt, iP['pts'])
                u = U @ iP['coef'][:,0]
                v = U @ iP['coef'][:,1]
                pt = np.array([u,v])
            pt=pt/iDim
            if startPoint == 'center':
                pt[0] += xc
                pt[1] += yc

            if iInterp == 0:
                px = np.round(pt).astype(int)
                if px[0] >= 0 and px[0] < X and px[1] >=0 and px[1] < Y: 
                    s = iImage[px[1], px[0]]
                    oImage[y, x] = s

            elif iInterp == 1:
                px = np.floor(pt).astype(int)
                
                if px[0] >= 0 and px[0] < X and px[1] >=0 and px[1] < Y:
                    a = abs(pt[0] - (px[0]+1)) * abs(pt[1] - (px[1]+1))
                    b = abs(pt[0] - (px[0]+0)) * abs(pt[1] - (px[1]+1))
                    c = abs(pt[0] - (px[0]+1)) * abs(pt[1] - (px[1]+0))
                    d = abs(pt[0] - (px[0]+0)) * abs(pt[1] - (px[1]+0))
                    
                    sa = iImage[px[1], px[0]]
                    sb = iImage[px[1], min(px[0]+1,iImage.shape[1]-1)]
                    sc = iImage[min(px[1]+1,iImage.shape[0]-1), px[0]]
                    sd = iImage[min(px[1]+1,iImage.shape[0]-1), min(px[0]+1,iImage.shape[1]-1)]

                    s = sa*a + sb*b + sc*c + sd*d  
                    oImage[y, x] = s

    return oImage

def displayPoints(iXY, iMarker):
    plt.plot(iXY[:,0], iXY[:,1], iMarker, ms=10, lw=2)


if __name__ == "__main__":
    # Naloga 1. Vaje
    orig_size = [256,512]
    pxDim = [2,1]
    image = loadImage(f"./vaja6/data/lena-256x512-08bit.raw",orig_size, np.uint8)
    #displayImage(image,"original image",iGridX=[0,511],iGridY=[0,511])

    # Naloga 2. Vaje
    #Taffine = getParameters(iType = 'affine', rot=30, scale=[1,1], trans=[0,0], shear=[0,0])
    #print(Taffine)

    #XY = np.array([[0,0], [511,0], [0,511], [511,511]])
    #UV = np.array([[0,0], [511,0], [0,511], [255,255]])
    #P_radial = getParameters(iType='radial',orig_pts=XY, mapped_pts=UV)
    #print(P_radial)

    # Naloga 3. Vaje
    #tI = transformImage(iType='affine', iImage=image, iDim=pxDim, iP=np.linalg.inv(Taffine), iBgr=63)
    #displayImage(tI, 'Afino preslikana slika', iGridX=[0,511], iGridY=[0,511])

    #tI2 = transformImage(iType='radial', iImage=image, iDim=pxDim, iP=P_radial, iBgr=63)
    #displayImage(tI2, 'Afino preslikana slika', iGridX=[0,511], iGridY=[0,511])

    # Naloga 1. Doma
    # Spremenite funkcijo transformImage() tako, da bo za določanje sivinskih vrednosti v preslikani sliki poleg interpolacije ničtega reda omogočla tudi interpolacijo prvega reda.
    # Izvedite afino preslikavo nad sliko grid-256x512-08bit.raw s paramteroma k_y = 0.8 in g_xy = 0.5 (preostali paramanteri naj ne vplivajo na prelikavo), pri čemer za določanje 
    # sivinskih vrednosti enkrat uporabite interpolacijo ničtega reda, drugič pa interpolacijo prvega reda.
    # Priložite programsko kodo spremenjene funkcije transformImage() ter izrisa slik po preslikavi.
    print("Naloga 1.")
    print("Izvedite afino preslikavo nad sliko grid-256x512-08bit.raw s paramteroma k_y = 0.8 in g_xy = 0.5 (preostali paramanteri naj ne vplivajo na prelikavo), pri čemer za določanje sivinskih vrednosti enkrat uporabite interpolacijo ničtega reda, drugič pa interpolacijo prvega reda. Priložite programsko kodo spremenjene funkcije transformImage() ter izrisa slik po preslikavi.")
    image_grid = loadImage(f"./vaja6/data/grid-256x512-08bit.raw", orig_size, np.uint8)
    displayImage(image_grid, "Originalna grid slika", iGridX=[0,511], iGridY=[0,511])

    affine = getParameters(iType='affine', rot=0, scale=[1,0.8], trans=[0,0], shear=[0.5,0])
    transform_grid_inter0 = transformImage(iType='affine',iImage=image_grid, iDim=pxDim, iP=np.linalg.inv(affine), iBgr=63)
    displayImage(transform_grid_inter0,"Afino preslikana grid slika z uporabo interpolacije ničtega reda",iGridX=[0,511], iGridY=[0,511])

    transform_grid_inter1 = transformImage(iType='affine', iImage=image_grid, iDim=pxDim, iP=np.linalg.inv(affine), iBgr=63, iInterp=1)
    displayImage(transform_grid_inter1, "Afino prelikana grid slika z uporabo interpolacije prvega reda",iGridX=[0,511], iGridY=[0,511])

    # Naloga 2. Doma
    # Izvedite vsako od naslednjih afinih preslikav nad sliko lena-256x512-08bit.raw, pri čemer uporabite samo podane parameter (preostali paramteri naj ne vplivajo na prelikavo)
    # ter interpolacijo siviniskih vrednosti prvega reda.
    print("Naloga 2.")
    print("Izvedite vsako od naslednjih afinih preslikav nad sliko lena-256x512-08bit.raw, pri čemer uporabite samo podane parameter (preostali paramteri naj ne vplivajo na prelikavo) ter interpolacijo siviniskih vrednosti prvega reda.")
    # a) k_x = 0.7 k_y=1.4
    affine_lenaA = getParameters(iType='affine', rot=0, scale=[0.7,1.4], trans=[0,0], shear=[0,0])
    transform_lena_inter1_A = transformImage(iType='affine', iImage=image, iDim=pxDim, iP=np.linalg.inv(affine_lenaA), iBgr=63, iInterp=1)
    displayImage(transform_lena_inter1_A, '2. a) k_x=0.7 k_y=1.4',iGridX=[0,511], iGridY=[0,511])
    # b) t_x = 20 mm t_y = -30 mm
    affine_lenaB = getParameters(iType='affine', rot=0, scale=[1,1], trans=[10,-30], shear=[0,0])
    transform_lena_inter1_B = transformImage(iType='affine', iImage=image, iDim=pxDim, iP=np.linalg.inv(affine_lenaB), iBgr=63, iInterp=1)
    displayImage(transform_lena_inter1_B, '2. b) t_x=20 mm t_y=-30 mm',iGridX=[0,511], iGridY=[0,511])
    # c) phi = 30
    affine_lenaC = getParameters(iType='affine', rot=-30, scale=[1,1], trans=[0,0], shear=[0,0])
    transform_lena_inter1_C = transformImage(iType='affine', iImage=image, iDim=pxDim, iP=np.linalg.inv(affine_lenaC), iBgr=63, iInterp=1)
    displayImage(transform_lena_inter1_C, '2. c) phi=30',iGridX=[0,511], iGridY=[0,511])
    # d) g_xy = 0.1 g_yx = 0.5
    affine_lenaD = getParameters(iType='affine', rot=0, scale=[1,1], trans=[0,0], shear=[0.1,0.5])
    transform_lena_inter1_D = transformImage(iType='affine', iImage=image, iDim=pxDim, iP=np.linalg.inv(affine_lenaD), iBgr=63, iInterp=1)
    displayImage(transform_lena_inter1_D, '2. d) g_xy=0.1 g_yx=0.5',iGridX=[0,511], iGridY=[0,511])
    # e) t_x = -10 mm t_y = 20 mm phi = 15
    affine_lenaE = getParameters(iType='affine', rot=15, scale=[1,1], trans=[-5,20], shear=[0,0])
    transform_lena_inter1_E = transformImage(iType='affine', iImage=image, iDim=pxDim, iP=np.linalg.inv(affine_lenaE), iBgr=63, iInterp=1)
    displayImage(transform_lena_inter1_E, '2. e) t_x=-10 mm t_y=20 mm phi=15',iGridX=[0,511], iGridY=[0,511])
    # f) k_x = 0.7 k_y = 0.7 t_x = 30 mm t_y = -20 mm phi = -15
    affine_lenaF = getParameters(iType='affine', rot=-15, scale=[0.7,0.7], trans=[15,-20], shear=[0,0])
    transform_lena_inter1_F = transformImage(iType='affine', iImage=image, iDim=pxDim, iP=np.linalg.inv(affine_lenaF), iBgr=63, iInterp=1)
    displayImage(transform_lena_inter1_F, '2. f) k_x=0.7 k_y=0.7 t_x=30 mm t_y=-20 mm phi=-15',iGridX=[0,511], iGridY=[0,511])

    # Naloga 3. Doma
    # Kako se imenuje preslikava iz vprašanja 2(e) in kako preslikava iz vprašanja 2(f)? Opišite lastnosti teh preslikav.
    print("Naloga 3.")
    print("Kako se imenuje preslikava iz vprašanja 2. e) in kako preslikava iz vprašanja 2. f)? Opišite lastnosti teh preslikav.")
    
    print("Preslikava iz naloga 2. e) se imenuje toga preslikava, pri kateri se izvede rotacija, ter translacija slike. Lastnosti te preslikave so, da: ohranja vzporednost med premicami, ohranja kote med premicami, ohranja razdalje med poljubnimi točkami ")
    print("Preslikava iz naloge 2. f) se imenuje podobnostna preslikava, pri kateri se izvede toga preslikava in izotropno skaliranje slike. Lastnosti te preslikave so, da: ohranja vzporednost med premicami, ohranja kote med premicami, ne ohranja razdalj med poljubnimi točkami\n")
    
    # Naloga 4. Doma
    # Izvedite afini preslikavi iz vprašanja 2(c) in vprašanja 2(d) nad sliko lena-256x512-08bit.raw, pri čemer izhodišče koordinatnega sistema preslikave prestavitev v središče slike (tako da se slika npr. vrti okoli svojega središča)
    # Priložite izrise slike za vsako preslikavo ter programsko kodo, s katero ste dosegli prestavitev koordinatenga sistema preslikave
    print("Naloga 4.")
    print("Izvedite afini preslikavi iz vprašanja 2. c) in vprašanja 2. d) nad sliko lena-256x512-08bit.raw, pri čemer izhodišče koordinatnega sistema prelikave prestavite v središče slike (tako da se slika npr. vrti okoli svojega središča). Priložite izris slik za vsako preslikavo ter programsko kodo, s katero ste dosegli prestavitev koordinatnega sistema preslikave.")

    # c) phi = 30
    affine_lenaC = getParameters(iType='affine', rot=-30, scale=[1,1], trans=[0,0], shear=[0,0])
    transform_lena_inter1_C = transformImage(iType='affine', iImage=image, iDim=pxDim, iP=np.linalg.inv(affine_lenaC), iBgr=63, iInterp=1, startPoint='center')
    displayImage(transform_lena_inter1_C, '2. c) phi=30',iGridX=[0,511], iGridY=[0,511])
    # d) g_xy = 0.1 g_yx = 0.5
    affine_lenaD = getParameters(iType='affine', rot=0, scale=[1,1], trans=[0,0], shear=[0.1,0.5])
    transform_lena_inter1_D = transformImage(iType='affine', iImage=image, iDim=pxDim, iP=np.linalg.inv(affine_lenaD), iBgr=63, iInterp=1, startPoint='center')
    displayImage(transform_lena_inter1_D, '2. d) g_xy=0.1 g_yx=0.5',iGridX=[0,511], iGridY=[0,511])

    # Naloga 5. Doma
    # Izvedite radialno preslikavo z naslednjimi kontrolnimi točkami (x_k, y_k):
    # (x_1,y_1) = (0,0) mm; (x_2,y_2) = (511,0) mm; (x_3,y_3) = (0,511) mm; (x_4,y_4) = (511,511) mm; 
    # (x_5,y_5) = (63,63) mm; (x_6,y_6) = (63,447) mm; (x_7,y_7) = (447,63) mm; (x_8,y_8) = (447,447) mm;
    # ter pripadajočimi preslikanimi kontrolnimi točkami (u_k,v_k):
    # (u_1,v_1) = (0,0) mm; (u_2,v_2) = (511,0) mm; (u_3,v_3) = (0,511) mm; (u_4,v_4) = (511,511) mm; 
    # (u_5,v_5) = (127,95) mm; (u_6,v_6) = (127,415) mm; (u_7,v_7) = (383,95) mm; (u_8,v_8) = (383,415) mm;

    # Na vhodno in preslikano sliko narišite kontrolne točke z križci rdeče barve in preslikane kontrolne točke z krožci modre barve, kar storite z uporabo naslednje funkcije nesporedno po klicu funkcije displayImage():
    # kjer vhodni argument iXY predstavlja matriko točk [x_j, y_j] = (x_j, y_i) (j-ta vrstica matrike predstavlja j-to od skupno J točk), iMarker pa barvo in vrsto izrisa točk (npr. 'rx' za rdeče križce, 'bo' za modre krožce).
    # Da bo izris deloval pravilno je potrebno 'zakomentirati' ukaz plt.show() na koncu funkcije displayImage().

    # Priložite izrise originalne in preslikane slike z vrisanimi kontrolnimi in preslikanimi točkami, in sicer za radialno preslikavo nad sliko grid-256x512-08bit.raw ter za radialno preslikavo nad sliko lena-256x512-08bit.raw

    # Ali glede na položaj točk preslikava deluje pravilno? Obrazložite odgovor.

    print("Naloga 5.")

    print("Izvedite radialno preslikavo z naslednjimi kontrolnimi točkami (x_k, y_k), ter pripadajočimi preslikanimi kontrolnimi točkami (u_k,v_k)")
    print("Priložite izrise originalne in preslikane slike z vrisanimi kontrolnimi in preslikanimi točkami, in sicer za radialno preslikavo nad sliko grid-256x512-08bit.raw ter za radialno preslikavo nad sliko lena-256x512-08bit.raw")

    XY = np.array([[0,0], [511,0], [0,511], [511,511], [63,63], [63,447], [447,63], [447,447]])
    UV = np.array([[0,0], [511,0], [0,511], [511,511], [127,95], [127,415], [383,95], [383,415]])
    #displayImage(image, 'Originalna slika Lene', iGridX=[0,511], iGridY=[0,511], points=True)
    
    radial_parameters = getParameters(iType='radial', orig_pts=XY, mapped_pts=UV)
    radial_lena = transformImage(iType='radial', iImage=image, iDim=pxDim, iP=radial_parameters, iBgr=63, iInterp=1)
    displayImage(radial_lena, 'Lena radialna preslikava', iGridX=[0,511], iGridY=[0,511], points=True)
    displayPoints(UV, "bo")
    displayPoints(XY, "rx")
    
    
    radial_lena = transformImage(iType='radial', iImage=image_grid, iDim=pxDim, iP=radial_parameters, iBgr=63, iInterp=1)
    displayImage(radial_lena, 'Grid radialna preslikava', iGridX=[0,511], iGridY=[0,511], points=True)
    displayPoints(UV, "bo")
    displayPoints(XY, "rx")

    print("Ali glede na položaj točk preslikava deluje pravilno? Obrazložite odgovor.")

    print("Preslikava ne deluje po pričakovanjih, saj deluje obratno. Obratno deluje zato, ker se preslikane kontrolne točke preslikajo v kontrolne točke.")
