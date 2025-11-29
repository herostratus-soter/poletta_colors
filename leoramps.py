import colour
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

libPath = "python-oklch/src"
if libPath not in sys.path:
    sys.path.insert(0, libPath)

from oklch import colors, tools

cfg = {
    "h_factor": 0.8060, # Coeficientes de desplazamiento (Hues/Tono)
    
    # Factores de Luminosidad (Lightness)
    "lw_factor": 0.9,  # Factor para aclarar (White)
    "lb_factor": 0.8,  # Factor para oscurecer (Black)
    
    # Chroma (Saturación)
    "cf_white": -0.8,
    "cf_black": 0.6,
    
    # matices
    "marillo_hex": "#ffee00",
    "morao_hex": "#7700ff",
    
    #dimensión ploteo
    "width": 32,
    "height": 256
}

#-----------------------------------------------------------función para encontrar centroide a partir de una lista de colores

def get_centroid(colors_obj):

    if not colors_obj:
        # Devuelve un gris medio neutro convertido a OKLCH.
        return tools.colors.OKLAB(0.5, 0.0, 0.0).to_OKLCH()

    sum_l = 0.0
    sum_a = 0.0
    sum_b = 0.0

    # sumatoria
    for color in colors_obj:
        # conversion hex - oklab
        oklab_color = color.to_OKLAB()
        sum_l += oklab_color.l
        sum_a += oklab_color.a
        sum_b += oklab_color.b

    N = len(colors_obj)

    # promedio
    avg_l = sum_l / N
    avg_a = sum_a / N
    avg_b = sum_b / N

    # 3. Crear el objeto OKLAB centroide
    oklab_centroid = tools.colors.OKLAB(avg_l, avg_a, avg_b)

    # 4. CONVERSIÓN FINAL CRUCIAL: Convertir de OKLAB a OKLCH.
    gris = oklab_centroid.to_OKLCH()

    return gris

def itp(t, a, b):
    return t* (b - a) + a

def a_itp(t, a, b): #Interpolar angulos alrededor del circulo
    diff = abs(b-a)
    if diff > 180:
        len = a + 360 - b
        len = t * len
        r = a - len
        if r < 0: r = 360 + r
        return r
    else:
        return itp(t, a, b)

def chromatize_endpoints(colors_obj, marillo, morao):
  
    #nuevo centroide identico al anterior
    gris = get_centroid(colors_obj)
    
    #how much to move to destination hue, 0, 1, like if interpolating
    hFactor = cfg["h_factor"]
    #factor relativo de ilumanicon/oscurecimiento para b/n
    lwFactor = cfg["lw_factor"]
    lbFactor = cfg["lb_factor"]
    #Chroma factors
    cfWhite = cfg["cf_white"]
    cfBlack = cfg["cf_black"]
    
    bHue = colors.HEX(morao).to_OKLCH().h
    wHue = colors.HEX(marillo).to_OKLCH().h
    
    bHue = a_itp(hFactor, gris.h, bHue)
    wHue = a_itp(hFactor, gris.h, wHue)
    
    cWhite = tools.chromatize( cfWhite, tools.lighten( lwFactor, colors.OKLCH(gris.l, gris.c, wHue)) )
    cBlack = tools.chromatize( cfBlack, tools.lighten(-lbFactor, colors.OKLCH(gris.l, gris.c, bHue)) )
    
    return cWhite, cBlack
  
def plot(gradient_data, name):
    plt.axis('off')
    plt.imshow(gradient_data)
    plt.savefig(name, bbox_inches='tight', pad_inches=0)
    plt.plot()
    
def gradient_ok(polette, mode, cWhite, cBlack):
  
    gradient_data = np.zeros((height, width * segments, 3))
    
    for g in range(0, segments):
    
        #pick one color
        thisColor = polette[g]
      
        #print(str(palette[1]))
        #print(colour.notation.HEX_to_RGB(str(palette[1]) ) )
      
        halve = int(height/2)
      
        for i in range(0, halve):
            fract = i/halve
            hereColor = tools.interpolate(fract, cWhite, thisColor, mode).to_HEX()
            gradient_data[i, g * width : (g+1) * width, :] = colour.notation.HEX_to_RGB(str(hereColor))
        
        for i in range(0, halve):
            fract = i/halve
            hereColor = tools.interpolate(fract, thisColor, cBlack, mode).to_HEX()
            gradient_data[i + halve, g * width : (g+1) * width, :] = colour.notation.HEX_to_RGB(str(hereColor))
        
    return gradient_data
    
#----------------------------------------------------------------@title Definicion de colores inicial
#Definir colores primarios y secundarios

colors_hex = np.array([
    "#ff624f",
    "#be83fa",
    "#00c8d0",
    "#81c916",
    "#be83fa",
    "#00c8d0",
    "#81c916"
])

colors_obj = []

for col in colors_hex:
  colors_obj.append(colors.HEX(col))

#now do the interpolations (mantener saturacion para que no salga tan desaturado?)
inter_col = []
mode = "shortest"
#mode = "use_OKLAB"

for color in range(len(colors_obj)): # ---- añade a la lista todas las interpolaciones de colores intermedios (no se va a usar)
    color2 = (color+1)%len(colors_obj)
    inter_col.append(tools.interpolate(0.5, colors_obj[color], colors_obj[color2], mode) )

#now fuse
palette = []
for col in colors_obj:
    palette.append(col)
#for col in inter_col:
#palette.append(col.to_HEX())

#tonos
marillo = cfg["marillo_hex"]
morao = cfg["morao_hex"]

cWhite, cBlack = chromatize_endpoints(colors_obj)

width = cfg["width"]
height = cfg["height"]
polette = palette
segments = len(polette)

mode_oklab = "use_OKLAB"
mode_oklch = "shortest"

name_oklab = '01_oklab.png'
name_oklch = '02_oklch.png'

gradient_oklab = gradient_ok(palette, mode_oklab, cWhite, cBlack)
gradient_oklch = gradient_ok(palette, mode_oklch, cWhite, cBlack)

plot(gradient_oklab, name_oklab)
plot(gradient_oklch, name_oklch)

comparative_gradient = np.concatenate((gradient_oklab, gradient_oklch), axis=1)

plot(comparative_gradient, "03_tmp")




