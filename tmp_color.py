import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import colorsys
import numpy as np
import math
from scipy.interpolate import interp1d
from okhsl import okhsl_to_srgb, srgb_to_okhsl


#---------------------------------------------------------------------------------------


def ryb_to_ok_hue(h_ryb): #hace la interpolacion ryb

    h_ryb = h_ryb % 360
    
    #puntos para interpolar
    ok_points = np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360])
    #ryb_points = np.array([0, 45, 95, 135, 170, 210, 230, 245, 260, 275, 300, 330, 360]) #viejo ajuste
    #ryb_points = np.array([-20, 0, 45, 90, 135, 195, 210, 225, 240, 255, 280, 320, 340]) #nuevo ajueste no sirve
    ryb_points = np.array([0, 20, 65, 110, 155, 205, 223, 240, 260, 280, 300, 340, 360]) #nuevo ajueste
    #ryb_points = np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360]) #normal sin alterar
    
    #Interpolación lineal simple
    h_ok = np.interp(h_ryb, ryb_points, ok_points)
    
    return h_ok #convierte un color RYB a "OKHUE" (el ángulo correcto dentro del espacio oklab para el hue)


def lista_okhsl_to_rgb(lista_okhsl):
    lista_rgb = []
    for i in lista_okhsl:
        r,g,b = okhsl_to_srgb(i) #rgb rango de 0-1
        lista_rgb.append((r,g,b))
    return lista_rgb #convierte una lista con colores okhsl (h=360,s=1,l=1) a rgb
  

def paleta_okhsl(lista_colores, division, s_list, l, h_list, desviacion): #hace una paleta individual
  
    rango = 360
    angulo = rango/division
    
    for i in range(len(h_list)):
        
        posicion = h_list[i]
        h_ryb=((angulo*posicion)+desviacion)%rango #angulo de geometría poligonal
        h_ok = ryb_to_ok_hue(h_ryb)
        s = s_list[i]
        
        print("paleta_okhsl: ",h_ok," ",s," ",l)
  
        lista_colores.append([h_ok, s, l])

    return lista_colores#funcion que crea las paletas


def plot_paleta(colores, directorio, prefijo, cols, indice):
    rows = math.ceil(len(colores) / cols)
    
    # 1. Crear Figura
    fig, ax = plt.subplots(figsize=(cols * 1.5, rows * 1.5))
    
    # 2. Dibujar los rectángulos de color
    for i, color in enumerate(colores):
        fila = i // cols
        col = i % cols
        ax.add_patch(plt.Rectangle((col, rows - 1 - fila), 1, 1, color=color))
    
    # Ajustes visuales básicos
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.axis('off')
    
    # --- OUTPUT 1: PNG LIMPIO (Guardar antes de escribir texto) ---
    plt.savefig(
        f"{directorio}{prefijo}{indice:03d}_paleta.png", 
        bbox_inches='tight', 
        pad_inches=0,
        dpi=100 # dpi bajo para que sea ligero
    )
    
    # 3. Calcular y escribir textos encima de lo que ya existe
    for i, color in enumerate(colores):
        fila = i // cols
        col = i % cols
        
        # Cálculo de contraste (Blanco/Negro)
        rgb = mcolors.to_rgb(color)
        luminancia = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        text_color = 'black' if luminancia > 0.5 else 'white'
        
        # Escribir Hex
        ax.text(
            col + 0.5, 
            rows - 1 - fila + 0.5, 
            mcolors.to_hex(color).upper(), 
            color=text_color, 
            ha='center', va='center', fontsize=10
        )
        
    # --- OUTPUT 2: PDF CON TEXTOS (Vectorial) ---
    plt.savefig(
        f"{directorio}{prefijo}{indice:03d}_paleta.pdf", 
        bbox_inches='tight', 
        pad_inches=0
    )
    
    plt.close()#plotea la paleta, nada mas
  


#----------------------------paleta------------------------


start = 0
stop = 1 #12 
rango = 1 #20


for elemento in range(start, stop, 1):
  
    colores_okhsl = []
    h_division_equipo = 8
    desviacion_global = 85
    
   
    colores_okhsl = paleta_okhsl(
        lista_colores = colores_okhsl,
        division = h_division_equipo,
        s_list = [90,90,90,90,90,90,90,90],
        l = 67,
        h_list = [1,2,3,4,5,6,7,8],
        desviacion = desviacion_global
    )
    
    """
    
    h_division_maps = 18
    compl = h_division_maps/2

    angulo = 360/rango
    angulo_total = angulo*stop
    correccion = angulo_total/2
    desviacion = (angulo*elemento) - correccion + desviacion_global
    
    h_map = [16,16,2,2]
    s_map = [40,65,65,40]


    colores_okhsl = paleta_okhsl(
        lista_colores = colores_okhsl,
        division = h_division_maps,
        s_list = s_map,
        l = 30,
        h_list = h_map,
        desviacion = desviacion
    )
    
    colores_okhsl = paleta_okhsl(
        lista_colores = colores_okhsl,
        division = h_division_maps,
        s_list = s_map,
        l = 50,
        h_list = h_map,
        desviacion = desviacion
    )
    
    colores_okhsl = paleta_okhsl(
        lista_colores = colores_okhsl,
        division = h_division_maps,
        s_list = s_map,
        l = 70,
        h_list = h_map,
        desviacion = desviacion
    )
    
    colores_okhsl = paleta_okhsl(
        lista_colores = colores_okhsl,
        division = h_division_maps,
        s_list = s_map,
        l = 90,
        h_list = h_map,
        desviacion = desviacion
    )
    

    colores_okhsl = paleta_okhsl(
        lista_colores = colores_okhsl,
        division = h_division_maps,
        s_list = [86,86],
        l =65,
        h_list = [compl,compl],
        desviacion = desviacion
    )
    
    colores_okhsl = paleta_okhsl(
        lista_colores = colores_okhsl,
        division = h_division_maps,
        s_list = [86,86],
        l = 80,
        h_list = [compl,compl],
        desviacion = desviacion
    )
    
    """
    
    colores_rgb = lista_okhsl_to_rgb(colores_okhsl)
    
    plot_paleta(
        colores = colores_rgb,
        directorio = "/home/acid-factory-usr/Imágenes/tmp_palette/",
        prefijo = f"tmp_equipos{desviacion_global}_",
        cols = 4,
        indice = elemento+1
        )





#---------------------------------Esto solo muestra los gamuts de HSL con corrección RYB


def plot_okhsl_wheel(L=50, size=800, rotate_deg=0, show_edges=False, bgcolor=(1,1,1)):

    N = size
    # coordenadas cartesianas centradas en 0: x,y ∈ [-1,1]
    xs = np.linspace(-1, 1, N)
    ys = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(xs, ys)

    R = np.sqrt(X**2 + Y**2)            # 0..√2, usaremos R<=1
    TH = np.arctan2(Y, X)              # -π .. π

    # Hue en grados 0..360: mapear -π..π -> 0..360, y aplicar rotación si quieres
    H_deg = (np.degrees(TH) + 360 + rotate_deg) % 360
    
    
    H_ok = ryb_to_ok_hue(H_deg)
    
    

    # Saturation: normalizamos radio a [0,1]
    S = np.clip(R, 0.0, 1.0) * 100.0   # en % para okhsl func
    L_arr = np.full_like(S, L)         # luminancia constante

    # Preparamos la imagen RGBA
    img = np.zeros((N, N, 3), dtype=float)
    mask = R <= 1.0

    # Procesar solo píxeles dentro del círculo (vectorizado parcialmente)
    idxs = np.nonzero(mask)
    H_vals = H_ok[idxs]
    S_vals = S[idxs]
    L_vals = L_arr[idxs]

    # Convierte punto por punto (la función okhsl_to_srgb es Python puro; loop razonable)
    # Si quieres, se puede paralelizar o optimizar más adelante.
    rgb_flat = np.zeros((H_vals.size, 3), dtype=float)
    for i, (h, s, l) in enumerate(zip(H_vals, S_vals, L_vals)):
        rgb_flat[i, :] = np.clip(okhsl_to_srgb([float(h), float(s), float(l)]), 0.0, 1.0)

    # Volver a reventar en la imagen
    img[idxs] = rgb_flat

    # Fondo
    bg = np.array(bgcolor, dtype=float)
    img_out = np.zeros((N, N, 3), dtype=float)
    img_out[:, :] = bg
    img_out[mask] = img[mask]

    # Mostrar con imshow en ejes cartesianos — esto preserva orientación y grados
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(img_out, origin='lower', extent=[-1, 1, -1, 1], aspect='equal')
    ax.set_xlim(-1,1); ax.set_ylim(-1,1)
    ax.set_xticks([]); ax.set_yticks([])
    if show_edges:
        circle = plt.Circle((0,0), 1.0, transform=ax.transData, fill=False, color='k', linewidth=0.6)
        ax.add_patch(circle)
    ax.set_title(f"OKHSL wheel — L={L}%, rotate={rotate_deg}°")
    plt.tight_layout()
   

if __name__ == "__main__":
    plot_okhsl_wheel(L=76, size=700, rotate_deg=0, show_edges=True, bgcolor=(0.95,0.95,0.95))
    plot_okhsl_wheel(L=85, size=700, rotate_deg=0, show_edges=True, bgcolor=(0.95,0.95,0.95))
    plot_okhsl_wheel(L=35, size=700, rotate_deg=0, show_edges=True, bgcolor=(0.95,0.95,0.95))


#plt.show()



