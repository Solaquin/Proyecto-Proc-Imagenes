import cv2
import numpy as np
import os
import pandas as pd
import pywt
from skimage.measure import label, regionprops
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler



# -------------------- Preprocesamiento y características --------------------

def preprocesar(imagen):
    if len(imagen.shape) == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    imagen = cv2.equalizeHist(imagen)
    imagen = cv2.GaussianBlur(imagen, (5, 5), 0)
    return imagen / 255.0

def binarizar(imagen):
    _, binaria = cv2.threshold((imagen * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binaria

def extraer_region(binaria):
    etiqueta = label(binaria)
    props = regionprops(etiqueta)
    if len(props) > 0:
        return props[0]
    return None

def clasificar_forma(region):
    # Relación de aspecto
    bbox = region.bbox
    alto = bbox[2] - bbox[0]
    ancho = bbox[3] - bbox[1]
    aspecto = ancho / alto if alto != 0 else 0

    # Circularidad: 4π * área / perímetro^2
    perimetro = region.perimeter if region.perimeter != 0 else 1
    circularidad = 4 * np.pi * region.area / (perimetro ** 2)

    if aspecto > 2:
        return "Alargada"
    elif circularidad > 0.85:
        return "Redonda"
    elif 0.6 < circularidad <= 0.85:
        return "Ovalada"
    else:
        return "Irregular"


def calcular_orientacion(imagen):
    gx = cv2.Sobel(imagen, cv2.CV_64F, 1, 0, ksize=5)
    gy = cv2.Sobel(imagen, cv2.CV_64F, 0, 1, ksize=5)
    orientacion = np.arctan2(gy, gx) * (180 / np.pi)
    return np.mean(orientacion)

def aplicar_gabor(imagen, n=8):
    respuestas = []
    for i in range(n):
        theta = i * np.pi / n
        kernel = cv2.getGaborKernel((21, 21), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        filtrada = cv2.filter2D(imagen, cv2.CV_8UC3, kernel)
        respuestas.append(filtrada)
    return np.mean(respuestas)

def aplicar_wavelet(imagen):
    coeffs = pywt.wavedec2(imagen, 'db1', level=2)
    cA2, (cH2, cV2, cD2), _ = coeffs
    energia = np.sum(np.square(cA2)) + np.sum(np.square(cH2)) + np.sum(np.square(cV2)) + np.sum(np.square(cD2))
    return energia

# NUEVA FUNCIÓN: extraer características de frecuencia con Transformada de Fourier
def aplicar_fourier(imagen):
    f = np.fft.fft2(imagen)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift)
    energia_frecuencia = np.mean(magnitude_spectrum)
    return energia_frecuencia

# ACTUALIZACIÓN: Añadir Fourier al vector de características
def extraer_caracteristicas_fruta(imagen):
    imagen_pre = preprocesar(imagen)
    binaria = binarizar(imagen_pre)
    region = extraer_region(binaria)
    orient = calcular_orientacion(imagen_pre)
    gabor = aplicar_gabor(imagen_pre)
    wavelet = aplicar_wavelet(imagen_pre)
    fourier = aplicar_fourier(imagen_pre)  # NUEVO

    if region:
        forma_solidez = region.solidity
        forma_extension = region.extent
        forma_eje_mayor = region.major_axis_length
        forma_eje_menor = region.minor_axis_length
        forma_relacion_aspecto = forma_eje_mayor / forma_eje_menor if forma_eje_menor != 0 else 0

        return [
            region.area,
            region.eccentricity,
            forma_solidez,
            forma_extension,
            forma_relacion_aspecto,
            orient,
            np.mean(gabor),
            wavelet,
            fourier  # NUEVA característica
        ]
    else:
        return [0, 0, 0, 0, 0, orient, np.mean(gabor), wavelet, fourier]  # Consistencia

# -------------------- Clasificación de madurez y calidad --------------------

def evaluar_calidad(imagen):
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    _, umbral = cv2.threshold(gris, 40, 255, cv2.THRESH_BINARY_INV)
    manchas = cv2.countNonZero(umbral)
    return "No óptima para consumo" if manchas > 500 else "Óptima para consumo"

def clasificar_madurez_por_fruta(fruta, imagen):
    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
    h_mean = np.mean(hsv[:, :, 0])
    s_mean = np.mean(hsv[:, :, 1])
    v_mean = np.mean(hsv[:, :, 2])

    if fruta == "mango":
        if h_mean > 50 and v_mean < 80:
            return "Biche"
        elif h_mean > 30:
            return "En su punto"
        else:
            return "Maduro"
    elif fruta == "platano":
        if h_mean > 40 and s_mean > 100:
            return "Biche"
        elif h_mean > 25:
            return "En su punto"
        else:
            return "Maduro"
    elif fruta == "aguacate":
        if v_mean > 120:
            return "Biche"
        elif v_mean > 80:
            return "En su punto"
        else:
            return "Maduro"
    elif fruta == "limon":
        if h_mean > 70:
            return "Biche"
        elif h_mean > 40:
            return "En su punto"
        else:
            return "Maduro"
    else:
        return "Desconocido"

# -------------------- Base de datos y prueba --------------------

def crear_base_caracteristicas(dataset):
    base = {}
    for fruta in os.listdir(dataset):
        ruta_fruta = os.path.join(dataset, fruta)
        if not os.path.isdir(ruta_fruta):
            continue
        print(f"Procesando imágenes para la fruta: {fruta}")
        base[fruta] = []
        for archivo in os.listdir(ruta_fruta):
            path = os.path.join(ruta_fruta, archivo)
            img = cv2.imread(path)
            if img is None:
                continue
            caract = extraer_caracteristicas_fruta(img)
            base[fruta].append(caract)
    print(base)  # Verifica la estructura del diccionario
    np.save("base_caracteristicas.npy", base)
    print("Base de características creada.")

def cargar_ultima_imagen(ruta):
    archivos = sorted(os.listdir(ruta), key=lambda x: os.path.getctime(os.path.join(ruta, x)))
    ultima = archivos[-1]
    path = os.path.join(ruta, ultima)
    imagen = cv2.imread(path)
    return imagen, ultima

def cargar_imagen(ruta, idx):
    archivos = sorted(os.listdir(ruta), key=lambda x: os.path.getctime(os.path.join(ruta, x)))
    if idx < 0 or idx >= len(archivos):
        raise IndexError("Índice fuera de rango")
    ultima = archivos[idx]
    path = os.path.join(ruta, ultima)
    imagen = cv2.imread(path)
    return imagen, ultima



def clasificar_imagen(IMG_PRUEBAS):
    # Cargar la imagen de prueba
    img = cv2.imread(IMG_PRUEBAS)
    caract_imagen = extraer_caracteristicas_fruta(img)

    # Cargar la base de características
    base_caracteristicas = np.load("base_caracteristicas.npy", allow_pickle=True).item()

    # Preparar vectores y etiquetas
    vectores = []
    etiquetas = []

    for fruta, lista_vectores in base_caracteristicas.items():
        for vec in lista_vectores:
            vectores.append(vec)
            etiquetas.append(fruta)

    # Añadir la imagen de prueba al final
    vectores.append(caract_imagen)

    # Normalizar todos juntos
    scaler = StandardScaler()
    vectores_normalizados = scaler.fit_transform(vectores)

    # Extraer la imagen de prueba ya normalizada (última fila)
    prueba_normalizada = vectores_normalizados[-1]
    base_normalizada = vectores_normalizados[:-1]

    # Comparar distancias
    menor_distancia = float('inf')
    mejor_fruta = None

    for i, vector in enumerate(base_normalizada):
        dist = euclidean(prueba_normalizada, vector)
        if dist < menor_distancia:
            menor_distancia = dist
            mejor_fruta = etiquetas[i]

    return mejor_fruta

# -------------------- Main --------------------

def main():
    imagen, nombre_archivo = cargar_imagen("IMG_PRUEBAS", 4)
    alto, ancho, canales = imagen.shape
    imagen_resized = cv2.resize(imagen, (ancho // 2, alto // 2))
    cv2.imshow("Fruta a evaluar", imagen_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    caract = extraer_caracteristicas_fruta(imagen)
    base = np.load("base_caracteristicas.npy", allow_pickle=True).item()
    fruta = clasificar_imagen("IMG_PRUEBAS/" + nombre_archivo)

    madurez = clasificar_madurez_por_fruta(fruta.lower(), imagen)
    calidad = evaluar_calidad(imagen)
    imagen_pre = preprocesar(imagen)
    binaria = binarizar(imagen_pre)
    region = extraer_region(binaria)
    forma = clasificar_forma(region) if region else "Desconocida"
    # Cargar la base
    base = np.load("base_caracteristicas.npy", allow_pickle=True).item()

    # Convertir a tabla
    rows = []
    for fruta, vectores in base.items():
        for vector in vectores:
            row = {'Fruta': fruta}
            row.update({f'F{i+1}': v for i, v in enumerate(vector)})
            rows.append(row)

    df = pd.DataFrame(rows)  # ← Aquí se usa `pd`
    print(df)

    base_caracteristicas = np.load("base_caracteristicas.npy", allow_pickle=True).item()
    for fruta, caracteristicas in base_caracteristicas.items():
     print(f"Fruta: {fruta}, Número de características: {len(caracteristicas)}")
    
    print("Subcarpetas encontradas en 'dataset':", os.listdir("dataset"))   
    print(f"Tu fruta a evaluar es: {nombre_archivo}")
    print(f"La fruta es: {fruta}")
    print(f"Forma geométrica: {forma}")
    print(f"Estado de madurez: {madurez}")
    print(f"Calidad para consumo: {calidad}")

    cv2.imshow("Fruta a evaluar", imagen)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 


if __name__ == "__main__":
    if not os.path.exists("base_caracteristicas.npy"):
        crear_base_caracteristicas("dataset")
    main()
