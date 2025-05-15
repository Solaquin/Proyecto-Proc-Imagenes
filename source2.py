import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import math
from collections import defaultdict

class FruitClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Clasificador de Frutas por Forma, Color y Madurez")

        self.create_scrollable_frame()

        # Variables de imagen
        self.imagen_original = None
        self.imagen_hsv = None
        self.mask_actual = None
        self.contorno_actual = None
        
        # Rangos de color para cada fruta (en HSV)
        self.rangos_color = {
            "Plátano": {"lower": np.array([20, 50, 50]), "upper": np.array([40, 255, 255])},
            "Mango": {"lower": np.array([5, 97, 95]), "upper": np.array([39, 152, 166])},
            "Aguacate": {"lower": np.array([27, 126, 73]), "upper": np.array([42, 202, 140])},
            "Limón": {"lower": np.array([16, 66, 70]), "upper": np.array([36, 170, 199])}
        }

        # Rangos para detección de manchas (madurez)
        self.rangos_manchas = {
            "oscuro": {"lower": np.array([0, 0, 0]), "upper": np.array([180, 255, 100])},
            "maduro": {"lower": np.array([0, 50, 150]), "upper": np.array([180, 255, 255])}
        }

        # Configuración de la interfaz
        self.setup_ui()

    def create_scrollable_frame(self):
        # Crear un canvas y scrollbar vertical
        self.canvas = tk.Canvas(self.root)
        self.scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        # Configurar el frame para que se actualice el scroll
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )
        
        # Crear una ventana en el canvas para el frame
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Empaquetar los elementos
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

    def setup_ui(self):
        # Paneles de visualización
        self.panel_original = tk.Label(self.scrollable_frame)
        self.panel_original.grid(row=0, column=0, padx=5, pady=5)

        self.panel_contorno = tk.Label(self.scrollable_frame)
        self.panel_contorno.grid(row=0, column=1, padx=5, pady=5)

        self.panel_mask = tk.Label(self.scrollable_frame)
        self.panel_mask.grid(row=0, column=2, padx=5, pady=5)

        self.panel_madurez = tk.Label(self.scrollable_frame)
        self.panel_madurez.grid(row=1, column=1, padx=5, pady=5)

        # Botón para cargar imagen
        btn_cargar = tk.Button(self.scrollable_frame, text="Cargar Imagen", command=self.cargar_imagen)
        btn_cargar.grid(row=1, column=0, pady=5, sticky='we')

        # Sliders HSV para fruta
        self.sliders = {}
        etiquetas = ['H low', 'S low', 'V low', 'H high', 'S high', 'V high']
        rangos = [(0,179), (0,255), (0,255), (0,179), (0,255), (0,255)]
        valores_iniciales = [20, 50, 50, 80, 255, 255]

        for i, (etiqueta, rango, val) in enumerate(zip(etiquetas, rangos, valores_iniciales)):
            tk.Label(self.scrollable_frame, text=etiqueta).grid(row=2+i, column=0, sticky='w')
            self.sliders[etiqueta] = tk.Scale(self.scrollable_frame, from_=rango[0], to=rango[1], orient='horizontal',
                                            command=self.actualizar_mascara)
            self.sliders[etiqueta].set(val)
            self.sliders[etiqueta].grid(row=2+i, column=1, columnspan=2, sticky='we')

        # Botón para clasificar
        btn_clasificar = tk.Button(self.scrollable_frame, text="Clasificar Fruta", command=self.clasificar_fruta)
        btn_clasificar.grid(row=8, column=0, columnspan=3, pady=10, sticky='we')

        # Área de resultados
        self.resultado_label = tk.Label(self.scrollable_frame, text="Seleccione una imagen y ajuste la máscara", 
                                      font=("Arial", 12))
        self.resultado_label.grid(row=9, column=0, columnspan=3)

        # Debug info
        self.debug_text = tk.Text(self.scrollable_frame, height=12, width=80, state='disabled')
        self.debug_text.grid(row=10, column=0, columnspan=3, padx=5, pady=5)

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def cargar_imagen(self):
        path = filedialog.askopenfilename()
        if not path:
            return
        
        self.imagen_original = cv2.imread(path)
        if self.imagen_original is None:
            self.mostrar_error("No se pudo cargar la imagen")
            return
        
        self.imagen_hsv = cv2.cvtColor(self.imagen_original, cv2.COLOR_BGR2HSV)
        self.mostrar_imagen(self.imagen_original, self.panel_original)
        self.actualizar_mascara()
        self.log_debug(f"Imagen cargada: {path}\nDimensiones: {self.imagen_original.shape}")

    def mostrar_imagen(self, img_cv, panel):
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_pil = img_pil.resize((300, 300), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img_pil)
        panel.img_tk = img_tk
        panel.config(image=img_tk)

    def mostrar_contorno(self, img_cv, contorno, panel):
        img_contorno = img_cv.copy()
        cv2.drawContours(img_contorno, [contorno], -1, (0, 255, 0), 3)
        
        rect = cv2.minAreaRect(contorno)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(img_contorno, [box], 0, (0, 0, 255), 2)
        
        self.mostrar_imagen(img_contorno, panel)

    def mostrar_mascara(self, mask, panel):
        mask_visual = mask.copy()
        img_pil = Image.fromarray(mask_visual)
        img_pil = img_pil.resize((300, 300), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        panel.img_tk = img_tk
        panel.config(image=img_tk)

    def actualizar_mascara(self, event=None):
        if self.imagen_hsv is None:
            return
            
        h_low = self.sliders['H low'].get()
        s_low = self.sliders['S low'].get()
        v_low = self.sliders['V low'].get()
        h_high = self.sliders['H high'].get()
        s_high = self.sliders['S high'].get()
        v_high = self.sliders['V high'].get()

        lower_hsv = np.array([h_low, s_low, v_low])
        upper_hsv = np.array([h_high, s_high, v_high])

        self.mask_actual = cv2.inRange(self.imagen_hsv, lower_hsv, upper_hsv)
        self.mostrar_mascara(self.mask_actual, self.panel_mask)
        self.actualizar_contornos()

    def actualizar_contornos(self):
        if self.mask_actual is None:
            return
            
        contornos, _ = cv2.findContours(self.mask_actual, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contornos:
            self.contorno_actual = max(contornos, key=cv2.contourArea)
            self.mostrar_contorno(self.imagen_original, self.contorno_actual, self.panel_contorno)
            self.log_debug(f"Contorno detectado - Área: {cv2.contourArea(self.contorno_actual):.1f} px")
        else:
            self.contorno_actual = None
            blank_image = np.zeros((300, 300, 3), dtype=np.uint8)
            self.mostrar_imagen(blank_image, self.panel_contorno)
            self.log_debug("No se detectaron contornos")

    def detectar_madurez(self, img_hsv, mask):
        """Determina el estado de madurez basado en manchas oscuras y color"""
        # Máscara para áreas oscuras (manchas)
        mask_oscuro = cv2.inRange(img_hsv, 
                                self.rangos_manchas["oscuro"]["lower"], 
                                self.rangos_manchas["oscuro"]["upper"])
        
        # Máscara para áreas maduras (colores intensos)
        mask_maduro = cv2.inRange(img_hsv,
                                 self.rangos_manchas["maduro"]["lower"],
                                 self.rangos_manchas["maduro"]["upper"])
        
        # Aplicar máscara de fruta
        mask_oscuro = cv2.bitwise_and(mask_oscuro, mask)
        mask_maduro = cv2.bitwise_and(mask_maduro, mask)
        
        # Calcular porcentajes
        total_pixels = np.sum(mask) / 255
        pix_oscuros = np.sum(mask_oscuro) / 255
        pix_maduros = np.sum(mask_maduro) / 255
        
        if total_pixels == 0:
            return "Desconocido", np.zeros_like(self.imagen_original)
        
        porcentaje_oscuro = pix_oscuros / total_pixels
        porcentaje_maduro = pix_maduros / total_pixels
        
        # Visualización para debug
        img_madurez = self.imagen_original.copy()
        img_madurez[mask_oscuro == 255] = (0, 0, 255)  # Rojo para manchas
        img_madurez[mask_maduro == 255] = (0, 255, 0)  # Verde para madurez
        
        # Determinar estado
        if porcentaje_oscuro > 0.3:
            return "Biche", img_madurez
        elif porcentaje_maduro > 0.4:
            return "Maduro", img_madurez
        elif porcentaje_maduro > 0.2:
            return "En su punto", img_madurez
        else:
            return "Biche", img_madurez

    def clasificar_por_color(self, img_hsv, mask, tipo_fruta):
        if tipo_fruta not in self.rangos_color:
            return 0.0
            
        lower = self.rangos_color[tipo_fruta]["lower"]
        upper = self.rangos_color[tipo_fruta]["upper"]
        
        color_mask = cv2.inRange(img_hsv, lower, upper)
        combined_mask = cv2.bitwise_and(color_mask, mask)
        
        total_pixels = np.sum(mask) / 255
        matched_pixels = np.sum(combined_mask) / 255
        
        if total_pixels > 0:
            return matched_pixels / total_pixels
        return 0.0

    def clasificar_fruta(self):
        if self.imagen_original is None or self.contorno_actual is None:
            self.mostrar_error("Primero cargue una imagen y ajuste la máscara para detectar un contorno")
            return

        contorno = self.contorno_actual
        area = cv2.contourArea(contorno)
        
        # 1. Clasificación por forma
        rect = cv2.minAreaRect(contorno)
        (x, y), (w, h), angulo = rect
        
        if w > h:
            w, h = h, w
        
        relacion_aspecto = w / h
        perimetro = cv2.arcLength(contorno, True)
        circularidad = (4 * math.pi * area) / (perimetro ** 2) if perimetro > 0 else 0
        
        # Resultados preliminares por forma
        candidatos = []
        if relacion_aspecto < 0.5 and circularidad < 0.7:
            candidatos.append("Plátano")
        if 0.312 < relacion_aspecto < 0.77 and 0.020 < circularidad < 0.769:
            candidatos.append("Aguacate")
        if 0.591 < relacion_aspecto < 0.750 and 0.046 < circularidad < 0.772:
            candidatos.append("Limón")
        if 0.405 < relacion_aspecto < 0.889 and 0.019 < circularidad < 0.772:
            candidatos.append("Mango")
        
        if not candidatos:
            self.resultado_label.config(text="Resultado: No identificado (forma no coincide)")
            self.log_debug("\nClasificación: No se encontraron candidatos por forma")
            return
        
        # 2. Refinar por color
        resultados_color = {}
        for candidato in candidatos:
            score = self.clasificar_por_color(self.imagen_hsv, self.mask_actual, candidato)
            resultados_color[candidato] = score
            self.log_debug(f"Coincidencia de color con {candidato}: {score:.3%}")
        
        # Seleccionar el mejor candidato
        mejor_candidato = max(resultados_color.items(), key=lambda x: x[1])
        
        # 3. Detectar madurez
        madurez, img_madurez = self.detectar_madurez(self.imagen_hsv, self.mask_actual)
        self.mostrar_imagen(img_madurez, self.panel_madurez)
        
        # Debug info
        self.log_debug("\n=== Características de Forma ===")
        self.log_debug(f"Área: {area:.1f} px | Rel. aspecto: {relacion_aspecto:.3f}")
        self.log_debug(f"Circularidad: {circularidad:.3f}")
        self.log_debug("\n=== Resultados Madurez ===")
        self.log_debug(f"Estado: {madurez}")
        
        if mejor_candidato[1] > 0.3:
            self.resultado_label.config(text=f"Resultado: {mejor_candidato[0]} - {madurez} (Confianza: {mejor_candidato[1]:.0%})")
        else:
            self.resultado_label.config(text=f"Resultado: {candidatos[0]} - {madurez} (solo por forma)")

    def log_debug(self, message):
        self.debug_text.config(state='normal')
        self.debug_text.insert('end', message + '\n')
        self.debug_text.see('end')
        self.debug_text.config(state='disabled')

    def mostrar_error(self, message):
        self.resultado_label.config(text=f"Error: {message}", fg="red")
        self.log_debug(f"! Error: {message}")

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("960x800")  # Puedes ajustar estos valores según necesites
    root.minsize(960, 600)  # Ancho mínimo, alto mínimo
    app = FruitClassifierApp(root)
    root.mainloop()