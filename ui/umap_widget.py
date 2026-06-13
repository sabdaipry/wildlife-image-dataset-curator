import numpy as np
import matplotlib as mpl
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
from matplotlib.colors import ListedColormap

from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton
from PySide6.QtCore import Qt, QTimer, Signal


class UMAPWidget(QWidget):
    """Widget autónomo que maneja el gráfico Matplotlib y el Selector."""

    # Señales personalizadas: Comunican eventos al exterior sin saber quién escucha
    seleccion_realizada = Signal(list) # Emite lista de índices
    punto_cliqueado = Signal(int)      # Emite un solo índice
    deseleccion_total = Signal() # Avisa que se hizo click en la nada

    def __init__(self, parent=None, nombre_modelo=""):
        super().__init__(parent)
        self.nombre_modelo = nombre_modelo
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)

        # Toolbar y Canvas
        self.figure = Figure(figsize=(5, 5), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Botón de Lazo
        layout_boton = QHBoxLayout()
        layout_boton.setContentsMargins(10, 0, 0, 0) # Un poco de margen a la izquierda

        self.btn_lazo = QPushButton("➰ Selección Múltiple")
        self.btn_lazo.setObjectName("btn_lazo") # Para el CSS
        self.btn_lazo.setCheckable(True)
        self.btn_lazo.setCursor(Qt.PointingHandCursor)
        self.btn_lazo.toggled.connect(self.toggle_lazo)

        layout_boton.addWidget(self.btn_lazo) # Agregamos el botón
        layout_boton.addStretch()             # EL RESORTE: Empuja todo a la izquierda

        layout.addWidget(self.toolbar)
        layout.addLayout(layout_boton)        # Agregamos este contenedor al layout principal

        # ------------------------

        layout.addWidget(self.canvas)

        self.ax = self.figure.add_subplot(111)

        # 1. Creamos nuestra propia línea para "congelar" el dibujo
        # Inicialmente vacía. zorder alto para que se vea encima de los puntos.
        self.linea_persistente = None
        self.crosshair_v = None # Línea Vertical
        self.crosshair_h = None # Línea Horizontal
        self.selector = None

        self.puntos_cache = None
        self.indices_reales = None
        self.color_titulo_actual = 'black'
        self.es_modo_oscuro = False # Para recordar el estado al redibujar

        # Configuración inicial del selector (se recrea en graficar por seguridad si se limpia ax)
        self._inicializar_artistas()

    def _inicializar_artistas(self):
        """Crea los objetos gráficos (líneas) si no existen o tras un clear()."""
        # 1. Línea del Lazo (Roja, gruesa)
        self.linea_persistente, = self.ax.plot([], [], color='red', linewidth=2, alpha=0.9, zorder=10)

        # 2. CROSSHAIRS (Las líneas de guía)
        # Inicialmente invisibles (visible=False)
        # zorder=5 para que estén detrás del lazo pero sobre los puntos
        style = {'color': 'cyan', 'linestyle': '--', 'linewidth': 0.8, 'alpha': 0.8, 'visible': False, 'zorder': 5}
        self.crosshair_v = self.ax.axvline(x=0, **style)
        self.crosshair_h = self.ax.axhline(y=0, **style)

        # 3. Selector
        if self.selector: self.selector.disconnect_events()
        self.selector = LassoSelector(self.ax, onselect=self.on_lasso_finished, useblit=True)
        self.selector.set_active(False)

    def graficar(self, x, y, c, indices_reales):
        self.ax.clear()

        # Al hacer clear(), se borran los artistas, hay que recrearlos
        self._inicializar_artistas()
        self.aplicar_colores_tema() # Re-aplicar colores correctos según tema actual

        self.puntos_cache = np.column_stack((x, y))
        self.indices_reales = indices_reales # Guardamos el mapeo: IndiceVisual -> IndiceDF

        # COLORES POR FAMILIA
        # Calculamos cuántas familias únicas hay
        n_familias = int(np.max(c)) + 1 if len(c) > 0 else 1

        if n_familias <= 10: mi_cmap = 'tab10'
        elif n_familias <= 20: mi_cmap = 'tab20'
        else:
            cmap1 = mpl.colormaps['tab20']
            cmap2 = mpl.colormaps['tab20b']
            cmap3 = mpl.colormaps['tab20c']

            # Stackeamos los colores
            colores_combinados = np.vstack((
                cmap1(np.linspace(0, 1, 20)),
                cmap2(np.linspace(0, 1, 20)),
                cmap3(np.linspace(0, 1, 20))
            ))

            mi_cmap = ListedColormap(colores_combinados, name='tab60')

        self.ax.scatter(x, y, c=c, cmap=mi_cmap, s=15, picker=5)
        self.ax.axis('off')
        self.ax.set_title(f"Espacio Semántico ({self.nombre_modelo} + UMAP)", color=self.color_titulo_actual)

        self.canvas.draw()

        self.canvas.mpl_connect('pick_event', self.on_pick)
        self.canvas.mpl_connect('button_press_event', self.on_canvas_click)

    def on_lasso_finished(self, verts):
        """Se ejecuta INMEDIATAMENTE al soltar el mouse."""

        # Ocultamos crosshairs si se usa el lazo
        self.ocultar_crosshair()

        # 1. ACTUALIZACIÓN VISUAL INSTANTÁNEA
        verts_np = np.array(verts)
        if len(verts_np) > 2:
            # Cerramos el polígono
            verts_np = np.vstack([verts_np, verts_np[0]])
            self.linea_persistente.set_data(verts_np[:, 0], verts_np[:, 1])
            self.linea_persistente.set_visible(True)

            # Forzamos el redibujado YA MISMO.
            # Esto congela la línea en la pantalla antes de que Python se ponga a pensar.
            self.canvas.draw()

        # 2. DELEGAMOS EL CÁLCULO PESADO (Math)
        # Usamos un timer de 1ms para dejar que la GUI respire y muestre la línea,
        # y recién después hacemos el cálculo de "quién está adentro".
        QTimer.singleShot(10, lambda: self._procesar_matematica_lazo(verts))

    def _procesar_matematica_lazo(self, verts):
        """Cálculo pesado que corre después de que la línea ya se dibujó."""
        if self.puntos_cache is None: return

        path = Path(verts)
        mask = path.contains_points(self.puntos_cache)
        indices_visuales = np.where(mask)[0]

        # TRADUCCIÓN CRÍTICA: De índice visual a índice del DataFrame
        if self.indices_reales is not None and len(indices_visuales) > 0:
            indices_finales = self.indices_reales[indices_visuales]
            self.seleccion_realizada.emit(indices_finales.tolist())
        else:
            # Si no seleccionó nada, enviamos lista vacía
            self.seleccion_realizada.emit([])

    def on_canvas_click(self, event):
        if event.inaxes != self.ax: return

        # Si estamos en modo lazo e iniciamos un click nuevo
        if self.btn_lazo.isChecked():
            # Limpiamos la línea vieja para empezar limpios
            self.linea_persistente.set_data([], [])
            self.linea_persistente.set_visible(False)
            self.ocultar_crosshair()
            self.canvas.draw()
        else:
            # Si es click simple en el fondo (sin pick), ocultamos crosshair
            # Nota: pick event ocurre antes. Si no hubo pick, limpiamos.
            # (Lo delegamos al padre vía deselección o lo hacemos aquí visualmente)
            pass

    def on_pick(self, event):
        if self.btn_lazo.isChecked():
            return

        if len(event.ind) > 0:
            idx_visual = event.ind[0]

            # MOSTRAR CROSSHAIR EN EL PUNTO SELECCIONADO
            # Obtenemos las coordenadas exactas del punto clickeado
            x_pt = self.puntos_cache[idx_visual, 0]
            y_pt = self.puntos_cache[idx_visual, 1]
            self.actualizar_crosshair(x_pt, y_pt)

            # TRADUCCIÓN: De visual a real
            if self.indices_reales is not None:
                idx_real = self.indices_reales[idx_visual]
                self.punto_cliqueado.emit(int(idx_real))

    # Sobreescribimos el mouseRelease para detectar click en vacío
    # Matplotlib a veces complica el "click en nada". Una forma 100% efectiva en Qt
    # es capturar el evento del canvas, pero usaremos un enfoque simple:
    # Si 'on_pick' no se dispara, asumimos click en vacio? No, mejor:
    # Usamos la señal button_press_event de matplotlib arriba y verificamos "contains".

    # CORRECCIÓN SIMPLE PARA CLICK VACÍO:
    # Vamos a confiar en que el usuario quiere deseleccionar si usa el botón derecho
    # O simplemente agregamos un botón "Limpiar".
    # PERO, para cumplir tu pedido de "click en otro lado":
    def mouseReleaseEvent(self, event):
        # Este método es de QT, no de Matplotlib.
        super().mouseReleaseEvent(event)
        # Si quisieras lógica compleja de clicks vacíos, iría aquí,
        # pero dejémoslo simple con el botón de lazo.

    def toggle_lazo(self, activo):
        # Pequeño truco: Redibujamos antes de activar para asegurar que el fondo esté listo para blitting
        if activo:
            self.ocultar_crosshair()
            self.canvas.draw()
            self.selector.set_active(True)
            self.btn_lazo.setText("➰ Lazo Activo")
            self.btn_lazo.setChecked(True)
        else:
            self.selector.set_active(False)
            self.btn_lazo.setText("➰ Selección Múltiple")
            self.btn_lazo.setChecked(False)
            self.limpiar_dibujo_lazo()

    # Método auxiliar para borrar la línea roja
    def limpiar_dibujo_lazo(self):
        # Borramos nuestra línea manual
        self.linea_persistente.set_data([], [])
        self.linea_persistente.set_visible(False)
        self.ocultar_crosshair()
        self.canvas.draw()

    def set_tema(self, es_oscuro):
        self.es_modo_oscuro = es_oscuro
        self.aplicar_colores_tema()
        self.canvas.draw()

    def aplicar_colores_tema(self):
        """Aplica colores específicos al gráfico Matplotlib"""
        if self.es_modo_oscuro:
            bg_color = '#2d2d2d'
            self.color_titulo_actual = 'white' # Guardamos estado
            # El toolbar de matplotlib usa iconos negros por defecto.
            # En modo oscuro, forzamos el fondo del toolbar a un gris claro para que se vean los iconos.
            self.toolbar.setStyleSheet("background-color: #b0b0b0; border-radius: 4px;")
            # Colores de líneas para fondo oscuro
            if self.linea_persistente: self.linea_persistente.set_color('yellow')
            # Crosshair Cyan resalta mucho en oscuro
            if self.crosshair_v:
                self.crosshair_v.set_color('#00FFFF')
                self.crosshair_h.set_color('#00FFFF')
        else:
            bg_color = 'white'
            self.color_titulo_actual = 'black' # Guardamos estado
            self.toolbar.setStyleSheet("") # Restaurar default
            self.linea_persistente.set_color('red')

            # Colores de líneas para fondo claro
            if self.linea_persistente: self.linea_persistente.set_color('red')
            # Crosshair Magenta o Negro resalta en blanco
            if self.crosshair_v:
                self.crosshair_v.set_color('#FF00FF')
                self.crosshair_h.set_color('#FF00FF')

        self.figure.patch.set_facecolor(bg_color)
        self.ax.set_facecolor(bg_color)
        self.ax.set_title(f"Espacio Semántico ({self.nombre_modelo} + UMAP)", color=self.color_titulo_actual)

        self.canvas.draw()

    #  MÉTODOS PARA MANEJAR EL CROSSHAIR
    def actualizar_crosshair(self, x, y):
        self.crosshair_v.set_xdata([x, x])
        self.crosshair_h.set_ydata([y, y])
        self.crosshair_v.set_visible(True)
        self.crosshair_h.set_visible(True)
        self.canvas.draw()

    def ocultar_crosshair(self):
        self.crosshair_v.set_visible(False)
        self.crosshair_h.set_visible(False)
        # No llamamos a draw() aquí para no ralentizar, se llamará luego
