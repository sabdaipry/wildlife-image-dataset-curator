import os
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QLabel, QPushButton, QMessageBox,
                               QListWidget, QListWidgetItem,
                               QAbstractItemView, QSplitter, QTabWidget)
from PySide6.QtGui import QColor
from PySide6.QtCore import Qt

from config import ARCHIVO_DATOS, CARPETA_DESCARTES, ESTILOS, NOMBRE_MODELO_EMBEDDING
from core.data_manager import DataManager
from ui.umap_widget import UMAPWidget
from ui.visor_zoom_widget import VisorZoomWidget
from ui.pestana_estadisticas import StatisticsPanel


class AuditoriaMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sistema de Auditoría de Dataset")
        self.resize(1000, 700)

        self.manager = DataManager(ARCHIVO_DATOS, CARPETA_DESCARTES)
        self.ventana_stats = None

        self._setup_ui()
        self._conectar_senales()

        self.es_modo_oscuro = True
        self.aplicar_tema_oscuro()

        if not self.manager.df.empty:
            x, y, c, idxs = self.manager.get_puntos_umap()
            self.umap_widget.graficar(x, y, c, idxs)
        else:
            QMessageBox.warning(self, "Atención", "No hay datos para mostrar.")

    def _setup_ui(self):
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        pestana_auditoria = QWidget()
        layout_auditoria = QHBoxLayout(pestana_auditoria)

        splitter = QSplitter(Qt.Horizontal)

        self.umap_widget = UMAPWidget(nombre_modelo=NOMBRE_MODELO_EMBEDDING)
        splitter.addWidget(self.umap_widget)

        panel_derecho = QWidget()
        layout_derecho = QVBoxLayout(panel_derecho)

        layout_botones = QHBoxLayout()
        self.btn_tema = QPushButton()
        self.btn_tema.setObjectName("btn_tema")
        self.btn_tema.setCursor(Qt.PointingHandCursor)
        self.btn_tema.setFixedWidth(30)

        self.btn_borrar = QPushButton("🗑️ Eliminar Seleccionados")
        self.btn_borrar.setObjectName("btn_borrar")
        self.btn_borrar.setEnabled(False)
        self.btn_borrar.setCursor(Qt.PointingHandCursor)

        layout_botones.addWidget(self.btn_tema)
        layout_botones.addStretch()
        layout_botones.addWidget(self.btn_borrar)
        layout_derecho.addLayout(layout_botones)

        self.lbl_lista_titulo = QLabel("Archivos Seleccionados:")
        layout_derecho.addWidget(self.lbl_lista_titulo)

        self.lista_archivos = QListWidget()
        self.lista_archivos.setSelectionMode(QAbstractItemView.SingleSelection)
        layout_derecho.addWidget(self.lista_archivos, stretch=1)

        self.lbl_info = QLabel("Seleccione un punto en el mapa.")
        self.lbl_info.setWordWrap(True)
        self.lbl_info.setAlignment(Qt.AlignCenter)
        self.lbl_info.setStyleSheet("font-size: 14px; padding: 10px;")
        layout_derecho.addWidget(self.lbl_info)

        self.visor = VisorZoomWidget()
        layout_derecho.addWidget(self.visor, stretch=2)

        splitter.addWidget(panel_derecho)
        splitter.setSizes([700, 400])

        layout_auditoria.addWidget(splitter)

        self.tab_stats = StatisticsPanel(self.manager)

        self.tabs.addTab(pestana_auditoria, "🔍 Auditoría Visual")
        self.tabs.addTab(self.tab_stats, "📊 Estadísticas y Control")

    def _conectar_senales(self):
        self.umap_widget.seleccion_realizada.connect(self.procesar_seleccion_lote)
        self.umap_widget.punto_cliqueado.connect(self.procesar_seleccion_unica)

        self.lista_archivos.itemClicked.connect(self.cargar_preview_desde_lista)
        self.lista_archivos.itemChanged.connect(self.actualizar_conteo_borrado)
        self.btn_borrar.clicked.connect(self.ejecutar_borrado)
        self.btn_tema.clicked.connect(self.alternar_tema)

        self.umap_widget.canvas.mpl_connect('button_press_event', self.on_canvas_click_main)

        self.manager.data_changed.connect(self.recargar_mapa)

    def recargar_mapa(self):
        x, y, c, idxs = self.manager.get_puntos_umap()
        self.umap_widget.graficar(x, y, c, idxs)
        self.limpiar_seleccion()

    def on_canvas_click_main(self, event):
        if event.button != 1:
            return
        if self.umap_widget.btn_lazo.isChecked():
            return
        contains, _ = self.umap_widget.ax.collections[0].contains(event)
        if not contains:
            self.limpiar_seleccion()

    def limpiar_seleccion(self):
        self.lista_archivos.clear()
        self.lbl_info.setText("Selección limpiada.")
        self.btn_borrar.setText("🗑️ Eliminar Seleccionados")
        self.btn_borrar.setEnabled(False)
        self.visor.scene.clear()
        self.umap_widget.limpiar_dibujo_lazo()

    def procesar_seleccion_lote(self, indices_potenciales):
        if not indices_potenciales:
            self.limpiar_seleccion()
            return

        self.lista_archivos.blockSignals(True)
        self.lista_archivos.clear()
        count = 0

        for idx in indices_potenciales:
            reg = self.manager.get_info_registro(idx)
            if reg['estado'] == 'borrado':
                continue
            item = QListWidgetItem(f"{reg['archivo']} ({reg.get('nombre_comun','?')})")
            item.setData(Qt.UserRole, int(idx))
            item.setCheckState(Qt.Checked)
            self.lista_archivos.addItem(item)
            count += 1

        self.lista_archivos.blockSignals(False)
        self.actualizar_conteo_borrado()

        self.lbl_info.setText(f"Lote seleccionado: {count} imágenes.")
        self.btn_borrar.setEnabled(count > 0)
        if count > 0:
            self.btn_borrar.setText(f"🗑️ Eliminar {count} elementos")

    def procesar_seleccion_unica(self, idx):
        reg = self.manager.get_info_registro(idx)
        if reg['estado'] == 'borrado':
            self.lbl_info.setText("❌ Imagen ya eliminada.")
            return

        self.lista_archivos.clear()
        item = QListWidgetItem(f"{reg['archivo']}")
        item.setData(Qt.UserRole, int(idx))
        item.setCheckState(Qt.Checked)
        self.lista_archivos.addItem(item)
        self.lista_archivos.setCurrentRow(0)

        self.cargar_preview_desde_lista(item)
        self.btn_borrar.setEnabled(True)
        self.btn_borrar.setText("🗑️ Eliminar 1 elemento")

    def cargar_preview_desde_lista(self, item):
        idx = item.data(Qt.UserRole)
        reg = self.manager.get_info_registro(idx)

        familia = reg.get('family', reg.get('familia', 'Desconocida'))
        genero = reg.get('genus', reg.get('genero', 'Desconocido'))
        nombre_comun = reg.get('nombre_comun', 'Sin nombre común')
        nombre_cientifico = reg.get('nombre_cientifico', 'Scientific Name')

        ruta = str(reg['ruta_absoluta'])
        existe_fisicamente = os.path.exists(ruta)

        if existe_fisicamente:
            aviso_estado = ""
        else:
            self.visor.scene.clear()
            aviso_estado = ("<br><br><h2 style='color: #ff4444; background-color: #330000; padding: 5px;'>"
                            "⚠️ IMAGEN NO DISPONIBLE<br><span style='font-size:12px'>"
                            "(Archivo eliminado o movido)</span></h2>")

        texto = (f"<h2 style='margin:0'>{nombre_comun}</h2>"
                 f"<i style='color:#888; font-size: 14px'>{nombre_cientifico}</i><br>"
                 f"<b>Familia:</b> {familia} | <b>Género:</b> {genero}"
                 f"</p>"
                 f"{aviso_estado}")
        self.lbl_info.setText(texto)

        self.visor.cargar_imagen(ruta)

    def ejecutar_borrado(self):
        indices_a_borrar = []
        for i in range(self.lista_archivos.count()):
            item = self.lista_archivos.item(i)
            if item.checkState() == Qt.Checked:
                indices_a_borrar.append(item.data(Qt.UserRole))

        if not indices_a_borrar:
            return

        resp = QMessageBox.question(self, "Confirmar",
                                    f"¿Mover {len(indices_a_borrar)} imágenes a descartes?",
                                    QMessageBox.Yes | QMessageBox.No)
        if resp == QMessageBox.No:
            return

        movidos, errores = self.manager.mover_a_descartes(indices_a_borrar)
        self.lbl_info.setText(f"{movidos} movidos. {errores} errores.")
        self.visor.scene.clear()
        self.btn_borrar.setEnabled(False)

    def alternar_tema(self):
        self.es_modo_oscuro = not self.es_modo_oscuro
        if self.es_modo_oscuro:
            self.aplicar_tema_oscuro()
        else:
            self.aplicar_tema_claro()

    def aplicar_tema_oscuro(self):
        self.setStyleSheet(ESTILOS["oscuro"])
        self.umap_widget.set_tema(True)
        self.visor.setBackgroundBrush(QColor("#1e1e1e"))
        self.btn_tema.setText("🌞")
        self.btn_tema.setToolTip("Cambiar a modo claro")
        if self.ventana_stats:
            self.ventana_stats.set_tema(True)

    def aplicar_tema_claro(self):
        self.setStyleSheet(ESTILOS["claro"])
        self.umap_widget.set_tema(False)
        self.visor.setBackgroundBrush(QColor("#f0f0f0"))
        self.btn_tema.setText("🌚")
        self.btn_tema.setToolTip("Cambiar a modo oscuro")
        if self.ventana_stats:
            self.ventana_stats.set_tema(False)

    def actualizar_conteo_borrado(self, item=None):
        count = sum(
            1 for i in range(self.lista_archivos.count())
            if self.lista_archivos.item(i).checkState() == Qt.Checked
        )
        if count > 0:
            self.btn_borrar.setEnabled(True)
            self.btn_borrar.setText(f"🗑️ Eliminar {count} Seleccionados")
        else:
            self.btn_borrar.setEnabled(False)
            self.btn_borrar.setText("🗑️ Eliminar Seleccionados")
