import os
import pytest
import pandas as pd
from core.data_manager import DataManager


# ── sample data ───────────────────────────────────────────────────────────────

SAMPLE_ROWS = [
    {
        "nombre_cientifico": "Panthera leo",
        "nombre_comun": "Lion",
        "familia": "Felidae",
        "genero": "Panthera",
        "x": 1.0, "y": 1.0,
        "ruta_absoluta": "/data/images/cat1/img1.jpg",
        "estado": "activo",
    },
    {
        "nombre_cientifico": "Panthera leo",
        "nombre_comun": "Lion",
        "familia": "Felidae",
        "genero": "Panthera",
        "x": 2.0, "y": 2.0,
        "ruta_absoluta": "/data/images/cat1/img2.jpg",
        "estado": "activo",
    },
    {
        "nombre_cientifico": "Ailurus fulgens",
        "nombre_comun": "Red Panda",
        "familia": "Ailuridae",
        "genero": "Ailurus",
        "x": 5.0, "y": 5.0,
        "ruta_absoluta": "/data/images/cat2/img3.jpg",
        "estado": "borrado",
    },
]


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def csv_full(tmp_path):
    p = tmp_path / "datos.csv"
    pd.DataFrame(SAMPLE_ROWS).to_csv(p, index=False)
    return p


@pytest.fixture
def csv_no_estado(tmp_path):
    rows = [{k: v for k, v in r.items() if k != "estado"} for r in SAMPLE_ROWS]
    p = tmp_path / "datos_sin_estado.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


@pytest.fixture
def dm(qapp, csv_full, tmp_path):
    """DataManager with 3 rows: indices 0 and 1 active, index 2 deleted."""
    return DataManager(str(csv_full), str(tmp_path / "trash"))


# ── _cargar_datos (via __init__) ──────────────────────────────────────────────

class TestLoadData:

    def test_loads_csv_correctly(self, qapp, csv_full, tmp_path):
        dm = DataManager(str(csv_full), str(tmp_path / "trash"))
        assert len(dm.df) == 3
        assert list(dm.df["estado"]) == ["activo", "activo", "borrado"]

    def test_csv_without_estado_column_adds_it_as_activo(self, qapp, csv_no_estado, tmp_path):
        dm = DataManager(str(csv_no_estado), str(tmp_path / "trash"))
        assert "estado" in dm.df.columns
        assert (dm.df["estado"] == "activo").all()

    def test_missing_file_returns_empty_dataframe(self, qapp, tmp_path):
        dm = DataManager(str(tmp_path / "no_existe.csv"), str(tmp_path / "trash"))
        assert dm.df.empty


# ── get_resumen_global ────────────────────────────────────────────────────────

class TestGetGlobalSummary:

    def test_summary_with_normal_data(self, dm):
        resumen = dm.get_resumen_global()
        assert resumen["total_imgs"] == 3
        assert resumen["activas"] == 2
        assert resumen["borradas"] == 1
        assert resumen["n_especies"] == 2   # Panthera leo + Ailurus fulgens
        assert resumen["n_familias"] == 2   # Felidae + Ailuridae

    def test_empty_dataframe_returns_empty_dict(self, qapp, tmp_path):
        dm = DataManager(str(tmp_path / "no_existe.csv"), str(tmp_path / "trash"))
        assert dm.get_resumen_global() == {}


# ── filtrar_por_lazo ──────────────────────────────────────────────────────────

class TestFilterByLasso:

    # Square enclosing (1,1) and (2,2) but not (5,5)
    LASSO_WITH_POINTS = [(-0.5, -0.5), (2.5, -0.5), (2.5, 2.5), (-0.5, 2.5)]
    EMPTY_LASSO       = [(10.0, 10.0), (20.0, 10.0), (20.0, 20.0), (10.0, 20.0)]

    def test_returns_indices_of_points_inside(self, dm):
        indices = dm.filtrar_por_lazo(self.LASSO_WITH_POINTS)
        assert set(indices) == {0, 1}

    def test_empty_lasso_returns_empty_list(self, dm):
        assert dm.filtrar_por_lazo(self.EMPTY_LASSO) == []

    def test_excludes_deleted_points_even_if_inside_lasso(self, qapp, tmp_path):
        csv_path = tmp_path / "con_borrado.csv"
        pd.DataFrame([
            {"x": 1.0, "y": 1.0, "estado": "activo"},
            {"x": 1.5, "y": 1.5, "estado": "borrado"},  # inside the lasso but deleted
        ]).to_csv(csv_path, index=False)
        dm_local = DataManager(str(csv_path), str(tmp_path / "trash"))
        indices = dm_local.filtrar_por_lazo(self.LASSO_WITH_POINTS)
        assert indices == [0]


# ── _calcular_ruta_destino ────────────────────────────────────────────────────

class TestCalculateDestinationPath:

    def test_path_with_images_preserves_intermediate_structure(self, dm, tmp_path):
        trash = dm.trash_path
        ruta = os.path.join(str(tmp_path), "proyecto", "images", "aves", "foto.jpg")
        carpeta, destino = dm._calcular_ruta_destino(ruta)
        assert carpeta == os.path.join(trash, "aves")
        assert destino == os.path.join(trash, "aves", "foto.jpg")

    def test_path_without_images_goes_to_trash_root(self, dm, tmp_path):
        trash = dm.trash_path
        ruta = os.path.join(str(tmp_path), "proyecto", "data", "foto.jpg")
        carpeta, destino = dm._calcular_ruta_destino(ruta)
        assert carpeta == trash
        assert destino == os.path.join(trash, "foto.jpg")
