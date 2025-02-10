# -- Información del proyecto ------------------------------------------------

project = "DeepSampler"
author = "Diego Izaguirre, Fabian Prado, Fernando Candia"
release = "0.0.1"
# Para versiones mayores, también se puede definir 'version' (p.ej.: "0.0")

# -- Configuración general ---------------------------------------------------

# Extensiones de Sphinx que se usarán
extensions = [
    "sphinx.ext.autodoc",  # Para extraer documentación de los docstrings.
    "sphinx.ext.napoleon",  # Para soportar docstrings en estilo Google o NumPy.
    "sphinx.ext.viewcode",  # Añade enlaces al código fuente documentado.
]

# Rutas a los templates, si se desea personalizar la apariencia
templates_path = ["_templates"]

# Patrones que se deben excluir al buscar documentos
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Opciones para la salida HTML --------------------------------------------

autodoc_member_order = "bysource"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
