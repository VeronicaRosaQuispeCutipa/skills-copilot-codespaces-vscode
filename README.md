# skills-copilot-codespaces-vscode

Aplicación web en **Streamlit** para análisis masivo de literatura científica.

## Funcionalidades

- Ingreso masivo de artículos por:
  - Texto libre (DOIs o títulos, uno por línea)
  - Archivo **CSV/Excel** con columna seleccionable
- Consulta de metadatos en APIs académicas abiertas:
  - Crossref (búsqueda por DOI/título)
  - Semantic Scholar (búsqueda por título)
- Extracción de:
  - Título oficial
  - Autores
  - Año de publicación
  - Abstract
- Análisis estructurado con LLM por artículo:
  - Resumen analítico
  - Métodos/metodologías
  - Métricas o resultados clave
  - Conclusiones
- Visualización en tabla interactiva y exportación a Excel.

## Ejecución

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Configuración del LLM

Para usar el análisis con LLM, configura `OPENAI_API_KEY`:

- En la barra lateral de la app, o
- En `secrets.toml` de Streamlit.

Si no hay API key, la app sigue procesando metadatos y marca el análisis LLM como no disponible.
