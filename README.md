# skills-copilot-codespaces-vscode

Aplicación Streamlit para ingesta masiva de literatura científica con enriquecimiento de metadatos, análisis LLM opcional y exportación a Excel.

## Ejecutar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Funcionalidades

- Ingesta por lote de DOIs y/o títulos (una entrada por línea)
- Enriquecimiento automático de metadatos desde Crossref
- Análisis opcional con un endpoint LLM compatible con OpenAI Chat Completions
- Exportación de resultados a archivo `.xlsx`
