import io
import json
import re
from html import unescape
from typing import Any

import pandas as pd
import requests
import streamlit as st
from openai import OpenAI

DOI_REGEX = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)


def is_doi(value: str) -> bool:
    return bool(DOI_REGEX.search(value.strip()))


def clean_abstract(raw_abstract: str | None) -> str:
    if not raw_abstract:
        return ""
    text = re.sub(r"<[^>]+>", " ", raw_abstract)
    text = unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_authors(authors: list[dict[str, Any]] | None) -> str:
    if not authors:
        return ""
    names: list[str] = []
    for author in authors:
        given = author.get("given") or author.get("name") or ""
        family = author.get("family") or ""
        full_name = " ".join(part for part in [given, family] if part).strip()
        if full_name:
            names.append(full_name)
    return "; ".join(names)


def fetch_crossref_by_doi(doi: str) -> dict[str, Any]:
    url = f"https://api.crossref.org/works/{doi}"
    response = requests.get(url, timeout=20)
    if response.status_code != 200:
        return {}
    message = response.json().get("message", {})
    title = ""
    if isinstance(message.get("title"), list) and message.get("title"):
        title = message.get("title")[0]
    published = message.get("published-print") or message.get("published-online") or {}
    year = ""
    date_parts = published.get("date-parts") or []
    if date_parts and date_parts[0]:
        year = str(date_parts[0][0])

    return {
        "doi": message.get("DOI") or doi,
        "official_title": title,
        "authors": normalize_authors(message.get("author")),
        "year": year,
        "abstract": clean_abstract(message.get("abstract")),
    }


def fetch_semantic_scholar_by_title(title_query: str) -> dict[str, Any]:
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": title_query,
        "limit": 1,
        "fields": "title,authors,year,abstract,externalIds",
    }
    response = requests.get(url, params=params, timeout=20)
    if response.status_code != 200:
        return {}

    results = response.json().get("data", [])
    if not results:
        return {}

    paper = results[0]
    doi = (paper.get("externalIds") or {}).get("DOI", "")
    authors = "; ".join(author.get("name", "") for author in paper.get("authors", []) if author.get("name"))

    return {
        "doi": doi,
        "official_title": paper.get("title", ""),
        "authors": authors,
        "year": str(paper.get("year", "") or ""),
        "abstract": clean_abstract(paper.get("abstract")),
    }


def fetch_crossref_by_title(title_query: str) -> dict[str, Any]:
    url = "https://api.crossref.org/works"
    params = {"query.bibliographic": title_query, "rows": 1}
    response = requests.get(url, params=params, timeout=20)
    if response.status_code != 200:
        return {}
    items = response.json().get("message", {}).get("items", [])
    if not items:
        return {}

    item = items[0]
    title = item.get("title", [""])
    published = item.get("published-print") or item.get("published-online") or {}
    year = ""
    date_parts = published.get("date-parts") or []
    if date_parts and date_parts[0]:
        year = str(date_parts[0][0])

    return {
        "doi": item.get("DOI", ""),
        "official_title": title[0] if title else "",
        "authors": normalize_authors(item.get("author")),
        "year": year,
        "abstract": clean_abstract(item.get("abstract")),
    }


def fetch_metadata(entry: str) -> dict[str, Any]:
    value = entry.strip()
    if not value:
        return {}

    try:
        if is_doi(value):
            doi_match = DOI_REGEX.search(value)
            if not doi_match:
                return {}
            doi = doi_match.group(0)
            doi_data = fetch_crossref_by_doi(doi)
            if doi_data:
                return doi_data
        else:
            semantic_data = fetch_semantic_scholar_by_title(value)
            if semantic_data:
                return semantic_data
            crossref_data = fetch_crossref_by_title(value)
            if crossref_data:
                return crossref_data
    except requests.RequestException:
        return {}

    return {}


def analyze_abstract_with_llm(title: str, abstract: str, model: str) -> dict[str, str]:
    empty_response = {
        "resumen_analitico": "",
        "metodos": "",
        "metricas_resultados": "",
        "conclusiones": "",
    }

    if not abstract:
        return empty_response

    api_key = st.secrets.get("OPENAI_API_KEY") or st.session_state.get("OPENAI_API_KEY")
    if not api_key:
        return {
            **empty_response,
            "resumen_analitico": "No se pudo analizar con LLM: configura OPENAI_API_KEY en Streamlit secrets o en el campo lateral.",
        }

    client = OpenAI(api_key=api_key)
    prompt = f"""
Analiza técnicamente el siguiente abstract científico y responde en formato JSON válido con exactamente estas claves:
- resumen_analitico
- metodos
- metricas_resultados
- conclusiones

Instrucciones:
1. Usa español técnico, claro y conciso.
2. No inventes datos que no estén explícitos.
3. Si algún dato no está disponible, indica "No reportado".
4. Resume métodos, métricas/resultados cuantitativos y conclusiones clave.

Título: {title}
Abstract: {abstract}
"""

    try:
        completion = client.chat.completions.create(
            model=model,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "Eres un analista experto en literatura científica."},
                {"role": "user", "content": prompt},
            ],
        )
        content = completion.choices[0].message.content or "{}"
        parsed = json.loads(content)
        return {
            "resumen_analitico": str(parsed.get("resumen_analitico", "")),
            "metodos": str(parsed.get("metodos", "")),
            "metricas_resultados": str(parsed.get("metricas_resultados", "")),
            "conclusiones": str(parsed.get("conclusiones", "")),
        }
    except Exception as exc:
        return {
            **empty_response,
            "resumen_analitico": f"No se pudo analizar con LLM: {exc}",
        }


def parse_entries(text_input: str, uploaded_file) -> list[str]:
    entries: list[str] = []

    if text_input.strip():
        entries.extend(line.strip() for line in text_input.splitlines() if line.strip())

    if uploaded_file is not None:
        if uploaded_file.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        if not df.empty:
            chosen_column = st.selectbox(
                "Selecciona la columna que contiene DOIs o títulos:",
                options=list(df.columns),
            )
            entries.extend(df[chosen_column].dropna().astype(str).str.strip().tolist())

    unique_entries = list(dict.fromkeys(entry for entry in entries if entry))
    return unique_entries


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="analisis_literatura")
    return output.getvalue()


def main() -> None:
    st.set_page_config(page_title="Análisis masivo de literatura científica", layout="wide")
    st.title("Análisis masivo de literatura científica")
    st.write(
        "Ingresa DOIs/títulos manualmente o carga un archivo para consultar metadatos académicos y analizarlos con IA."
    )

    with st.sidebar:
        st.header("Configuración LLM")
        st.session_state["OPENAI_API_KEY"] = st.text_input(
            "OPENAI_API_KEY",
            type="password",
            help="Opcional: también puedes configurarla en secrets.toml",
        )
        llm_model = st.text_input("Modelo", value="gpt-4o-mini")

    text_input = st.text_area(
        "Pega una lista de DOIs o títulos (uno por línea):",
        height=220,
        placeholder="10.1038/nature12373\nA Survey on Transfer Learning",
    )
    uploaded_file = st.file_uploader("O sube un archivo CSV o Excel", type=["csv", "xlsx", "xls"])

    entries = parse_entries(text_input, uploaded_file)

    if st.button("Procesar artículos", type="primary"):
        if not entries:
            st.warning("No hay entradas para procesar.")
            st.stop()

        rows: list[dict[str, Any]] = []
        progress_bar = st.progress(0)

        for index, entry in enumerate(entries, start=1):
            metadata = fetch_metadata(entry)
            analysis = analyze_abstract_with_llm(
                title=metadata.get("official_title", ""),
                abstract=metadata.get("abstract", ""),
                model=llm_model,
            )

            row = {
                "entrada_original": entry,
                "doi": metadata.get("doi", ""),
                "titulo_oficial": metadata.get("official_title", ""),
                "autores": metadata.get("authors", ""),
                "anio": metadata.get("year", ""),
                "abstract": metadata.get("abstract", ""),
                "resumen_analitico": analysis.get("resumen_analitico", ""),
                "metodos": analysis.get("metodos", ""),
                "metricas_resultados": analysis.get("metricas_resultados", ""),
                "conclusiones": analysis.get("conclusiones", ""),
            }
            rows.append(row)
            progress_bar.progress(index / len(entries))

        result_df = pd.DataFrame(rows)
        st.success(f"Se procesaron {len(result_df)} artículos.")
        st.dataframe(result_df, use_container_width=True)

        excel_data = to_excel_bytes(result_df)
        st.download_button(
            label="Descargar matriz en Excel",
            data=excel_data,
            file_name="matriz_analisis_literatura.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


if __name__ == "__main__":
    main()
