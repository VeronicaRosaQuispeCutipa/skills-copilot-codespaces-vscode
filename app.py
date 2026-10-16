import io
import json
import os
import re
import time
from typing import Dict, List, Optional

import pandas as pd
import requests
import streamlit as st


DOI_REGEX = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
DEFAULT_TIMEOUT = 20


def normalize_lines(text: str) -> List[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def detect_doi(value: str) -> Optional[str]:
    match = DOI_REGEX.search(value)
    return match.group(0) if match else None


def clean_abstract(value: Optional[str]) -> str:
    if not value:
        return ""
    cleaned = re.sub(r"<[^>]+>", " ", value)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def parse_crossref_item(item: Dict) -> Dict:
    title = ""
    titles = item.get("title") or []
    if titles:
        title = titles[0]

    authors = []
    for author in item.get("author") or []:
        given = author.get("given", "").strip()
        family = author.get("family", "").strip()
        full_name = " ".join([n for n in [given, family] if n])
        if full_name:
            authors.append(full_name)

    year = None
    issued = item.get("issued", {}).get("date-parts", [])
    if issued and issued[0]:
        year = issued[0][0]

    doi = item.get("DOI", "")
    abstract = clean_abstract(item.get("abstract"))

    return {
        "title": title,
        "authors": ", ".join(authors),
        "year": year,
        "abstract": abstract,
        "doi": doi,
        "source": "Crossref",
    }


def fetch_crossref_by_doi(doi: str, timeout: int) -> Optional[Dict]:
    url = f"https://api.crossref.org/works/{doi}"
    response = requests.get(url, timeout=timeout)
    if response.status_code != 200:
        return None
    payload = response.json()
    message = payload.get("message")
    if not message:
        return None
    return parse_crossref_item(message)


def fetch_crossref_by_title(title: str, timeout: int) -> Optional[Dict]:
    url = "https://api.crossref.org/works"
    response = requests.get(url, params={"query.title": title, "rows": 1}, timeout=timeout)
    if response.status_code != 200:
        return None
    items = (response.json().get("message") or {}).get("items") or []
    if not items:
        return None
    return parse_crossref_item(items[0])


def parse_semantic_scholar_item(item: Dict) -> Dict:
    title = item.get("title", "")
    authors = ", ".join([a.get("name", "") for a in item.get("authors") or [] if a.get("name")])
    year = item.get("year")
    abstract = clean_abstract(item.get("abstract", ""))
    external_ids = item.get("externalIds") or {}
    doi = external_ids.get("DOI", "")

    return {
        "title": title,
        "authors": authors,
        "year": year,
        "abstract": abstract,
        "doi": doi,
        "source": "Semantic Scholar",
    }


def fetch_semantic_scholar_by_doi(doi: str, timeout: int) -> Optional[Dict]:
    url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}"
    params = {"fields": "title,authors,year,abstract,externalIds"}
    response = requests.get(url, params=params, timeout=timeout)
    if response.status_code != 200:
        return None
    return parse_semantic_scholar_item(response.json())


def fetch_semantic_scholar_by_title(title: str, timeout: int) -> Optional[Dict]:
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": title,
        "limit": 1,
        "fields": "title,authors,year,abstract,externalIds",
    }
    response = requests.get(url, params=params, timeout=timeout)
    if response.status_code != 200:
        return None
    data = response.json().get("data") or []
    if not data:
        return None
    return parse_semantic_scholar_item(data[0])


def merge_metadata(primary: Optional[Dict], secondary: Optional[Dict]) -> Dict:
    primary = primary or {}
    secondary = secondary or {}

    def pick(key: str):
        return primary.get(key) or secondary.get(key) or ""

    source = primary.get("source") or secondary.get("source") or ""
    if primary and secondary and primary.get("source") != secondary.get("source"):
        source = f"{primary.get('source')} + {secondary.get('source')}"

    return {
        "title": pick("title"),
        "authors": pick("authors"),
        "year": pick("year"),
        "abstract": pick("abstract"),
        "doi": pick("doi"),
        "source": source,
    }


def fetch_article_metadata(entry: str, timeout: int) -> Dict:
    doi = detect_doi(entry)

    crossref = None
    semantic = None

    try:
        crossref = fetch_crossref_by_doi(doi, timeout) if doi else fetch_crossref_by_title(entry, timeout)
    except requests.RequestException:
        crossref = None

    try:
        semantic = fetch_semantic_scholar_by_doi(doi, timeout) if doi else fetch_semantic_scholar_by_title(entry, timeout)
    except requests.RequestException:
        semantic = None

    merged = merge_metadata(crossref, semantic)

    return {
        "input": entry,
        "detected_doi": doi or "",
        "title": merged.get("title", ""),
        "authors": merged.get("authors", ""),
        "year": merged.get("year", ""),
        "abstract": merged.get("abstract", ""),
        "source": merged.get("source", ""),
    }


def parse_json_response(content: str) -> Optional[Dict]:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
    return None


def analyze_abstract_with_llm(title: str, abstract: str, timeout: int) -> Dict:
    if not abstract:
        return {
            "summary": "No se encontró abstract para analizar.",
            "methods": "No disponible",
            "metrics": "No disponible",
            "conclusions": "No disponible",
        }

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")

    if not api_key:
        return {
            "summary": "LLM no configurado (defina OPENAI_API_KEY).",
            "methods": "LLM no configurado",
            "metrics": "LLM no configurado",
            "conclusions": "LLM no configurado",
        }

    system_prompt = (
        "Eres un analista de literatura científica. Devuelve SOLO JSON válido con las claves: "
        "summary, methods, metrics, conclusions. "
        "Cada valor debe estar en español técnico y ser breve pero preciso."
    )

    user_prompt = (
        "Analiza este artículo y clasifica la información:\n"
        f"Título: {title}\n"
        f"Abstract: {abstract}\n\n"
        "Requisitos:\n"
        "1) summary: resumen analítico estructurado del estudio.\n"
        "2) methods: métodos/metodologías de investigación.\n"
        "3) metrics: desempeños técnicos, métricas o resultados cuantitativos clave.\n"
        "4) conclusions: conclusiones principales."
    )

    payload = {
        "model": model,
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    headers = {
        "Authorization": f"******",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        parsed = parse_json_response(content)
        if not parsed:
            raise ValueError("Respuesta JSON inválida del LLM")

        return {
            "summary": parsed.get("summary", "No disponible"),
            "methods": parsed.get("methods", "No disponible"),
            "metrics": parsed.get("metrics", "No disponible"),
            "conclusions": parsed.get("conclusions", "No disponible"),
        }
    except (requests.RequestException, KeyError, ValueError):
        return {
            "summary": "No fue posible analizar el abstract con el LLM.",
            "methods": "Error de análisis",
            "metrics": "Error de análisis",
            "conclusions": "Error de análisis",
        }


def parse_uploaded_file(uploaded_file) -> pd.DataFrame:
    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="analisis_tecnico")
    buffer.seek(0)
    return buffer.getvalue()


def main() -> None:
    st.set_page_config(page_title="Análisis Masivo de Literatura", layout="wide")
    st.title("Análisis masivo de literatura científica")
    st.write(
        "Ingresa DOIs/títulos manualmente o carga un archivo CSV/Excel para extraer metadatos y analizar abstracts con IA."
    )

    with st.expander("Configuración", expanded=False):
        timeout = st.number_input("Timeout de solicitudes (segundos)", min_value=5, max_value=120, value=DEFAULT_TIMEOUT)
        delay_seconds = st.number_input("Pausa entre artículos (segundos)", min_value=0.0, max_value=5.0, value=0.0, step=0.1)
        st.caption("Configura OPENAI_API_KEY (y opcionalmente OPENAI_MODEL/OPENAI_BASE_URL) para activar el análisis con LLM.")

    input_mode = st.radio("Modo de ingreso", ["Pegar DOIs/títulos", "Subir archivo CSV/Excel"], horizontal=True)

    entries: List[str] = []

    if input_mode == "Pegar DOIs/títulos":
        text_blob = st.text_area(
            "Pega una lista (uno por línea)",
            height=260,
            placeholder="10.1038/s41586-020-2649-2\nA survey on transfer learning",
        )
        entries = normalize_lines(text_blob)
    else:
        uploaded_file = st.file_uploader("Sube un archivo CSV o Excel", type=["csv", "xlsx", "xls"])
        if uploaded_file:
            try:
                df_input = parse_uploaded_file(uploaded_file)
                st.write("Vista previa del archivo:")
                st.dataframe(df_input.head(10), use_container_width=True)
                selected_column = st.selectbox("Selecciona la columna con DOI/título", options=list(df_input.columns))
                entries = [str(v).strip() for v in df_input[selected_column].dropna().tolist() if str(v).strip()]
            except Exception as exc:
                st.error(f"No se pudo leer el archivo: {exc}")

    if st.button("Procesar artículos", type="primary"):
        if not entries:
            st.warning("No hay entradas para procesar.")
            return

        records = []
        progress = st.progress(0)
        status_box = st.empty()

        for idx, entry in enumerate(entries, start=1):
            status_box.info(f"Procesando {idx}/{len(entries)}: {entry[:120]}")
            metadata = fetch_article_metadata(entry, int(timeout))
            analysis = analyze_abstract_with_llm(metadata.get("title", entry), metadata.get("abstract", ""), int(timeout))

            records.append(
                {
                    "Entrada": metadata.get("input", entry),
                    "DOI detectado": metadata.get("detected_doi", ""),
                    "Título oficial": metadata.get("title", ""),
                    "Autores": metadata.get("authors", ""),
                    "Año": metadata.get("year", ""),
                    "Abstract completo": metadata.get("abstract", ""),
                    "Resumen analítico estructurado": analysis.get("summary", ""),
                    "Métodos y metodologías": analysis.get("methods", ""),
                    "Desempeños técnicos / métricas": analysis.get("metrics", ""),
                    "Conclusiones principales": analysis.get("conclusions", ""),
                    "Fuente de metadatos": metadata.get("source", ""),
                }
            )

            progress.progress(idx / len(entries))
            if delay_seconds > 0:
                time.sleep(delay_seconds)

        status_box.success("Procesamiento completado.")

        result_df = pd.DataFrame(records)
        st.subheader("Matriz consolidada de análisis técnico")
        st.dataframe(result_df, use_container_width=True)

        excel_data = to_excel_bytes(result_df)
        st.download_button(
            "Descargar matriz en Excel",
            data=excel_data,
            file_name="matriz_analisis_literatura.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


if __name__ == "__main__":
    main()
