import io
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import streamlit as st

DOI_PATTERN = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
CROSSREF_API_BASE = "https://api.crossref.org/works"
OPENAI_CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"


@dataclass
class RecordInput:
    raw: str
    kind: str
    value: str


def normalize_text(text: Optional[str]) -> str:
    if not text:
        return ""
    compact = re.sub(r"<[^>]+>", " ", text)
    compact = re.sub(r"\s+", " ", compact)
    return compact.strip()


def parse_input_lines(block: str) -> List[RecordInput]:
    records: List[RecordInput] = []
    for line in block.splitlines():
        raw = line.strip()
        if not raw:
            continue

        doi_match = DOI_PATTERN.search(raw)
        if doi_match:
            records.append(RecordInput(raw=raw, kind="doi", value=doi_match.group(0)))
        else:
            records.append(RecordInput(raw=raw, kind="title", value=raw))
    return records


def call_crossref_for_doi(doi: str, timeout: int = 20) -> Dict[str, Any]:
    encoded_doi = requests.utils.quote(doi, safe="")
    response = requests.get(f"{CROSSREF_API_BASE}/{encoded_doi}", timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    return payload.get("message", {})


def call_crossref_for_title(title: str, timeout: int = 20) -> Dict[str, Any]:
    response = requests.get(
        CROSSREF_API_BASE,
        params={"query.title": title, "rows": 1, "sort": "relevance"},
        timeout=timeout,
    )
    response.raise_for_status()
    items = response.json().get("message", {}).get("items", [])
    if not items:
        raise ValueError("No se encontró coincidencia en Crossref")
    return items[0]


def format_authors(authors: Optional[List[Dict[str, Any]]]) -> str:
    if not authors:
        return ""
    formatted = []
    for author in authors:
        given = author.get("given", "")
        family = author.get("family", "")
        full_name = f"{given} {family}".strip()
        if full_name:
            formatted.append(full_name)
    return "; ".join(formatted)


def extract_year(metadata: Dict[str, Any]) -> Optional[int]:
    date_parts = metadata.get("issued", {}).get("date-parts", [])
    if date_parts and date_parts[0]:
        year = date_parts[0][0]
        if isinstance(year, int):
            return year
    return None


def metadata_to_row(source: RecordInput, metadata: Dict[str, Any]) -> Dict[str, Any]:
    title_parts = metadata.get("title", [])
    title = title_parts[0] if title_parts else ""

    return {
        "input_original": source.raw,
        "tipo_input": source.kind,
        "query": source.value,
        "doi": metadata.get("DOI", ""),
        "titulo": title,
        "revista": metadata.get("container-title", [""])[0] if metadata.get("container-title") else "",
        "anio": extract_year(metadata),
        "autores": format_authors(metadata.get("author")),
        "publisher": metadata.get("publisher", ""),
        "url": metadata.get("URL", ""),
        "resumen_fuente": normalize_text(metadata.get("abstract", "")),
        "estado": "ok",
        "error": "",
    }


def fetch_metadata(source: RecordInput) -> Dict[str, Any]:
    try:
        metadata = (
            call_crossref_for_doi(source.value)
            if source.kind == "doi"
            else call_crossref_for_title(source.value)
        )
        return metadata_to_row(source, metadata)
    except Exception as exc:  # pylint: disable=broad-except
        return {
            "input_original": source.raw,
            "tipo_input": source.kind,
            "query": source.value,
            "doi": "",
            "titulo": "",
            "revista": "",
            "anio": None,
            "autores": "",
            "publisher": "",
            "url": "",
            "resumen_fuente": "",
            "estado": "error",
            "error": str(exc),
        }


def build_llm_prompt(row: Dict[str, Any], instructions: str) -> str:
    base_context = {
        "title": row.get("titulo", ""),
        "journal": row.get("revista", ""),
        "year": row.get("anio"),
        "authors": row.get("autores", ""),
        "abstract": row.get("resumen_fuente", ""),
        "doi": row.get("doi", ""),
    }
    context_json = json.dumps(base_context, ensure_ascii=False, indent=2)

    return (
        "Analiza el siguiente registro bibliográfico y responde SOLO con JSON válido. "
        "Incluye exactamente estas claves: resumen_llm, palabras_clave, categoria_tematica, relevancia. "
        "`palabras_clave` debe ser una lista de strings y `relevancia` un número entre 0 y 100.\n\n"
        f"Instrucciones adicionales del usuario: {instructions or 'Ninguna'}\n\n"
        f"Contexto:\n{context_json}"
    )


def safe_parse_llm_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    payload = json.loads(text)

    return {
        "resumen_llm": str(payload.get("resumen_llm", "")),
        "palabras_clave": ", ".join(payload.get("palabras_clave", [])) if isinstance(payload.get("palabras_clave"), list) else str(payload.get("palabras_clave", "")),
        "categoria_tematica": str(payload.get("categoria_tematica", "")),
        "relevancia": payload.get("relevancia", ""),
    }


def call_openai_compatible_chat(
    api_key: str,
    model: str,
    prompt: str,
    timeout: int = 60,
) -> str:
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + api_key,
    }
    payload = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": "Eres un analista de literatura científica."},
            {"role": "user", "content": prompt},
        ],
    }

    response = requests.post(OPENAI_CHAT_COMPLETIONS_URL, headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()
    body = response.json()
    return body["choices"][0]["message"]["content"]


def analyze_rows_with_llm(
    rows: List[Dict[str, Any]],
    enable_llm: bool,
    api_key: str,
    model: str,
    custom_instructions: str,
) -> List[Dict[str, Any]]:
    if not enable_llm:
        for row in rows:
            row.update(
                {
                    "resumen_llm": "",
                    "palabras_clave": "",
                    "categoria_tematica": "",
                    "relevancia": "",
                    "error_llm": "",
                }
            )
        return rows

    enriched: List[Dict[str, Any]] = []
    progress = st.progress(0, text="Analizando con LLM...")

    for index, row in enumerate(rows, start=1):
        if row.get("estado") != "ok":
            row.update(
                {
                    "resumen_llm": "",
                    "palabras_clave": "",
                    "categoria_tematica": "",
                    "relevancia": "",
                    "error_llm": "Registro inválido por error previo",
                }
            )
            enriched.append(row)
            progress.progress(index / max(len(rows), 1), text=f"Analizando con LLM... {index}/{len(rows)}")
            continue

        try:
            prompt = build_llm_prompt(row, custom_instructions)
            raw_answer = call_openai_compatible_chat(
                api_key=api_key,
                model=model,
                prompt=prompt,
            )
            parsed = safe_parse_llm_json(raw_answer)
            row.update(parsed)
            row["error_llm"] = ""
        except Exception as exc:  # pylint: disable=broad-except
            row.update(
                {
                    "resumen_llm": "",
                    "palabras_clave": "",
                    "categoria_tematica": "",
                    "relevancia": "",
                    "error_llm": str(exc),
                }
            )

        enriched.append(row)
        progress.progress(index / max(len(rows), 1), text=f"Analizando con LLM... {index}/{len(rows)}")

    return enriched


def rows_to_excel_bytes(rows: List[Dict[str, Any]]) -> bytes:
    buffer = io.BytesIO()
    df = pd.DataFrame(rows)

    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="literatura")

    return buffer.getvalue()


def run() -> None:
    st.set_page_config(page_title="Ingesta científica + LLM", layout="wide")
    st.title("📚 Ingesta masiva de literatura científica")
    st.caption("Carga DOIs o títulos, enriquece metadatos, analiza con LLM y exporta a Excel.")

    with st.sidebar:
        st.header("Configuración LLM")
        enable_llm = st.checkbox("Activar análisis LLM", value=False)
        model = st.text_input("Modelo", value="gpt-4o-mini")
        api_key = st.text_input("API Key", type="password")
        custom_instructions = st.text_area(
            "Instrucciones de análisis",
            value="Evalúa la relevancia para revisión de literatura y sintetiza aportes clave.",
        )

    st.subheader("Entradas")
    input_text = st.text_area(
        "Pega una lista (una línea por DOI o título):",
        placeholder="10.1038/s41586-020-2649-2\nDeep learning for medical image analysis",
        height=220,
    )

    process_clicked = st.button("Procesar lote", type="primary")

    if not process_clicked:
        return

    records = parse_input_lines(input_text)
    if not records:
        st.warning("No se detectaron entradas válidas.")
        return

    if enable_llm and (not api_key.strip() or not model.strip()):
        st.error("Para el análisis LLM debes completar API Key y modelo.")
        return

    st.info(f"Procesando {len(records)} registros...")
    progress = st.progress(0, text="Consultando Crossref...")

    rows: List[Dict[str, Any]] = []
    for idx, record in enumerate(records, start=1):
        rows.append(fetch_metadata(record))
        progress.progress(idx / len(records), text=f"Consultando Crossref... {idx}/{len(records)}")

    rows = analyze_rows_with_llm(
        rows=rows,
        enable_llm=enable_llm,
        api_key=api_key,
        model=model,
        custom_instructions=custom_instructions,
    )

    df = pd.DataFrame(rows)
    st.subheader("Resultados")
    st.dataframe(df, use_container_width=True)

    success_count = int((df["estado"] == "ok").sum()) if "estado" in df.columns else 0
    llm_ok_count = int((df["error_llm"] == "").sum()) if "error_llm" in df.columns else 0
    st.metric("Registros con metadata ok", success_count)
    if enable_llm:
        st.metric("Registros analizados por LLM sin error", llm_ok_count)

    excel_bytes = rows_to_excel_bytes(rows)
    st.download_button(
        label="📥 Descargar Excel",
        data=excel_bytes,
        file_name="literatura_enriquecida.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


if __name__ == "__main__":
    run()
