import os
import base64
import json
import fitz  # PyMuPDF
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
import time

start_time = time.time()

# ---------- Configuration ----------
# pdf_path = "Sample4.pdf"
images_dir = "Images"
os.makedirs(images_dir, exist_ok=True)

AZURE_DOC_INTEL_ENDPOINT = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
AZURE_DOC_INTEL_KEY = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_ENDPOINT_URL")
AZURE_OPENAI_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_VERSION = "2024-08-01-preview"
DEPLOYMENT_NAME = os.getenv("GPT4O_MODEL_NAME")

# ---------- Initialize Clients ----------
doc_client = DocumentIntelligenceClient(
    endpoint=AZURE_DOC_INTEL_ENDPOINT,
    credential=AzureKeyCredential(AZURE_DOC_INTEL_KEY)
)

openai_client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    api_version=OPENAI_API_VERSION,
    base_url=f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{DEPLOYMENT_NAME}"
)

# ---------- Helper: Extract figure image ----------
def extract_figure_as_image(pdf_path, page_number, polygon, method, fig_index):
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    image_name = f"{method}_page{page_number}_fig{fig_index}.png"
    output_path = os.path.join(images_dir, image_name)

    doc = fitz.open(pdf_path)
    page = doc[page_number - 1]
    x_points = polygon[0::2]
    y_points = polygon[1::2]
    rect = fitz.Rect(min(x_points)*72, min(y_points)*72, max(x_points)*72, max(y_points)*72) & page.rect

    pix = page.get_pixmap(clip=rect, dpi=300)
    pix.save(output_path)
    return output_path

# ---------- Helper: Describe image with Azure OpenAI ----------
def describe_image(image_path):
    with open(image_path, "rb") as img_file:
        base64_image = base64.b64encode(img_file.read()).decode("utf-8")
    image_data_url = f"data:image/png;base64,{base64_image}"

    response = openai_client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[
            {"role": "system", "content": "You are an AI that analyzes any uploaded figure and extracts all information."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze the image fully and provide a detailed textual report. "
    "If it contains a chart or graph, describe the axes and scale, and list all data points year by year (or label by label), "
    "including Value, Growth Rate (%) if visible, and approximate X/Y coordinates for each plotted point. "
    "Distinguish between current and projected data if applicable, and note shading, colors, or highlights. "
    "Describe trends, patterns, and any significant changes over time. "
    "If it contains a diagram, map, or other figure, describe all objects with labels, positions, and relationships. "
    "Output should be structured, readable text only — no tables, Markdown, or JSON."},
                    {"type": "image_url", "image_url": {"url": image_data_url}}
                ]
            }
        ],
        max_tokens=500
    )
    return response.choices[0].message.content

# ---------- Helper: Clean unwanted keys ----------
def clean_data(data):
    unwanted_keys = {
        "polygon", "words", "lines", "spans", "boundingRegions",
        "confidence", "span", "angle", "width", "height", "unit"
    }
    if isinstance(data, dict):
        for key in list(data.keys()):
            if key in unwanted_keys:
                data.pop(key)
        for v in data.values():
            clean_data(v)
    elif isinstance(data, list):
        for item in data:
            clean_data(item)

# ---------- Fallback Image Extraction ----------
def extract_fallback_images(pdf_path, page_num):
    doc = fitz.open(pdf_path)
    page = doc[page_num - 1]
    image_list = page.get_images(full=True)

    fallback_images = []
    if image_list:
        for image_index, img in enumerate(image_list, start=1):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            image_name = f"pymupdf_page{page_num}_fig{image_index}.{image_ext}"
            image_path = os.path.join(images_dir, image_name)
            with open(image_path, "wb") as image_file:
                image_file.write(image_bytes)

            try:
                description = describe_image(image_path)
            except Exception as e:
                description = f"Failed to describe fallback image: {e}"

            # --- Match structure with Document Intelligence ---
            fallback_images.append({
                "caption": f"Extracted raw image {image_index} from page {page_num}",
                "caption_paragraphs": [],
                "elements": [],
                "description": description,
                "image_file": image_path,
                "method": "pymupdf"
            })

    return fallback_images

# ---------- Extract structured data + figures ----------
def extract_page_summary(pdf_path,result):
    paragraph_lookup = {idx: p for idx, p in enumerate(result.paragraphs)} if hasattr(result, "paragraphs") else {}
    pages_summary = []

    for page in result.pages:
        page_num = page.page_number
        page_text = " ".join(line.content for line in page.lines)
        headings, subheadings, footers = [], [], []

        for idx, para in paragraph_lookup.items():
            if para.bounding_regions and any(br.page_number == page_num for br in para.bounding_regions):
                if getattr(para, "role", "") == "sectionHeading":
                    headings.append(para.content)
                elif getattr(para, "role", "") == "subHeading":
                    subheadings.append(para.content)
                elif getattr(para, "role", "") == "footer":
                    footers.append(para.content)

        handwritten_present = any(style.is_handwritten for style in result.styles)

        # Extract tables
        tables = []
        for table in getattr(result, "tables", []):
            if table.bounding_regions and any(br.page_number == page_num for br in table.bounding_regions):
                tbl = {
                    "caption": table.caption.content if getattr(table, "caption", None) else "",
                    "rowCount": table.row_count,
                    "columnCount": table.column_count,
                    "cells": [
                        {"rowIndex": c.row_index, "columnIndex": c.column_index, "content": c.content, "kind": getattr(c, "kind", None)}
                        for c in table.cells
                    ]
                }
                tables.append(tbl)

        # Extract figures or fallback images
        figures = []
        figures_found = False

        for idx, fig in enumerate(getattr(result, "figures", []), start=1):
            if fig.bounding_regions and any(br.page_number == page_num for br in fig.bounding_regions):
                figures_found = True
                br = fig.bounding_regions[0]

                image_path = extract_figure_as_image(pdf_path, br.page_number, br.polygon, "docintel", idx)

                try:
                    description = describe_image(image_path)
                except Exception as e:
                    description = f"Failed to describe image: {e}"

                figures.append({
                    "caption": fig.caption.content if getattr(fig, "caption", None) else "",
                    "caption_paragraphs": [],
                    "elements": [],
                    "description": description,
                    "image_file": image_path,
                    "method": "docintel"
                })

        if not figures_found:
            print(f"[!] No figure bounds for page {page_num}, using fallback images...")
            fallback_figures = extract_fallback_images(pdf_path, page_num)
            if fallback_figures:
                figures.extend(fallback_figures)
            else:
                print(f"[!] No figures found in fallback for page {page_num}")

        page_summary = {
            "pageNumber": page_num,
            "handwrittenContent": handwritten_present,
            "content": page_text,
            "headings": headings,
            "subheadings": subheadings,
            "footers": footers,
            "tables": tables,
            "figures": figures
        }
        pages_summary.append(page_summary)

    return pages_summary


def analyze_pdf_with_docintel(pdf_path, output_path="structured_output_with_Document.json"):
    """
    Analyze a PDF using Azure Document Intelligence (prebuilt-layout model),
    process results, and save structured summary to a JSON file.

    Args:
        pdf_path (str): Path to the PDF file.
        doc_client (DocumentIntelligenceClient): Azure Document Intelligence client.
        output_path (str): Path to save the structured JSON output.

    Returns:
        dict: The extracted summary dictionary.
    """

    start_time = time.time()

    try:
        # --- Analyze PDF using Azure Document Intelligence ---
        with open(pdf_path, "rb") as f:
            poller = doc_client.begin_analyze_document("prebuilt-layout", f)
            result = poller.result()

        # --- Convert result to dictionary ---
        result_dict = result.to_dict() if hasattr(result, "to_dict") else result.__dict__

        # --- Clean & Extract Summary ---
        clean_data(result_dict)  # assuming you already have this function
        summary = extract_page_summary(pdf_path,result)  # assuming you already have this function
        # wrapped_summary = {"document": summary}

        # --- Save results to JSON ---
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)

        print(f"Structured output with figure descriptions saved to '{output_path}'")

    except Exception as e:
        print(f"Error analyzing PDF {pdf_path}: {e}")
        summary = {}

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

    return summary