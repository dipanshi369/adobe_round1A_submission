import fitz  # PyMuPDF
import json
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.linear_model import LogisticRegression
import joblib

# -------------------- Text Block Extraction --------------------
def extract_blocks(pdf_path):
    doc = fitz.open(pdf_path)
    blocks = []

    for page_num, page in enumerate(doc, start=1):
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span["text"].strip()
                    if text:
                        blocks.append({
                            "text": text,
                            "font_size": span["size"],
                            "font": span["font"],
                            "bbox": span["bbox"],  # [x0, y0, x1, y1]
                            "page": page_num
                        })

    return blocks

# -------------------- Load Transformer Model --------------------
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

model = AutoModel.from_pretrained(
    "distilbert-base-uncased",
    device_map="cpu"
    # remove load_in_8bit=True unless you specifically use bitsandbytes & GPU
)

# -------------------- Embedding Generation --------------------
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding

# -------------------- Feature Vector Construction --------------------
def build_feature_vector(block):
    embedding = get_embedding(block["text"])
    layout_feats = np.array([
        block["font_size"],
        block["bbox"][1],  # Y0 position (vertical)
    ])
    return np.concatenate([embedding, layout_feats])

# -------------------- Load Classifier --------------------
clf = joblib.load("heading_classifier.joblib")

# -------------------- Outline Building --------------------
def build_outline(pdf_path):
    blocks = extract_blocks(pdf_path)
    outline = []
    title = None

    for block in blocks:
        features = build_feature_vector(block)
        level = clf.predict([features])[0]

        if level == 0:
            continue

        if title is None and level == 1:
            title = block["text"]

        outline.append({
            "level": f"H{level}",
            "text": block["text"],
            "page": block["page"]
        })

    result = {
        "title": title if title else "Unknown Title",
        "outline": outline
    }

    with open("output.json", "w") as f:
        json.dump(result, f, indent=2)

    return result

# -------------------- Entry Point --------------------
if __name__ == "__main__":
    result = build_outline("sample.pdf")
    print(json.dumps(result, indent=2))
