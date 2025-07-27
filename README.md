# adobe_round1A_submission


🧠 PDF Outline Extractor
A machine learning-powered Python tool to extract semantic outlines from PDF documents.
This utility uses both traditional PDF parsing techniques and ML-based heading classification to generate a structured document outline.

📘 Overview
This tool extracts text from PDF files and intelligently detects headings (H1, H2, H3) to build a document outline. It optionally compares the ML-generated outline to the document's native bookmarks (TOC) if available.

The solution combines:

Heuristics for font size estimation and layout features

Transformer-based sentence embeddings

A trained scikit-learn classifier

Optionally, it can fall back to using PDF bookmarks.

🔍 Features
Extracts headings and body content from PDF files.

Predicts heading level (H1, H2, H3) using a machine learning model.

Embeds semantic features using SentenceTransformer.

Supports a basic and an advanced text extraction mode.

Optional comparison with PDF bookmarks.

Outputs clean, hierarchical JSON structure.

🧪 Machine Learning
The model is trained to classify text blocks into heading levels using:

Text embeddings from sentence-transformers (all-MiniLM-L6-v2)

Layout metadata like font size and vertical position (y0)

A scikit-learn classifier (e.g., RandomForest or similar) saved via joblib.

Input Features:
384-dimensional text embedding vector

Font size

Vertical position on the page (y0)

📦 Dependencies
pip install pypdf sentence-transformers numpy scikit-learn joblib


🧰 Project Structure
adobe_submission/
│
├── .gitignore
├── requirements.txt
├── README.md            
│
├── train_model.py
├── create_training_data.py
├── classify.py
├── main.py              ✅ (entry point)
│
├── sample.pdf           ✅ (sample input)
├── outline.json         ✅ (sample output)
│
├── app/
│   ├── model/
│   │   ├── heading_classifier.joblib
│   │   ├── inference_config.json
│   │   ├── evaluation_results.json
│   └── ...
│
├── dockerfile
└── venv/  
build command
docker build -t adobe-submission .
run commmand 
docker run adobe-submission

