import json
import re
from main import AdvancedPDFExtractor

def estimate_label(block):
    """Estimate heading level based on heuristics."""
    text = block["text"]
    font_size = block["font_size"]
    
   
    if font_size > 16 or re.match(r'^(CHAPTER|PART)\s+\d+', text) or text.isupper():
        return 1  
    elif font_size > 12 or re.match(r'^\d+\.\d+\s+', text) or re.match(r'^(Abstract|Introduction|Conclusion)', text):
        return 2  
    elif font_size > 10 or re.match(r'^\d+\.\d+\.\d+', text) or text.endswith(":"):
        return 3  
    else:
        return 0  


pdf_files = ["sample.pdf"] 


extractor = AdvancedPDFExtractor(model_path=None)  


data = []
for pdf_path in pdf_files:
    try:
        blocks = extractor.extract_blocks_advanced(pdf_path)
        for block in blocks:
        
            predicted_label = estimate_label(block)
           
            label = input(f"Label for '{block['text']}' (0=body, 1=H1, 2=H2, 3=H3) [Suggested: {predicted_label}]: ") or predicted_label
            features = extractor.build_feature_vector(block)
            row = {
                "text": block["text"],
                "embeddings": [float(v) for v in features[:-10].tolist()],  
                "font_size": float(block["font_size"]),
                "y0": float(block["bbox"][1]),
                "height": float(block["bbox"][3] - block["bbox"][1]),
                "width": float(block["bbox"][2] - block["bbox"][0]),
                "text_length": len(block["text"]),
                "flags": block.get("flags", 0),
                "is_upper": 1 if block["text"].isupper() else 0,
                "ends_colon": 1 if block["text"].endswith(":") else 0,
                "is_numbered": 1 if re.match(r"^\d+\.", block["text"]) else 0,
                "word_count": len(block["text"].split()),
                "label": int(label)
            }
            data.append(row)
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        continue

with open("training_data.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)
print("Training data saved to training_data.json")