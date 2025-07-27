import pypdf
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import joblib
import logging
from pathlib import Path
from typing import List, Dict, Optional
import re
import traceback
 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFOutlineExtractor:
    def __init__(self, model_path: str = "app/model/heading_classifier.joblib"):
        """Initialize the PDF outline extractor."""
        
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.model_data = self._load_model(model_path) if model_path else None
        self.classifier = self.model_data["classifier"] if self.model_data else None
        
    def _load_model(self, model_path: str):
        """Load the pre-trained classifier and model metadata."""
        try:
            model_data = joblib.load(model_path)
            logger.info(f"Loaded model with classes: {model_data['classes']}")
            logger.info(f"Embedding dimension: {model_data['embedding_dim']}")
            return model_data
        except (FileNotFoundError, EOFError) as e:
            logger.error(f"Model not found or corrupted at {model_path}: {e}")
            raise
    
    def extract_blocks(self, pdf_path: str) -> List[Dict]:
        """Extract text blocks with metadata from PDF using pypdf."""
        try:
            with open(pdf_path, 'rb') as file:
                reader = pypdf.PdfReader(file)
                blocks = []
                
                for page_num, page in enumerate(reader.pages, start=1):
                    try:
                        # Extract text with basic positioning info
                        text = page.extract_text()
                        if not text.strip():
                            continue
                            
                        # Split text into lines and process each
                        lines = text.split('\n')
                        y_position = 800  # Start from top of page (approximate)
                        
                        for line in lines:
                            line = line.strip()
                            if not line or len(line) < 2:
                                continue
                                
                            # Estimate font size and formatting from text characteristics
                            font_size = self._estimate_font_size(line)
                            
                            # Create approximate bounding box
                            bbox = [50, y_position, 550, y_position + font_size]
                            
                            blocks.append({
                                "text": line,
                                "font_size": font_size,
                                "font": "unknown",  # pypdf doesn't provide font info easily
                                "bbox": bbox,
                                "page": page_num,
                                "y0": y_position,  # Add y0 for consistency with training data
                                "flags": self._get_text_flags(line),
                                "color": 0
                            })
                            
                            y_position -= font_size + 2  # Move down for next line
                            
                    except Exception as e:
                        logger.warning(f"Error processing page {page_num}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Failed to open PDF: {e}")
            raise
            
        logger.info(f"Extracted {len(blocks)} text blocks from {pdf_path}")
        return blocks
    
    def _estimate_font_size(self, text: str) -> float:
        """Estimate font size based on text characteristics."""
        # Enhanced heuristics matching training data patterns
        if self._is_main_heading(text):
            return 22.0  # H1 level
        elif self._is_section_heading(text):
            return 18.0  # H2 level
        elif self._is_subsection_heading(text):
            return 16.0  # H3 level
        elif len(text) < 50 and text[0].isupper():
            return 14.0  # Potential heading
        else:
            return 12.0  # Body text
    
    def _is_main_heading(self, text: str) -> bool:
        """Check if text is a main heading (H1 level)."""
        patterns = [
            r'^(CHAPTER|PART)\s+\d+',
            r'^\d+\.\s+[A-Z].*[^.]$',  # "1. Introduction"
            r'^[A-Z\s]{10,}$',  # Long all-caps text
            r'^(Introduction|Conclusion|Abstract|Summary|References)$',
        ]
        return any(re.match(p, text, re.IGNORECASE) for p in patterns)
    
    def _is_section_heading(self, text: str) -> bool:
        """Check if text is a section heading (H2 level)."""
        patterns = [
            r'^\d+\.\d+\s+[A-Z]',  # "1.1 Section"
            r'^[A-Z][a-z]+(\s+[A-Z][a-z]+)*$',  # Title Case
            r'^\d+\.\d+\s+',  # Section numbering
        ]
        return any(re.match(p, text) for p in patterns)
    
    def _is_subsection_heading(self, text: str) -> bool:
        """Check if text is a subsection heading (H3 level)."""
        patterns = [
            r'^\d+\.\d+\.\d+\s+[A-Z]',  # "1.1.1 Subsection"
            r'^[a-z]+\)\s+[A-Z]',  # "a) Subsection"
            r'^[A-Z][a-z]+\s*:',  # "Method:"
        ]
        return any(re.match(p, text) for p in patterns)
    
    def _looks_like_heading(self, text: str) -> bool:
        """Check if text looks like a heading based on patterns."""
        return (self._is_main_heading(text) or 
                self._is_section_heading(text) or 
                self._is_subsection_heading(text))
    
    def _get_text_flags(self, text: str) -> int:
        """Generate flags based on text formatting characteristics."""
        flags = 0
        if text.isupper():
            flags |= 1  # All caps
        if self._looks_like_heading(text):
            flags |= 2  # Heading pattern
        if text.endswith(':'):
            flags |= 4  # Colon ending
        return flags
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text using SentenceTransformer (same as training)."""
        try:
            embedding = self.embedder.encode([text])[0]
            return embedding
        except Exception as e:
            logger.warning(f"Error generating embedding for text: {text[:50]}... Error: {e}")
            # Return zero vector with correct dimensions
            return np.zeros(self.model_data["embedding_dim"] if self.model_data else 384)
    
    def build_feature_vector(self, block: Dict) -> np.ndarray:
        """Build feature vector matching the training data format."""
        # Get text embedding
        text_embedding = self.get_embedding(block["text"])
        
        # Layout features (font_size, y0) - matching training script
        layout_features = np.array([
            block["font_size"],
            block["y0"]
        ])
        
        # Combine embeddings with layout features (same as training)
        feature_vector = np.hstack([text_embedding, layout_features])
        
        return feature_vector
    
    def predict_label(self, block: Dict) -> str:
        """Predict label for a text block."""
        if self.classifier is None:
            return "Body"  # Default if no classifier
            
        try:
            features = self.build_feature_vector(block)
            predicted_label = self.classifier.predict([features])[0]
            
            # Get prediction confidence
            probabilities = self.classifier.predict_proba([features])[0]
            confidence = max(probabilities)
            
            logger.debug(f"Predicted '{predicted_label}' with confidence {confidence:.3f} for text: {block['text'][:50]}...")
            
            return predicted_label
            
        except Exception as e:
            logger.warning(f"Error predicting label for block: {e}")
            return "Body"
    
    def post_process_outline(self, outline: List[Dict]) -> List[Dict]:
        """Post-process outline to fix hierarchical inconsistencies."""
        if not outline:
            return outline
            
        processed = []
        prev_level = 0
        
        for item in outline:
            # Handle different label formats
            if item["level"].startswith("H"):
                current_level = int(item["level"][1])  # Extract number from "H1", "H2", etc.
            else:
                # Fallback for other formats
                current_level = 1
            
            # Ensure hierarchical consistency
            if current_level > prev_level + 1:
                current_level = prev_level + 1
                item["level"] = f"H{current_level}"
                
            processed.append(item)
            prev_level = current_level
            
        return processed
    
    def build_outline(self, pdf_path: str, output_path: Optional[str] = None) -> Dict:
        """Build outline from PDF."""
        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        logger.info(f"Processing PDF: {pdf_path}")
        
        blocks = self.extract_blocks(pdf_path)
        if not blocks:
            logger.warning("No text blocks extracted from PDF")
            return {
                "title": "Empty Document", 
                "outline": [], 
                "metadata": {
                    "total_blocks": 0, 
                    "heading_blocks": 0, 
                    "pages": 0
                }
            }
        
        outline = []
        title = None
        heading_blocks = 0
        
        logger.info("Classifying text blocks...")
        for i, block in enumerate(blocks):
            try:
                predicted_label = self.predict_label(block)
                
                # Skip body text
                if predicted_label == "Body":
                    continue
                
                heading_blocks += 1
                
                # First H1 becomes the title
                if title is None and predicted_label == "H1":
                    title = block["text"]
                
                outline.append({
                    "level": predicted_label,
                    "text": block["text"],
                    "page": block["page"],
                    "font_size": block["font_size"],
                    "bbox": block["bbox"],
                    "y0": block["y0"]
                })
                
            except Exception as e:
                logger.warning(f"Error processing block {i}: {e}")
                continue
        
        # Post-process for consistency
        outline = self.post_process_outline(outline)
        
        result = {
            "title": title if title else Path(pdf_path).stem,
            "outline": outline,
            "metadata": {
                "total_blocks": len(blocks),
                "heading_blocks": heading_blocks,
                "pages": max(block["page"] for block in blocks) if blocks else 0,
                "pdf_filename": Path(pdf_path).name,
                "model_classes": list(self.model_data["classes"]) if self.model_data else []
            }
        }
        
        # Save output
        if output_path:
            try:
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                logger.info(f"Outline saved to: {output_path}")
            except Exception as e:
                logger.error(f"Failed to save output: {e}")
        
        return result
    
    def extract_toc_from_bookmarks(self, pdf_path: str) -> List[Dict]:
        """Extract table of contents from PDF bookmarks if available."""
        try:
            with open(pdf_path, 'rb') as file:
                reader = pypdf.PdfReader(file)
                
                if reader.outline:
                    logger.info(f"Found {len(reader.outline)} bookmark entries")
                    return self._parse_outline_recursive(reader.outline, level=1)
                else:
                    logger.info("No bookmarks found in PDF")
                    
        except Exception as e:
            logger.warning(f"Could not extract bookmarks: {e}")
        
        return []
    
    def _parse_outline_recursive(self, outline_items, level: int = 1) -> List[Dict]:
        """Recursively parse pypdf outline structure."""
        parsed_items = []
        
        for item in outline_items:
            if isinstance(item, list):
                # Nested outline items
                parsed_items.extend(self._parse_outline_recursive(item, level + 1))
            else:
                # Individual outline item
                try:
                    page_num = None
                    if hasattr(item, 'page') and item.page:
                        page_num = item.page.idnum if hasattr(item.page, 'idnum') else None
                    
                    parsed_items.append({
                        "level": f"H{level}",
                        "text": str(item.title) if hasattr(item, 'title') else str(item),
                        "page": page_num
                    })
                except Exception as e:
                    logger.warning(f"Error parsing outline item: {e}")
                    continue
        
        return parsed_items

class AdvancedPDFExtractor(PDFOutlineExtractor):
    """Enhanced extractor with better text extraction methods."""
    
    def __init__(self, model_path: str = None):
        super().__init__(model_path)
    
    def extract_blocks_advanced(self, pdf_path: str) -> List[Dict]:
        """Extract text blocks using enhanced methods."""
        try:
            with open(pdf_path, 'rb') as file:
                reader = pypdf.PdfReader(file)
                all_blocks = []
                
                for page_num, page in enumerate(reader.pages, start=1):
                    try:
                        blocks = self._extract_page_blocks_advanced(page, page_num)
                        all_blocks.extend(blocks)
                    except Exception as e:
                        logger.warning(f"Error processing page {page_num}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Failed to open PDF: {e}")
            raise
            
        logger.info(f"Extracted {len(all_blocks)} text blocks from {pdf_path}")
        return all_blocks
    
    def _extract_page_blocks_advanced(self, page, page_num: int) -> List[Dict]:
        """Extract text blocks from a single page with enhanced processing."""
        blocks = []
        
        # Enhanced text extraction
        text = page.extract_text()
        if not text.strip():
            return blocks
            
        # Split into paragraphs first, then lines
        paragraphs = re.split(r'\n\s*\n', text)
        y_position = 800
        
        for paragraph in paragraphs:
            lines = paragraph.split('\n')
            
            for line_text in lines:
                line_text = line_text.strip()
                if not line_text or len(line_text) < 2:
                    continue
                    
                # Enhanced font size estimation
                font_size = self._estimate_font_size_advanced(line_text)
                
                blocks.append({
                    "text": line_text,
                    "font_size": font_size,
                    "font": "unknown",
                    "bbox": [50, y_position, 550, y_position + font_size],
                    "page": page_num,
                    "y0": y_position,
                    "flags": self._get_text_flags(line_text),
                    "color": 0
                })
                
                y_position -= font_size + 2
            
            # Add extra space between paragraphs
            y_position -= 6
            
        return blocks
    
    def _estimate_font_size_advanced(self, text: str) -> float:
        """Advanced font size estimation with better heuristics."""
        text_len = len(text)
        word_count = len(text.split())
        
        # Check for heading patterns with more precision
        if self._is_main_heading(text):
            return 22.0  # H1
        elif self._is_section_heading(text):
            return 18.0  # H2
        elif self._is_subsection_heading(text):
            return 16.0  # H3
        elif text_len < 50 and text[0].isupper() and word_count <= 8:
            return 14.0  # Potential small heading
        else:
            return 12.0  # Body text

def compare_outlines(ml_outline: List[Dict], bookmark_outline: List[Dict]) -> Dict:
    """Compare ML-generated outline with bookmark-based outline."""
    comparison = {
        "ml_count": len(ml_outline),
        "bookmark_count": len(bookmark_outline),
        "common_headings": [],
        "ml_only": [],
        "bookmark_only": []
    }
    
    ml_texts = {item["text"].lower().strip() for item in ml_outline}
    bookmark_texts = {item["text"].lower().strip() for item in bookmark_outline}
    
    comparison["common_headings"] = list(ml_texts & bookmark_texts)
    comparison["ml_only"] = list(ml_texts - bookmark_texts)
    comparison["bookmark_only"] = list(bookmark_texts - ml_texts)
    
    return comparison

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract outline from PDF using pypdf and ML classification")
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("--output", "-o", default="outline.json", help="Output JSON file path")
    parser.add_argument("--model", "-m", default="app/model/heading_classifier.joblib", 
                       help="Path to classifier model")
    parser.add_argument("--compare-bookmarks", action="store_true",
                       help="Compare with PDF bookmarks if available")
    parser.add_argument("--advanced", action="store_true",
                       help="Use advanced text extraction methods")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        if not Path(args.pdf_path).exists():
            raise FileNotFoundError(f"PDF file not found: {args.pdf_path}")
        
        if args.advanced:
            extractor = AdvancedPDFExtractor(args.model)
        else:
            extractor = PDFOutlineExtractor(args.model)
            
        result = extractor.build_outline(args.pdf_path, args.output)
        
        print(json.dumps(result, indent=2))
        
        if args.compare_bookmarks:
            bookmark_outline = extractor.extract_toc_from_bookmarks(args.pdf_path)
            if bookmark_outline:
                comparison = compare_outlines(result["outline"], bookmark_outline)
                with open("comparison.json", "w", encoding="utf-8") as f:
                    json.dump(comparison, f, indent=2)
                print("\n--- Comparison with PDF Bookmarks ---")
                print(json.dumps(comparison, indent=2))
        
    except Exception as e:
        logger.error(f"Failed to process PDF: {e}")
        if args.debug:
            print(traceback.format_exc())
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())