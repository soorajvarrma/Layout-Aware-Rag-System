"""
Layout-aware PDF parser for academic papers
Handles multi-column layouts, tables, and figures
"""

import fitz  # PyMuPDF
import pdfplumber
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
import json

@dataclass
class DocumentChunk:
    """Structured chunk with metadata"""
    content: str
    page_num: int
    chunk_type: str  # 'text', 'table', 'figure'
    bbox: tuple  # Bounding box (x0, y0, x1, y1)
    metadata: Dict[str, Any]
    
    def to_dict(self):
        return {
            'content': self.content,
            'page_num': self.page_num,
            'chunk_type': self.chunk_type,
            'bbox': self.bbox,
            'metadata': self.metadata
        }


class LayoutAwarePDFParser:
    """
    Extracts content from academic PDFs with layout awareness
    """
    
    def __init__(self, pdf_path: Path):
        self.pdf_path = pdf_path
        self.chunks: List[DocumentChunk] = []
        
    def parse(self) -> List[DocumentChunk]:
        """Main parsing pipeline"""
        print(f"Parsing PDF: {self.pdf_path}")
        
        # Extract text with layout awareness
        self._extract_text_with_layout()
        
        # Extract tables
        self._extract_tables()
        
        # Extract figures (metadata/captions)
        self._extract_figures()
        
        print(f"✅ Extracted {len(self.chunks)} chunks")
        return self.chunks
    
    def _extract_text_with_layout(self):
        """
        Extract text respecting multi-column layout
        Uses PyMuPDF's block detection for reading order
        """
        doc = fitz.open(self.pdf_path)
        
        for page_num, page in enumerate(doc, start=1):
            # Get text blocks with position info
            blocks = page.get_text("dict")["blocks"]
            
            # Sort blocks by position (top to bottom, left to right for columns)
            text_blocks = [b for b in blocks if b.get("type") == 0]  # Text blocks only
            
            # Sort by Y position first (top to bottom), then X (left to right)
            # This handles two-column layout
            sorted_blocks = sorted(text_blocks, key=lambda b: (b["bbox"][1], b["bbox"][0]))
            
            # Group into columns if needed
            columns = self._detect_columns(sorted_blocks, page.rect.width)
            
            for col_blocks in columns:
                for block in col_blocks:
                    text_content = self._extract_block_text(block)
                    
                    if text_content.strip():
                        chunk = DocumentChunk(
                            content=text_content,
                            page_num=page_num,
                            chunk_type='text',
                            bbox=block["bbox"],
                            metadata={
                                'font_info': self._get_font_info(block),
                                'column': self._get_column_position(block, page.rect.width)
                            }
                        )
                        self.chunks.append(chunk)
        
        doc.close()
    
    def _detect_columns(self, blocks: List[Dict], page_width: float) -> List[List[Dict]]:
        """
        Detect and separate columns in academic papers
        Simple heuristic: if blocks are on left/right half of page
        """
        if not blocks:
            return [[]]
        
        # Detect if this is a two-column layout
        mid_point = page_width / 2
        left_blocks = [b for b in blocks if b["bbox"][0] < mid_point]
        right_blocks = [b for b in blocks if b["bbox"][0] >= mid_point]
        
        # If both columns have content, it's two-column
        if len(left_blocks) > 2 and len(right_blocks) > 2:
            # Sort each column top to bottom
            left_sorted = sorted(left_blocks, key=lambda b: b["bbox"][1])
            right_sorted = sorted(right_blocks, key=lambda b: b["bbox"][1])
            return [left_sorted, right_sorted]
        else:
            # Single column or title page
            return [blocks]
    
    def _extract_block_text(self, block: Dict) -> str:
        """Extract clean text from a block"""
        lines = []
        for line in block.get("lines", []):
            line_text = ""
            for span in line.get("spans", []):
                line_text += span.get("text", "")
            lines.append(line_text)
        return "\n".join(lines)
    
    def _get_font_info(self, block: Dict) -> Dict:
        """Extract font information (useful for detecting headers)"""
        if not block.get("lines"):
            return {}
        
        first_span = block["lines"][0]["spans"][0] if block["lines"][0].get("spans") else {}
        return {
            'size': first_span.get('size', 0),
            'font': first_span.get('font', ''),
            'flags': first_span.get('flags', 0)
        }
    
    def _get_column_position(self, block: Dict, page_width: float) -> str:
        """Determine if block is in left or right column"""
        mid_point = page_width / 2
        return 'left' if block["bbox"][0] < mid_point else 'right'
    
    def _extract_tables(self):
        """Extract tables using pdfplumber"""
        with pdfplumber.open(self.pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                tables = page.extract_tables()
                
                for table_idx, table in enumerate(tables):
                    if not table:
                        continue
                    
                    # Convert table to structured text
                    table_text = self._format_table(table)
                    
                    chunk = DocumentChunk(
                        content=table_text,
                        page_num=page_num,
                        chunk_type='table',
                        bbox=(0, 0, 0, 0),  # pdfplumber doesn't give easy bbox
                        metadata={
                            'table_index': table_idx,
                            'rows': len(table),
                            'cols': len(table[0]) if table else 0
                        }
                    )
                    self.chunks.append(chunk)
    
    def _format_table(self, table: List[List[str]]) -> str:
        """Format table as readable text"""
        if not table:
            return ""
        
        # Use first row as header
        formatted = "TABLE:\n"
        header = table[0]
        formatted += " | ".join([str(cell) if cell else "" for cell in header]) + "\n"
        formatted += "-" * 50 + "\n"
        
        for row in table[1:]:
            formatted += " | ".join([str(cell) if cell else "" for cell in row]) + "\n"
        
        return formatted
    
    def _extract_figures(self):
        """
        Extract figure captions and metadata
        Note: Actual image processing would require OCR/vision models
        For now, we extract captions and positions
        """
        doc = fitz.open(self.pdf_path)
        
        for page_num, page in enumerate(doc, start=1):
            # Get images on page
            image_list = page.get_images()
            
            # Get text to find figure captions
            text_dict = page.get_text("dict")
            
            # Simple heuristic: look for "Figure" in text
            for block in text_dict["blocks"]:
                if block.get("type") == 0:  # Text block
                    text = self._extract_block_text(block)
                    
                    if "Figure" in text or "Fig." in text:
                        chunk = DocumentChunk(
                            content=text,
                            page_num=page_num,
                            chunk_type='figure_caption',
                            bbox=block["bbox"],
                            metadata={
                                'has_image': len(image_list) > 0,
                                'num_images': len(image_list)
                            }
                        )
                        self.chunks.append(chunk)
        
        doc.close()
    
    def save_chunks(self, output_path: Path):
        """Save extracted chunks to JSON"""
        chunks_dict = [chunk.to_dict() for chunk in self.chunks]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_dict, f, indent=2, ensure_ascii=False)
        
        print(f" Saved {len(chunks_dict)} chunks to {output_path}")


def load_chunks(json_path: Path) -> List[DocumentChunk]:
    """Load chunks from JSON"""
    with open(json_path, 'r', encoding='utf-8') as f:
        chunks_dict = json.load(f)
    
    return [
        DocumentChunk(**chunk) for chunk in chunks_dict
    ]