from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import layoutparser as lp
import numpy as np
from PIL import Image
import fitz  # PyMuPDF
from io import BytesIO
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ChunkMetadata:
    """Metadata for each chunk to maintain context"""
    page_number: int
    section_type: str  # header, paragraph, table, figure, etc.
    section_hierarchy: str  # e.g., "1.2.3" for nested sections
    parent_section: Optional[str] = None
    bbox: Optional[Tuple[float, float, float, float]] = None
    font_size: Optional[float] = None
    is_title: bool = False
    table_context: Optional[str] = None
    figure_context: Optional[str] = None

class ContextAwarePDFParser:
    """
    A context-aware PDF parser that maintains document structure and hierarchy
    for better RAG performance.
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 min_chunk_size: int = 100):
        """
        Initialize the parser with layout detection model and chunking parameters.
        
        Args:
            model_path: Path to custom layout detection model (optional)
            chunk_size: Maximum size of text chunks
            chunk_overlap: Overlap between chunks to maintain context
            min_chunk_size: Minimum size for a chunk to be valid
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        
        # Initialize layout detection model
        if model_path:
            self.model = lp.models.Detectron2LayoutModel(model_path)
        # Initialize layout detection model
        if model_path:
            self.model = lp.Detectron2LayoutModel(model_path)
            

        else:
            # Use pre-trained PubLayNet model for academic papers
            self.model = lp.Detectron2LayoutModel(
                'lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
                extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8]
            )
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        # Track document structurea
        self.section_hierarchy = []
        self.current_section = ""
        
    def extract_layout_elements(self, pdf_path: str) -> List[Dict]:
        """
        Extract layout elements from PDF using layout detection.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of layout elements with their properties
        """
        doc = fitz.Document(pdf_path)
        
        layout_elements = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Convert page to image for layout detection
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom
            img_data = pix.tobytes("png")
            # Convert page to image for layout detection
            zoom = 2  # 2x zoom
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)  # PyMuPDF 1.18.0+ syntax
            img_data = pix.tobytes("png")
            
            # Load image with PIL
            img = Image.open(BytesIO(img_data))
            
            # Extract text blocks with layout information
            blocks = page.get_text("dict")
            # Extract text blocks with layout information
            blocks = page.get_text("dict")  # For newer PyMuPDF versions
            # Alternative if above doesn't work: blocks = json.loads(page.get_text("json"))
                if "lines" in block:  # Text block
                    bbox = block["bbox"]
                    text_content = ""
                    font_sizes = []
                    
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text_content += span["text"]
                            font_sizes.append(span["size"])
                    
                    if text_content.strip():
                        # Find corresponding layout element
                        layout_type = self._classify_layout_element(
                            bbox, layout, text_content, font_sizes
                        )
                        
                        element = {
                            "page": page_num + 1,
                            "bbox": bbox,
                            "text": text_content.strip(),
                            "type": layout_type,
                            "font_size": np.mean(font_sizes) if font_sizes else 12,
                            "is_title": self._is_title(text_content, font_sizes)
                        }
                        
                        layout_elements.append(element)
        
        doc.close()
        return layout_elements
    
    def _classify_layout_element(self, bbox: Tuple, layout, text: str, font_sizes: List[float]) -> str:
        """Classify the type of layout element based on detection results."""
        # Find overlapping layout detection
        for element in layout:
            if self._bbox_overlap(bbox, element.coordinates):
                return element.type.lower()
        
        # Fallback classification based on text properties
        if self._is_title(text, font_sizes):
            return "title"
        elif len(text.split()) < 10:
            return "header"
        else:
            return "text"
    
    def _bbox_overlap(self, bbox1: Tuple, bbox2) -> bool:
        """Check if two bounding boxes overlap significantly."""
        x1, y1, x2, y2 = bbox1
        x3, y3, x4, y4 = bbox2.x1, bbox2.y1, bbox2.x2, bbox2.y2
        
        # Calculate overlap area
        overlap_x = max(0, min(x2, x4) - max(x1, x3))
        overlap_y = max(0, min(y2, y4) - max(y1, y3))
        overlap_area = overlap_x * overlap_y
        
        # Calculate areas
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x4 - x3) * (y4 - y3)
        
        # Check if overlap is significant (>50% of smaller area)
        min_area = min(area1, area2)
        return overlap_area / min_area > 0.5 if min_area > 0 else False
    
    def _is_title(self, text: str, font_sizes: List[float]) -> bool:
        """Determine if text is likely a title or header."""
        if not font_sizes:
            return False
            
        avg_font_size = np.mean(font_sizes)
        
        # Heuristics for title detection
        is_short = len(text.split()) <= 15
        is_large_font = avg_font_size > 14
        is_capitalized = text.isupper() or text.istitle()
        ends_with_punctuation = text.rstrip().endswith(('.', '!', '?'))
        
        return bool(is_short and (is_large_font or is_capitalized) and not ends_with_punctuation)
    
    def _build_section_hierarchy(self, elements: List[Dict]) -> List[Dict]:
        """Build hierarchical structure from layout elements."""
        structured_elements = []
        section_stack = []
        section_counter = {"1": 0, "2": 0, "3": 0, "4": 0}
        
        for element in elements:
            if element["type"] in ["title", "header"] or element["is_title"]:
                # Determine hierarchy level based on font size and position
                level = self._determine_hierarchy_level(element, section_stack)
                
                # Update section numbering
                section_counter[str(level)] += 1
                for i in range(level + 1, 5):
                    section_counter[str(i)] = 0
                
                # Build section number
                section_parts = []
                for i in range(1, level + 1):
                    if section_counter[str(i)] > 0:
                        section_parts.append(str(section_counter[str(i)]))
                
                section_number = ".".join(section_parts)
                
                # Update section stack
                section_stack = section_stack[:level-1]
                section_stack.append({
                    "title": element["text"],
                    "level": level,
                    "number": section_number
                })
                
                element["section_hierarchy"] = section_number
                element["hierarchy_level"] = level
            else:
                # Assign current section context
                if section_stack:
                    current_section = section_stack[-1]
                    element["section_hierarchy"] = current_section["number"]
                    element["parent_section"] = current_section["title"]
                    element["hierarchy_level"] = current_section["level"] + 1
            
            structured_elements.append(element)
        
        return structured_elements
    
    def _determine_hierarchy_level(self, element: Dict, section_stack: List[Dict]) -> int:
        """Determine the hierarchy level of a title/header element."""
        font_size = element.get("font_size", 12)
        
        if not section_stack:
            return 1
        
        # Compare with previous sections
        last_section = section_stack[-1]
        last_font_size = last_section.get("font_size", 12)
        
        if font_size > last_font_size + 2:
            return max(1, last_section["level"] - 1)
        elif font_size < last_font_size - 2:
            return min(4, last_section["level"] + 1)
        else:
            return last_section["level"]
    
    def create_context_aware_chunks(self, elements: List[Dict]) -> List[Document]:
        """
        Create chunks that maintain context and semantic coherence.
        
        Args:
            elements: List of structured layout elements
            
        Returns:
            List of Document objects with rich metadata
        """
        chunks = []
        current_chunk = ""
        current_metadata = None
        
        for element in elements:
            element_text = element["text"]
            
            # Create metadata for this element
            metadata = ChunkMetadata(
                page_number=element["page"],
                section_type=element["type"],
                section_hierarchy=element.get("section_hierarchy", ""),
                parent_section=element.get("parent_section"),
                bbox=element["bbox"],
                font_size=element.get("font_size"),
                is_title=element.get("is_title", False)
            )
            
            # Check if we should start a new chunk
            if (self._should_start_new_chunk(current_chunk, element_text, metadata, current_metadata) or
                len(current_chunk) + len(element_text) > self.chunk_size):
                
                # Save current chunk if it's substantial and has metadata
                if len(current_chunk.strip()) > self.min_chunk_size and current_metadata is not None:
                    chunks.append(self._create_document(current_chunk, current_metadata))
                
                # Start new chunk
                current_chunk = element_text
                current_metadata = metadata
            else:
                # Add to current chunk
                current_chunk += "\n\n" + element_text
                # Update metadata (keep the most recent)
                current_metadata = metadata
        
        # Add final chunk if it has metadata
        if len(current_chunk.strip()) > self.min_chunk_size and current_metadata is not None:
            chunks.append(self._create_document(current_chunk, current_metadata))
        
        return chunks
    
    def _should_start_new_chunk(self, current_chunk: str, new_text: str, 
    def _should_start_new_chunk(self, current_chunk: str, _new_text: str, 
                               new_metadata: ChunkMetadata, current_metadata: Optional[ChunkMetadata]) -> bool:
        """Determine if a new chunk should be started based on context."""
        if not current_chunk or not current_metadata:
            return False
        
        # Start new chunk for major section changes
        if (new_metadata.section_hierarchy != current_metadata.section_hierarchy and
            new_metadata.is_title):
            return True
        
        # Start new chunk for different element types (e.g., text to table)
        if (new_metadata.section_type != current_metadata.section_type and
            new_metadata.section_type in ["table", "figure"]):
            return True
        
        # Start new chunk for different pages if it's a major boundary
        if (new_metadata.page_number != current_metadata.page_number and
            new_metadata.is_title):
            return True
        
        return False
    def _create_document(self, text: str, metadata: ChunkMetadata) -> Document:
        """Create a Document object with rich metadata."""
        # Build context string
        context_parts = []
        if metadata.parent_section:
            context_parts.append(f"Section: {metadata.parent_section}")
        if metadata.section_hierarchy:
            context_parts.append(f"Hierarchy: {metadata.section_hierarchy}")
        
        context_string = " | ".join(context_parts)
        
        # Create enhanced text with context
        enhanced_text = text
        if context_string:
            enhanced_text = f"[{context_string}]\n\n{text}"
        
        return Document(
            page_content=enhanced_text,
            metadata={
                "page": metadata.page_number,
                "section_type": metadata.section_type,
                "section_hierarchy": metadata.section_hierarchy,
                "parent_section": metadata.parent_section,
                "bbox": metadata.bbox,
                "font_size": metadata.font_size,
                "is_title": metadata.is_title,
                "context": context_string,
                "chunk_size": len(text)
            }
        )
    
    def parse_pdf(self, pdf_path: str) -> List[Document]:
        """
        Main method to parse PDF and create context-aware chunks.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of Document objects ready for RAG indexing
        """
        logger.info(f"Starting PDF parsing: {pdf_path}")
        
        # Extract layout elements
        logger.info("Extracting layout elements...")
        elements = self.extract_layout_elements(pdf_path)
        
        # Build hierarchical structure
        logger.info("Building document structure...")
        structured_elements = self._build_section_hierarchy(elements)
        
        # Create context-aware chunks
        logger.info("Creating context-aware chunks...")
        chunks = self.create_context_aware_chunks(structured_elements)
        
        logger.info(f"Created {len(chunks)} chunks from {len(elements)} elements")
        
        return chunks

# Example usage and testing
# Example usage and testing
def main():
    """Example usage of the ContextAwarePDFParser"""
    
    # Initialize parser
    parser = ContextAwarePDFParser(
        chunk_size=1000,
        chunk_overlap=200,
        min_chunk_size=100
    )
    
    # Parse PDF
    pdf_path = os.path.join(os.path.dirname(__file__), "sample_document.pdf")  # Replace with your PDF path
    try:
        chunks = parser.parse_pdf(pdf_path)
        
        # Display results
        print(f"\nSuccessfully parsed PDF into {len(chunks)} chunks")
        print("\nFirst few chunks:")
        
        for i, chunk in enumerate(chunks[:3]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Page: {chunk.metadata['page']}")
            print(f"Section: {chunk.metadata.get('parent_section', 'N/A')}")
            print(f"Type: {chunk.metadata['section_type']}")
            print(f"Hierarchy: {chunk.metadata['section_hierarchy']}")
            print(f"Content preview: {chunk.page_content[:200]}...")
            
    except Exception as e:
        logger.error(f"Error parsing PDF: {e}")


    

if __name__ == "__main__":
    main()