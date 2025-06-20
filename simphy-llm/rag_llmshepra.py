import os
import re
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import json

from dataclasses import dataclass, asdict, field

from llmsherpa.readers import LayoutPDFReader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ChunkMetadata:
    """Enhanced metadata for each chunk"""
    page_number: int
    section_title: Optional[str] = None
    section_level: int = 0
    section_hierarchy: str = ""
    parent_sections: List[str] = field(default_factory=list)

    content_type: str = "text"  # text, table, list, figure_caption
    block_index: int = 0
    reading_order: int = 0
    bbox: Optional[Dict[str, float]] = None
    font_info: Optional[Dict[str, Any]] = None
    table_data: Optional[Dict] = None
    list_items: Optional[List[str]] = None
    context_window: Optional[str] = None
    tokens: int = 0

class LLMSherpaContextualParser:
    """
    Advanced PDF parser using LLMSherpa LayoutPDFReader for context-aware RAG chunking.
    Maintains document structure, hierarchy, and semantic relationships.
    """
    
    def __init__(self, 
                 llmsherpa_api_url: str = "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 min_chunk_size: int = 50,
                 max_chunk_size: int = 2000,
                 preserve_tables: bool = True,
                 preserve_lists: bool = True,
                 include_context_window: bool = True,
                 context_window_size: int = 2):
        """
        Initialize the LLMSherpa-based parser.
        
        Args:
            llmsherpa_api_url: API endpoint for LLMSherpa service
            chunk_size: Target size for text chunks
            chunk_overlap: Overlap between adjacent chunks
            min_chunk_size: Minimum size for a valid chunk
            max_chunk_size: Maximum size before force-splitting
            preserve_tables: Keep tables as single chunks when possible
            preserve_lists: Keep lists intact within chunks
            include_context_window: Add surrounding context to chunks
            context_window_size: Number of blocks before/after for context
        """
        self.llmsherpa_api_url = llmsherpa_api_url
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.preserve_tables = preserve_tables
        self.preserve_lists = preserve_lists
        self.include_context_window = include_context_window
        self.context_window_size = context_window_size
        
        # Initialize the LayoutPDFReader
        self.pdf_reader = LayoutPDFReader(llmsherpa_api_url)
        
        # Initialize text splitter for oversized chunks
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ": ", " ", ""]
        )
        
        # Initialize tokenizer for accurate token counting
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except:
            self.tokenizer = None
            logger.warning("Could not load tiktoken encoder, using character count approximation")
        
        # Document structure tracking
        self.section_hierarchy = []
        self.current_reading_order = 0
        
    def parse_pdf(self, pdf_path: str) -> List[Document]:
        """
        Main parsing method that converts PDF to context-aware chunks.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of Document objects with rich metadata
        """
        logger.info(f"Starting PDF parsing with LLMSherpa: {pdf_path}")
        
        try:
            # Parse document with LLMSherpa
            doc = self.pdf_reader.read_pdf(pdf_path)
            logger.info(f"LLMSherpa parsed document with {len(doc.chunks())} chunks")
            
            # Extract structured elements
            structured_elements = self._extract_structured_elements(doc)
            logger.info(f"Extracted {len(structured_elements)} structured elements")
            
            # Build document hierarchy
            hierarchical_elements = self._build_document_hierarchy(structured_elements)
            logger.info("Built document hierarchy")
            
            # Create context-aware chunks
            chunks = self._create_contextual_chunks(hierarchical_elements)
            logger.info(f"Created {len(chunks)} contextual chunks")
            
            # Post-process chunks
            final_chunks = self._post_process_chunks(chunks)
            logger.info(f"Final processing complete: {len(final_chunks)} chunks ready for RAG")
            
            return final_chunks
            
        except Exception as e:
            logger.error(f"Error parsing PDF: {str(e)}")
            raise
    
    def _extract_structured_elements(self, doc) -> List[Dict]:
        """Extract and classify structural elements from LLMSherpa document."""
        elements = []
        
        # Process sections and their content
        for section in doc.sections():
            # Add section header
            if hasattr(section, 'title') and section.title:
                section_element = {
                    'content': section.title,
                    'type': 'section_header',
                    'level': getattr(section, 'level', 1),
                    'bbox': getattr(section, 'bbox', {}),
                    'page_number': getattr(section, 'page_idx', 1) + 1,
                    'block_index': len(elements)
                }
                elements.append(section_element)
            
            # Process chunks within section
            for chunk in section.chunks():
                chunk_element = {
                    'content': chunk.to_context_text(),
                    'type': self._classify_chunk_type(chunk),
                    'level': getattr(section, 'level', 1) + 1,
                    'bbox': getattr(chunk, 'bbox', {}),
                    'page_number': getattr(chunk, 'page_idx', 1) + 1,
                    'block_index': len(elements),
                    'parent_section': getattr(section, 'title', ''),
                    'chunk_object': chunk  # Keep reference for detailed extraction
                }
                
                # Extract additional metadata based on chunk type
                if chunk_element['type'] == 'table':
                    chunk_element['table_data'] = self._extract_table_data(chunk)
                elif chunk_element['type'] == 'list':
                    chunk_element['list_items'] = self._extract_list_items(chunk)
                
                elements.append(chunk_element)
        
        # Also process top-level chunks (those not in sections)
        for chunk in doc.chunks():
            # Skip if already processed as part of a section
            if not any(elem.get('chunk_object') == chunk for elem in elements):
                chunk_element = {
                    'content': chunk.to_context_text(),
                    'type': self._classify_chunk_type(chunk),
                    'level': 0,
                    'bbox': getattr(chunk, 'bbox', {}),
                    'page_number': getattr(chunk, 'page_idx', 1) + 1,
                    'block_index': len(elements),
                    'parent_section': '',
                    'chunk_object': chunk
                }
                elements.append(chunk_element)
        
        return elements
    
    def _classify_chunk_type(self, chunk) -> str:
        """Classify the type of content chunk."""
        # Check if it's a table
        if hasattr(chunk, 'tables') and chunk.tables():
            return 'table'
        
        # Check if it's a list
        content = chunk.to_context_text().strip()
        lines = content.split('\n')
        
        # Simple heuristics for list detection
        list_indicators = 0
        for line in lines[:5]:  # Check first 5 lines
            line = line.strip()
            if (line.startswith(('•', '-', '*', '○')) or 
                re.match(r'^\d+\.', line) or 
                re.match(r'^[a-zA-Z]\.', line)):
                list_indicators += 1
        
        if list_indicators >= 2:
            return 'list'
        
        # Check for figure captions
        if (content.lower().startswith(('figure', 'fig.', 'image', 'chart', 'graph')) or
            'caption' in content.lower()):
            return 'figure_caption'
        
        # Default to text
        return 'text'
    
    def _extract_table_data(self, chunk) -> Dict:
        """Extract structured data from table chunks."""
        table_data = {
            'raw_text': chunk.to_context_text(),
            'tables': []
        }
        
        if hasattr(chunk, 'tables'):
            for table in chunk.tables():
                if hasattr(table, 'df'):
                    # Convert pandas DataFrame to dict
                    table_dict = {
                        'headers': table.df.columns.tolist() if hasattr(table.df, 'columns') else [],
                        'rows': table.df.values.tolist() if hasattr(table.df, 'values') else [],
                        'shape': table.df.shape if hasattr(table.df, 'shape') else (0, 0)
                    }
                    table_data['tables'].append(table_dict)
        
        return table_data
    
    def _extract_list_items(self, chunk) -> List[str]:
        """Extract individual items from list chunks."""
        content = chunk.to_context_text()
        lines = content.split('\n')
        
        list_items = []
        for line in lines:
            line = line.strip()
            if line and (line.startswith(('•', '-', '*', '○')) or 
                        re.match(r'^\d+\.', line) or 
                        re.match(r'^[a-zA-Z]\.', line)):
                # Clean up list markers
                cleaned_item = re.sub(r'^[•\-*○\d+a-zA-Z\.]\s*', '', line)
                if cleaned_item:
                    list_items.append(cleaned_item)
        
        return list_items
    
    def _build_document_hierarchy(self, elements: List[Dict]) -> List[Dict]:
        """Build hierarchical structure and assign section numbering."""
        hierarchical_elements = []
        section_stack = []
        section_counters = {}
        
        for element in elements:
            if element['type'] == 'section_header':
                level = element['level']
                
                # Update section counters
                if level not in section_counters:
                    section_counters[level] = 0
                section_counters[level] += 1
                
                # Reset deeper level counters
                for deeper_level in list(section_counters.keys()):
                    if deeper_level > level:
                        del section_counters[deeper_level]
                
                # Build hierarchy string
                hierarchy_parts = []
                for l in sorted(section_counters.keys()):
                    if l <= level:
                        hierarchy_parts.append(str(section_counters[l]))
                
                hierarchy_string = '.'.join(hierarchy_parts)
                
                # Update section stack
                section_stack = section_stack[:level-1] if level > 1 else []
                section_stack.append({
                    'title': element['content'],
                    'level': level,
                    'hierarchy': hierarchy_string
                })
                
                # Update element
                element['section_hierarchy'] = hierarchy_string
                element['parent_sections'] = [s['title'] for s in section_stack[:-1]]
            
            else:
                # Assign current section context to content elements
                if section_stack:
                    current_section = section_stack[-1]
                    element['section_hierarchy'] = current_section['hierarchy']
                    element['parent_sections'] = [s['title'] for s in section_stack]
                    element['section_title'] = current_section['title']
                else:
                    element['section_hierarchy'] = ''
                    element['parent_sections'] = []
                    element['section_title'] = None
            
            # Assign reading order
            element['reading_order'] = self.current_reading_order
            self.current_reading_order += 1
            
            hierarchical_elements.append(element)
        
        return hierarchical_elements
    
    def _create_contextual_chunks(self, elements: List[Dict]) -> List[Document]:
        """Create semantically coherent chunks with rich context."""
        chunks = []
        current_chunk_elements = []
        current_chunk_size = 0
        
        for i, element in enumerate(elements):
            element_content = element['content']
            element_size = self._count_tokens(element_content)
            
            # Decide whether to start a new chunk
            should_start_new = self._should_start_new_chunk(
                current_chunk_elements, element, current_chunk_size, element_size
            )
            
            if should_start_new and current_chunk_elements:
                # Create chunk from current elements
                chunk_doc = self._create_chunk_document(current_chunk_elements, elements, i)
                chunks.append(chunk_doc)
                
                # Start new chunk
                current_chunk_elements = [element]
                current_chunk_size = element_size
            else:
                # Add to current chunk
                current_chunk_elements.append(element)
                current_chunk_size += element_size
        
        # Handle final chunk
        if current_chunk_elements:
            chunk_doc = self._create_chunk_document(current_chunk_elements, elements, len(elements))
            chunks.append(chunk_doc)
        
        return chunks
    
    def _should_start_new_chunk(self, current_elements: List[Dict], new_element: Dict, 
                               current_size: int, new_size: int) -> bool:
        """Determine if a new chunk should be started."""
        if not current_elements:
            return False
        
        # Always start new chunk for section headers
        if new_element['type'] == 'section_header':
            return True
        
        # Check size limits
        if current_size + new_size > self.chunk_size:
            return True
        
        # Preserve table integrity
        if (self.preserve_tables and new_element['type'] == 'table' and 
            current_elements and current_elements[-1]['type'] != 'table'):
            return True
        
        # Check for section changes
        last_element = current_elements[-1]
        if (new_element.get('section_hierarchy') != last_element.get('section_hierarchy') and
            new_element['type'] != 'section_header'):
            return True
        
        # Check for content type changes that should trigger new chunks
        type_change_triggers = {
            ('text', 'table'), ('table', 'text'),
            ('text', 'list'), ('list', 'text'),
            ('table', 'list'), ('list', 'table')
        }
        
        current_type = last_element['type']
        new_type = new_element['type']
        
        if (current_type, new_type) in type_change_triggers:
            return True
        
        return False
    
    def _create_chunk_document(self, chunk_elements: List[Dict], all_elements: List[Dict], 
                              current_index: int) -> Document:
        """Create a Document object from chunk elements with rich metadata."""
        # Combine content from all elements in chunk
        content_parts = []
        chunk_metadata = ChunkMetadata(
            page_number=chunk_elements[0]['page_number'],
            block_index=chunk_elements[0]['block_index'],
            reading_order=chunk_elements[0]['reading_order']
        )
        
        # Process each element in the chunk
        for element in chunk_elements:
            content = element['content']
            
            # Add content type markers for better context
            if element['type'] == 'section_header':
                content = f"## {content}"
            elif element['type'] == 'table' and self.preserve_tables:
                content = f"[TABLE]\n{content}\n[/TABLE]"
            elif element['type'] == 'list' and self.preserve_lists:
                content = f"[LIST]\n{content}\n[/LIST]"
            elif element['type'] == 'figure_caption':
                content = f"[FIGURE CAPTION] {content}"
            
            content_parts.append(content)
            
            # Update chunk metadata with the most relevant element's data
            if element['type'] != 'section_header':  # Prefer content over headers for main metadata
                chunk_metadata.section_title = element.get('section_title')
                chunk_metadata.section_hierarchy = element.get('section_hierarchy', '')
                chunk_metadata.parent_sections = element.get('parent_sections', [])
                chunk_metadata.content_type = element['type']
                
                if element['type'] == 'table' and element.get('table_data'):
                    chunk_metadata.table_data = element['table_data']
                elif element['type'] == 'list' and element.get('list_items'):
                    chunk_metadata.list_items = element['list_items']
        
        # Combine content
        combined_content = '\n\n'.join(content_parts)
        
        # Add context window if enabled
        if self.include_context_window:
            context_window = self._build_context_window(all_elements, current_index, chunk_elements)
            chunk_metadata.context_window = context_window
            if context_window:
                combined_content = f"[CONTEXT: {context_window}]\n\n{combined_content}"
        
        # Count tokens
        chunk_metadata.tokens = self._count_tokens(combined_content)
        
        # Create the Document
        return Document(
            page_content=combined_content,
            metadata=asdict(chunk_metadata)
        )
    
    def _build_context_window(self, all_elements: List[Dict], current_index: int, 
                             chunk_elements: List[Dict]) -> str:
        """Build context window from surrounding elements."""
        if not all_elements or current_index == 0:
            return ""
        
        # Find the range of elements for this chunk
        chunk_start_idx = all_elements.index(chunk_elements[0])
        chunk_end_idx = all_elements.index(chunk_elements[-1])
        
        # Get preceding context
        context_start = max(0, chunk_start_idx - self.context_window_size)
        preceding_elements = all_elements[context_start:chunk_start_idx]
        
        # Get following context
        context_end = min(len(all_elements), chunk_end_idx + 1 + self.context_window_size)
        following_elements = all_elements[chunk_end_idx + 1:context_end]
        
        context_parts = []
        
        # Add preceding context
        if preceding_elements:
            preceding_text = ' ... '.join([elem['content'][:100] for elem in preceding_elements[-2:]])
            context_parts.append(f"Before: {preceding_text}")
        
        # Add following context
        if following_elements:
            following_text = ' ... '.join([elem['content'][:100] for elem in following_elements[:2]])
            context_parts.append(f"After: {following_text}")
        
        return ' | '.join(context_parts)
    
    def _post_process_chunks(self, chunks: List[Document]) -> List[Document]:
        """Post-process chunks for final optimization."""
        processed_chunks = []
        
        for chunk in chunks:
            # Handle oversized chunks
            if chunk.metadata['tokens'] > self.max_chunk_size:
                # Split oversized chunks while preserving some context
                sub_chunks = self._split_oversized_chunk(chunk)
                processed_chunks.extend(sub_chunks)
            elif len(chunk.page_content.strip()) >= self.min_chunk_size:
                # Keep chunks that meet minimum size
                processed_chunks.append(chunk)
            # Skip chunks that are too small
        
        # Add chunk sequence information
        for i, chunk in enumerate(processed_chunks):
            chunk.metadata['chunk_index'] = i
            chunk.metadata['total_chunks'] = len(processed_chunks)
        
        return processed_chunks
    
    def _split_oversized_chunk(self, chunk: Document) -> List[Document]:
        """Split oversized chunks while preserving metadata."""
        content = chunk.page_content
        metadata = chunk.metadata.copy()
        
        # Use text splitter for oversized content
        split_texts = self.text_splitter.split_text(content)
        
        sub_chunks = []
        for i, text in enumerate(split_texts):
            if len(text.strip()) >= self.min_chunk_size:
                # Create sub-chunk metadata
                sub_metadata = metadata.copy()
                sub_metadata['sub_chunk_index'] = i
                sub_metadata['total_sub_chunks'] = len(split_texts)
                sub_metadata['tokens'] = self._count_tokens(text)
                
                sub_chunk = Document(page_content=text, metadata=sub_metadata)
                sub_chunks.append(sub_chunk)
        
        return sub_chunks
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken or approximation."""
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except:
                pass
        
        # Fallback approximation: ~4 characters per token
        return len(text) // 4
    
    def get_document_stats(self, chunks: List[Document]) -> Dict:
        """Generate statistics about the parsed document."""
        if not chunks:
            return {}
        
        stats = {
            'total_chunks': len(chunks),
            'total_tokens': sum(chunk.metadata.get('tokens', 0) for chunk in chunks),
            'avg_chunk_size': sum(chunk.metadata.get('tokens', 0) for chunk in chunks) / len(chunks),
            'content_types': {},
            'sections': set(),
            'pages': set()
        }
        
        for chunk in chunks:
            # Count content types
            content_type = chunk.metadata.get('content_type', 'unknown')
            stats['content_types'][content_type] = stats['content_types'].get(content_type, 0) + 1
            
            # Collect sections and pages
            if chunk.metadata.get('section_title'):
                stats['sections'].add(chunk.metadata['section_title'])
            stats['pages'].add(chunk.metadata.get('page_number', 1))
        
        stats['unique_sections'] = len(stats['sections'])
        stats['total_pages'] = len(stats['pages'])
        
        return stats

# Example usage and testing
def main():
    """Example usage of the LLMSherpa parser"""
    
    # Initialize parser with custom settings
    parser = LLMSherpaContextualParser(
        chunk_size=800,
        chunk_overlap=150,
        min_chunk_size=50,
        preserve_tables=True,
        preserve_lists=True,
        include_context_window=True
    )
    
    
    pdf_path = os.path.join(SCRIPT_DIR, "docs", "SimpScriptGPart4Ch4.pdf")
  
    
    try:
        # Parse PDF
        chunks = parser.parse_pdf(pdf_path)
        
        # Get document statistics
        stats = parser.get_document_stats(chunks)
        
        # Display results
        print(f"\n=== Document Parsing Results ===")
        print(f"Total chunks: {stats['total_chunks']}")
        print(f"Total tokens: {stats['total_tokens']}")
        print(f"Average chunk size: {stats['avg_chunk_size']:.1f} tokens")
        print(f"Unique sections: {stats['unique_sections']}")
        print(f"Total pages: {stats['total_pages']}")
        print(f"Content types: {stats['content_types']}")
        
        print(f"\n=== Sample Chunks ===")
        for i, chunk in enumerate(chunks[:3]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Page: {chunk.metadata['page_number']}")
            print(f"Section: {chunk.metadata.get('section_title', 'N/A')}")
            print(f"Hierarchy: {chunk.metadata['section_hierarchy']}")
            print(f"Type: {chunk.metadata['content_type']}")
            print(f"Tokens: {chunk.metadata['tokens']}")
            print(f"Content preview:\n{chunk.page_content[:300]}...")
            
            if chunk.metadata.get('context_window'):
                print(f"Context: {chunk.metadata['context_window'][:200]}...")
        
        # Export chunks to JSON for inspection
        export_data = []
        for chunk in chunks:
            export_data.append({
                'content': chunk.page_content,
                'metadata': chunk.metadata
            })
        
        with open('parsed_chunks.json', 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nExported {len(chunks)} chunks to 'parsed_chunks.json'")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"Make sure you have LLMSherpa service running and the PDF file exists.")

if __name__ == "__main__":
    main()