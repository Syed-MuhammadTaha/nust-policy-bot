"""
Simplified preprocessing pipeline - reliable and effective.
"""

import os
import re
from typing import List, Dict, Any
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class SimplePreprocessor:
    """Simplified but effective document preprocessing."""
    
    def __init__(self, chunk_size=800, chunk_overlap=200, strategy="semantic"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
        
        # Enhanced separators for better semantic splitting
        self.separators = [
            "\n\n\n",  # Section breaks
            "\n\n",    # Paragraph breaks  
            "\n",      # Line breaks
            ". ",      # Sentence breaks
            "! ",      # Exclamation
            "? ",      # Question
            "; ",      # Semicolon
            ", ",      # Comma
            " ",       # Space
            ""         # Character fallback
        ]
    
    def process_pdf(self, file_path: str) -> List[Document]:
        """Process PDF with enhanced chunking."""
        # Load PDF
        loader = PyMuPDFLoader(file_path)
        pages = loader.load()
        
        file_name = os.path.basename(file_path)
        
        # Clean and combine text with page tracking
        all_docs = []
        
        for page_doc in pages:
            page_num = page_doc.metadata.get('page', 0) + 1  # 1-indexed
            text = self._clean_text(page_doc.page_content)
            
            if not text.strip():
                continue
            
            # Detect if this page has structure
            has_structure = self._has_structure(text)
            
            if self.strategy == "semantic" and has_structure:
                # Try to split by sections
                sections = self._split_by_sections(text)
                
                for section in sections:
                    if len(section['text']) > self.chunk_size * 1.5:
                        # Section too large, split it
                        sub_chunks = self._split_text(section['text'])
                        for chunk_text in sub_chunks:
                            all_docs.append(Document(
                                page_content=chunk_text,
                                metadata={
                                    'source': file_name,
                                    'page': page_num,
                                    'chunk_title': section.get('title', '')[:100],
                                }
                            ))
                    else:
                        # Section fits in one chunk
                        all_docs.append(Document(
                            page_content=section['text'],
                            metadata={
                                'source': file_name,
                                'page': page_num,
                                'chunk_title': section.get('title', '')[:100],
                            }
                        ))
            else:
                # Use fixed chunking
                chunks = self._split_text(text)
                for chunk_text in chunks:
                    title = self._extract_title(chunk_text)
                    all_docs.append(Document(
                        page_content=chunk_text,
                        metadata={
                            'source': file_name,
                            'page': page_num,
                            'chunk_title': title,
                        }
                    ))
        
        return all_docs
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove excessive whitespace but preserve structure
        text = re.sub(r'[ \t]+', ' ', text)  # Collapse spaces/tabs
        text = re.sub(r'\n{4,}', '\n\n\n', text)  # Max 3 newlines
        
        # Fix common issues
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Missing space
        
        # Normalize bullets
        text = re.sub(r'[•●○▪▫■□‣⁃]', '•', text)
        
        return text.strip()
    
    def _has_structure(self, text: str) -> bool:
        """Check if text has clear structure (headers, sections)."""
        # Look for numbered sections
        if re.search(r'\n\d+\.(?:\d+\.)*\s+[A-Z]', text):
            return True
        
        # Look for headers (all caps lines)
        if re.search(r'\n[A-Z][A-Z\s]{8,}\n', text):
            return True
        
        # Look for consistent list structures
        if len(re.findall(r'\n\s*[•\-\*]\s+', text)) >= 3:
            return True
        
        return False
    
    def _split_by_sections(self, text: str) -> List[Dict[str, str]]:
        """Split text into logical sections."""
        sections = []
        
        # Try numbered sections first (1., 1.1., etc.)
        pattern = r'(\n\d+\.(?:\d+\.)*\s+[^\n]{5,100})'
        matches = list(re.finditer(pattern, text))
        
        if matches:
            for i, match in enumerate(matches):
                start = match.start()
                end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                
                section_text = text[start:end].strip()
                title = match.group(1).strip()
                
                sections.append({
                    'title': title,
                    'text': section_text
                })
        else:
            # Try header-based sections (ALL CAPS)
            pattern = r'\n([A-Z][A-Z\s]{8,})\n'
            matches = list(re.finditer(pattern, text))
            
            if matches:
                for i, match in enumerate(matches):
                    start = match.start()
                    end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                    
                    section_text = text[start:end].strip()
                    title = match.group(1).strip()
                    
                    sections.append({
                        'title': title,
                        'text': section_text
                    })
            else:
                # No clear sections, return as single section
                sections.append({
                    'title': '',
                    'text': text
                })
        
        return sections
    
    def _split_text(self, text: str) -> List[str]:
        """Split text using RecursiveCharacterTextSplitter."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
        )
        return splitter.split_text(text)
    
    def _extract_title(self, text: str) -> str:
        """Extract a title/summary from chunk start."""
        lines = text.split('\n')
        for line in lines[:3]:
            line = line.strip()
            if line and len(line) < 100 and len(line) > 10:
                # Check if it looks like a title
                if line[0].isupper():
                    return line
        
        # Fallback: first 60 chars
        return text[:60].strip() + "..."


def preprocess_pdf_simple(file_path: str, chunk_size=800, chunk_overlap=200, strategy="semantic") -> List[Document]:
    """
    Simple but effective PDF preprocessing.
    
    Args:
        file_path: Path to PDF file
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between chunks
        strategy: "semantic" or "fixed"
    
    Returns:
        List of Document objects
    """
    preprocessor = SimplePreprocessor(chunk_size, chunk_overlap, strategy)
    return preprocessor.process_pdf(file_path)

