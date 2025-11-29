"""
Structure-aware preprocessing pipeline for robust document chunking.
Preserves semantic boundaries while maintaining optimal chunk sizes.
"""

import os
import re
from typing import List, Dict, Any
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class SimplePreprocessor:
    """
    Structure-aware document preprocessing with intelligent chunking.
    
    Strategy:
    1. Detects document structure (headings, sections, numbered lists)
    2. Preserves semantic boundaries (prevents breaking mid-concept)
    3. Applies recursive splitting for large sections
    4. Optimal chunk sizes: 300-500 words (≈700-1200 tokens)
    """
    
    def __init__(self, chunk_size=1000, chunk_overlap=150, strategy="structure_aware"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
        
        # Enhanced separators for structure-aware splitting
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
            
            if self.strategy == "structure_aware":
                # Structure-aware chunking: preserves headings and semantic boundaries
                sections = self._split_by_structure(text)
                
                for section in sections:
                    section_text = section['text']
                    section_title = section.get('title', '')
                    
                    # If section is too large, recursively split it
                    if len(section_text) > self.chunk_size:
                        sub_chunks = self._split_text(section_text)
                        for chunk_text in sub_chunks:
                            all_docs.append(Document(
                                page_content=chunk_text,
                                metadata={
                                    'source': file_name,
                                    'page': page_num,
                                    'chunk_title': section_title[:200],
                                }
                            ))
                    else:
                        # Section fits in one chunk - preserve it whole
                        all_docs.append(Document(
                            page_content=section_text,
                            metadata={
                                'source': file_name,
                                'page': page_num,
                                'chunk_title': section_title[:200],
                            }
                        ))
            else:
                # Fallback to fixed chunking
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
    
    def _split_by_structure(self, text: str) -> List[Dict[str, str]]:
        """
        Structure-aware splitting: detects and preserves semantic boundaries.
        Looks for headings, numbered sections, and other structural markers.
        """
        sections = []
        
        # Multi-pattern detection for comprehensive structure recognition
        # Pattern 1: ALL CAPS HEADINGS (e.g., "PAKISTANI / NATIONAL DRESS FOR MEN")
        # Pattern 2: Numbered sections (e.g., "1. Introduction")
        # Pattern 3: Numbered subsections (e.g., "1.1. Details")
        
        # Combined pattern that catches multiple heading styles
        patterns = [
            r'\n([A-Z][A-Z\s/]{8,})\n',  # ALL CAPS HEADINGS
            r'\n(\d+\.\s+[A-Z][^\n]{5,80})\n',  # Numbered sections starting with capital
            r'\n([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*:)',  # Title Case with colon
        ]
        
        all_matches = []
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                all_matches.append((match.start(), match.end(), match.group(1).strip()))
        
        # Sort matches by position
        all_matches.sort(key=lambda x: x[0])
        
        if all_matches:
            # Split text by detected structural boundaries
            for i, (start, end, title) in enumerate(all_matches):
                # Determine section boundaries
                section_start = start
                if i + 1 < len(all_matches):
                    section_end = all_matches[i + 1][0]
                else:
                    section_end = len(text)
                    
                section_text = text[section_start:section_end].strip()
                
                # Only create section if it has meaningful content
                if len(section_text) > 50:  # Minimum viable section
                    sections.append({
                        'title': title,
                        'text': section_text
                    })
            
            # Capture any text before first heading
            if all_matches[0][0] > 100:  # Significant text before first heading
                intro_text = text[:all_matches[0][0]].strip()
                if len(intro_text) > 50:
                    sections.insert(0, {
                        'title': 'Introduction',
                        'text': intro_text
                    })
            else:
            # No structural markers found - treat as single section
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


def preprocess_pdf_simple(file_path: str, chunk_size=1000, chunk_overlap=150, strategy="structure_aware") -> List[Document]:
    """
    Structure-aware PDF preprocessing with intelligent chunking.
    
    Args:
        file_path: Path to PDF file
        chunk_size: Target chunk size (default 1000 chars ≈ 300-500 words)
        chunk_overlap: Overlap between chunks (default 150 ≈ 10-15%)
        strategy: "structure_aware" (recommended) or "fixed"
    
    Returns:
        List of Document objects with preserved semantic boundaries
    """
    preprocessor = SimplePreprocessor(chunk_size, chunk_overlap, strategy)
    return preprocessor.process_pdf(file_path)

