"""
Document processing module for handling various document types and text extraction.
"""

import os
from pathlib import Path
from typing import List, Union, Optional
import logging

from langchain_community.document_loaders import (
    TextLoader,
    PDFPlumberLoader,
    UnstructuredMarkdownLoader,
    UnstructuredFileLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Handles loading and processing of various document types."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def load_document(self, file_path: Union[str, Path]) -> List[Document]:
        """
        Load a document from file path.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of Document objects
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Loading document: {file_path}")
        
        # Determine loader based on file extension
        suffix = file_path.suffix.lower()
        
        try:
            if suffix == '.txt':
                loader = TextLoader(str(file_path))
            elif suffix == '.pdf':
                loader = PDFPlumberLoader(str(file_path))
            elif suffix in ['.md', '.markdown']:
                loader = UnstructuredMarkdownLoader(str(file_path))
            elif suffix in ['.html', '.htm']:
                loader = UnstructuredFileLoader(
                    str(file_path),
                    mode="elements",
                    strategy="fast"
                )
            else:
                # Fallback to unstructured loader for other file types
                loader = UnstructuredFileLoader(
                    str(file_path),
                    mode="elements",
                    strategy="fast"
                )
            
            return loader.load()
            
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {e}")
            raise
    
    def load_directory(self, directory_path: Union[str, Path]) -> List[Document]:
        """
        Load all supported documents from a directory.
        
        Args:
            directory_path: Path to the directory
            
        Returns:
            List of Document objects
        """
        directory_path = Path(directory_path)
        
        if not directory_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory_path}")
        
        all_documents = []
        supported_extensions = {'.txt', '.pdf', '.md', '.markdown', '.html', '.htm'}
        
        logger.info(f"Loading documents from directory: {directory_path}")
        
        for file_path in directory_path.rglob("*"):
            if file_path.suffix.lower() in supported_extensions:
                try:
                    docs = self.load_document(file_path)
                    all_documents.extend(docs)
                    logger.info(f"Loaded {len(docs)} documents from {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {e}")
        
        logger.info(f"Total documents loaded: {len(all_documents)}")
        return all_documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of chunked Document objects
        """
        logger.info(f"Splitting {len(documents)} documents into chunks")
        
        split_docs = self.text_splitter.split_documents(documents)
        
        # Add metadata to each chunk
        for i, doc in enumerate(split_docs):
            doc.metadata['chunk_id'] = i
            if 'source' not in doc.metadata:
                doc.metadata['source'] = f"chunk_{i}"
        
        logger.info(f"Created {len(split_docs)} chunks")
        return split_docs
    
    def process_file(self, file_path: Union[str, Path]) -> List[Document]:
        """
        Complete processing pipeline for a single file.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of chunked Document objects
        """
        documents = self.load_document(file_path)
        return self.split_documents(documents)
    
    def process_directory(self, directory_path: Union[str, Path]) -> List[Document]:
        """
        Complete processing pipeline for a directory.
        
        Args:
            directory_path: Path to the directory
            
        Returns:
            List of chunked Document objects
        """
        documents = self.load_directory(directory_path)
        return self.split_documents(documents)