"""
Main CLI interface for document indexing pipeline.
"""

import os
import sys
import click
import logging
from pathlib import Path
from typing import Optional

from .document_processor import DocumentProcessor
from .embedding_generator import EmbeddingGenerator, VectorStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def cli(verbose: bool):
    """Document Indexing Pipeline CLI"""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


@cli.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option('--output', '-o', default='index', help='Output directory for the index')
@click.option('--model', '-m', default='sentence-transformers/all-MiniLM-L6-v2', 
              help='Embedding model name')
@click.option('--model-type', '-t', default='huggingface',
              type=click.Choice(['huggingface', 'openai', 'sentence-transformers']),
              help='Type of embedding model')
@click.option('--chunk-size', '-c', default=1000, type=int, help='Chunk size for text splitting')
@click.option('--chunk-overlap', '-l', default=200, type=int, help='Chunk overlap for text splitting')
@click.option('--openai-api-key', envvar='OPENAI_API_KEY', help='OpenAI API key')
def index(input_path: str, output: str, model: str, model_type: str,
          chunk_size: int, chunk_overlap: int, openai_api_key: Optional[str]):
    """
    Index documents from INPUT_PATH (file or directory).
    
    INPUT_PATH: Path to a document file or directory containing documents
    """
    try:
        # Set OpenAI API key if provided
        if openai_api_key:
            os.environ['OPENAI_API_KEY'] = openai_api_key
        
        logger.info("Starting document indexing pipeline")
        logger.info(f"Input path: {input_path}")
        logger.info(f"Output directory: {output}")
        logger.info(f"Model: {model} ({model_type})")
        logger.info(f"Chunk size: {chunk_size}, Overlap: {chunk_overlap}")
        
        # Initialize components
        processor = DocumentProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        embedding_generator = EmbeddingGenerator(model_name=model, model_type=model_type)
        vector_store = VectorStore(storage_path=output, embedding_generator=embedding_generator)
        
        # Process documents
        input_path_obj = Path(input_path)
        
        if input_path_obj.is_file():
            logger.info(f"Processing single file: {input_path}")
            documents = processor.process_file(input_path)
        else:
            logger.info(f"Processing directory: {input_path}")
            documents = processor.process_directory(input_path)
        
        # Add to vector store
        vector_store.add_documents(documents)
        
        # Save the index
        vector_store.save_to_json()
        
        # Save metadata
        stats = {
            "total_documents": len(documents),
            "total_chunks": len(vector_store.documents),
            "model_info": {
                "model_name": model,
                "model_type": model_type
            },
            "processing_params": {
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap
            },
            "input_path": str(input_path_obj.resolve())
        }
        
        stats_path = Path(output) / "index_stats.json"
        import json
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logger.info("Indexing completed successfully!")
        logger.info(f"Total documents processed: {len(documents)}")
        logger.info(f"Index saved to: {output}")
        
    except Exception as e:
        logger.error(f"Error during indexing: {e}")
        sys.exit(1)


@cli.command()
@click.argument('query')
@click.option('--index-path', '-i', default='index', help='Path to the index directory')
@click.option('--model', '-m', default='sentence-transformers/all-MiniLM-L6-v2', 
              help='Embedding model name')
@click.option('--model-type', '-t', default='huggingface',
              type=click.Choice(['huggingface', 'openai', 'sentence-transformers']),
              help='Type of embedding model')
@click.option('--top-k', '-k', default=5, type=int, help='Number of results to return')
@click.option('--openai-api-key', envvar='OPENAI_API_KEY', help='OpenAI API key')
def search(query: str, index_path: str, model: str, model_type: str,
           top_k: int, openai_api_key: Optional[str]):
    """
    Search documents in the index.
    
    QUERY: Search query string
    """
    try:
        # Set OpenAI API key if provided
        if openai_api_key:
            os.environ['OPENAI_API_KEY'] = openai_api_key
        
        logger.info(f"Searching for: {query}")
        logger.info(f"Index path: {index_path}")
        logger.info(f"Top K: {top_k}")
        
        # Initialize components
        embedding_generator = EmbeddingGenerator(model_name=model, model_type=model_type)
        vector_store = VectorStore(storage_path=index_path, embedding_generator=embedding_generator)
        
        # Load the index
        vector_store.load_from_json()
        
        # Perform search
        results = vector_store.similarity_search(query, k=top_k)
        
        # Display results
        if not results:
            click.echo("No results found.")
            return
        
        click.echo(f"\nSearch results for: '{query}'\n")
        click.echo("=" * 80)
        
        for i, (doc, similarity) in enumerate(results, 1):
            click.echo(f"\nResult {i} (Similarity: {similarity:.4f})")
            click.echo("-" * 40)
            
            # Show metadata
            metadata = doc.metadata
            if metadata:
                click.echo("Metadata:")
                for key, value in metadata.items():
                    click.echo(f"  {key}: {value}")
            
            # Show content preview
            content_preview = doc.page_content[:300]
            if len(doc.page_content) > 300:
                content_preview += "..."
            
            click.echo(f"\nContent preview:\n{content_preview}")
            click.echo("\n" + "=" * 80)
        
        logger.info(f"Found {len(results)} results")
        
    except Exception as e:
        logger.error(f"Error during search: {e}")
        sys.exit(1)


@cli.command()
@click.argument('index-path', type=click.Path(exists=True))
def stats(index_path: str):
    """
    Show statistics about the index.
    
    INDEX_PATH: Path to the index directory
    """
    try:
        import json
        
        stats_path = Path(index_path) / "index_stats.json"
        
        if not stats_path.exists():
            click.echo(f"Stats file not found: {stats_path}")
            return
        
        with open(stats_path, 'r', encoding='utf-8') as f:
            stats = json.load(f)
        
        click.echo("Index Statistics")
        click.echo("=" * 50)
        click.echo(f"Total documents: {stats.get('total_documents', 'N/A')}")
        click.echo(f"Total chunks: {stats.get('total_chunks', 'N/A')}")
        
        model_info = stats.get('model_info', {})
        if model_info:
            click.echo(f"Model: {model_info.get('model_name', 'N/A')}")
            click.echo(f"Model type: {model_info.get('model_type', 'N/A')}")
        
        processing_params = stats.get('processing_params', {})
        if processing_params:
            click.echo(f"Chunk size: {processing_params.get('chunk_size', 'N/A')}")
            click.echo(f"Chunk overlap: {processing_params.get('chunk_overlap', 'N/A')}")
        
        input_path = stats.get('input_path')
        if input_path:
            click.echo(f"Source: {input_path}")
        
        # Check vector store file
        vector_store_path = Path(index_path) / "vector_store.json"
        if vector_store_path.exists():
            size_mb = vector_store_path.stat().st_size / (1024 * 1024)
            click.echo(f"Index file size: {size_mb:.2f} MB")
        
    except Exception as e:
        logger.error(f"Error reading stats: {e}")
        sys.exit(1)


@cli.command()
@click.argument('index-path', type=click.Path(exists=True))
def serve(index_path: str):
    """
    Start a web server for searching the index.
    
    INDEX_PATH: Path to the index directory
    """
    try:
        from .web_server import create_app
        
        app = create_app(index_path)
        
        click.echo(f"Starting web server for index: {index_path}")
        click.echo("Open http://localhost:8000 in your browser")
        
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
        
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        sys.exit(1)


if __name__ == '__main__':
    cli()