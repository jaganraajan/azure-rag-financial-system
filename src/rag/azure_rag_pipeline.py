#!/usr/bin/env python3
"""
Azure RAG Pipeline

This module implements a comprehensive RAG pipeline using Azure AI Search for vector storage
and Azure OpenAI for embeddings and language model capabilities.

Key features:
- Azure AI Search integration for vector embeddings
- Azure OpenAI for text generation and embeddings
- Intelligent document chunking and processing
- Enhanced query capabilities with LangGraph
"""

import os
import logging
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import asyncio
from datetime import datetime

# Azure imports
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration,
)
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential

# OpenAI imports
from openai import AzureOpenAI

# Document processing
import tiktoken
from bs4 import BeautifulSoup
import pandas as pd
import re

# Configuration
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class AzureCredentialManager:
    """Manages Azure credentials and authentication."""
    
    def __init__(self):
        self.search_key = os.getenv('AZURE_SEARCH_ADMIN_KEY')
        self.openai_key = os.getenv('AZURE_OPENAI_API_KEY')
        
    def get_search_credential(self):
        """Get Azure Search credential."""
        if self.search_key:
            return AzureKeyCredential(self.search_key)
        else:
            return DefaultAzureCredential()
    
    def get_openai_client(self):
        """Get Azure OpenAI client."""
        return AzureOpenAI(
            api_key=self.openai_key,
            api_version=os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-01'),
            azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT')
        )


class DocumentProcessor:
    """Processes financial documents for RAG pipeline."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        except KeyError:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def extract_text_from_html(self, html_content: str) -> str:
        """Extract clean text from HTML SEC filing."""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text and clean it
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    
    def extract_metadata(self, filename: str) -> Dict[str, Any]:
        """Extract metadata from filename."""
        metadata = {
            'filename': filename,
            'processed_date': datetime.now().isoformat()
        }
        
        # Extract company symbol from filename (e.g., "GOOGL_10K_2023_xxx.htm")
        if '_' in filename:
            parts = filename.split('_')
            if len(parts) >= 3:
                metadata['company'] = parts[0]
                metadata['filing_type'] = parts[1]
                metadata['year'] = parts[2]
        
        return metadata
    
    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks."""
        # Tokenize the text
        tokens = self.tokenizer.encode(text)
        
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(tokens):
            # Calculate end position
            end = min(start + self.chunk_size, len(tokens))
            
            # Get chunk tokens and decode
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            # Create chunk metadata
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                'chunk_id': chunk_id,
                'start_token': start,
                'end_token': end,
                'token_count': len(chunk_tokens),
                'chunk_text_length': len(chunk_text)
            })
            
            chunks.append({
                'text': chunk_text,
                'metadata': chunk_metadata,
                'id': f"{metadata.get('company', 'unknown')}_{metadata.get('year', 'unknown')}_{chunk_id}"
            })
            
            # Move to next chunk with overlap
            start += self.chunk_size - self.chunk_overlap
            chunk_id += 1
        
        return chunks
    
    def process_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Process a single file and return chunks."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract text and metadata
            text = self.extract_text_from_html(content)
            metadata = self.extract_metadata(Path(file_path).name)
            
            # Create chunks
            chunks = self.chunk_text(text, metadata)
            
            logger.info(f"Processed {file_path}: {len(chunks)} chunks created")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return []


class AzureSearchManager:
    """Manages Azure AI Search operations."""
    
    def __init__(self, service_name: str, index_name: str, credential_manager: AzureCredentialManager):
        self.service_name = service_name
        self.index_name = index_name
        self.credential = credential_manager.get_search_credential()
        
        # Initialize clients
        self.search_client = SearchClient(
            endpoint=f"https://{service_name}.search.windows.net",
            index_name=index_name,
            credential=self.credential
        )
        
        self.index_client = SearchIndexClient(
            endpoint=f"https://{service_name}.search.windows.net",
            credential=self.credential
        )
    
    def create_index(self, vector_dimensions: int = 1536):
        """Create or update the search index."""
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(name="content", type=SearchFieldDataType.String),
            SearchField(
                name="vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=vector_dimensions,
                vector_search_profile_name="myHnswProfile"
            ),
            SimpleField(name="company", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="year", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="filing_type", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="chunk_id", type=SearchFieldDataType.Int32, filterable=True),
            SimpleField(name="token_count", type=SearchFieldDataType.Int32, filterable=True),
            SimpleField(name="processed_date", type=SearchFieldDataType.DateTimeOffset, filterable=True)
        ]
        
        # Configure vector search
        vector_search = VectorSearch(
            profiles=[VectorSearchProfile(
                name="myHnswProfile",
                algorithm_configuration_name="myHnsw"
            )],
            algorithms=[HnswAlgorithmConfiguration(
                name="myHnsw",
                parameters={
                    "m": 4,
                    "efConstruction": 400,
                    "efSearch": 500,
                    "metric": "cosine"
                }
            )]
        )
        
        index = SearchIndex(
            name=self.index_name,
            fields=fields,
            vector_search=vector_search
        )
        
        try:
            result = self.index_client.create_or_update_index(index)
            logger.info(f"Created/updated index: {result.name}")
            return True
        except Exception as e:
            logger.error(f"Error creating index: {e}")
            return False
    
    def upload_documents(self, documents: List[Dict[str, Any]]):
        """Upload documents to Azure Search."""
        try:
            result = self.search_client.upload_documents(documents)
            successful = sum(1 for r in result if r.succeeded)
            logger.info(f"Uploaded {successful}/{len(documents)} documents to Azure Search")
            return successful
        except Exception as e:
            logger.error(f"Error uploading documents: {e}")
            return 0
    
    def search(self, query_vector: List[float], top_k: int = 5, filters: Optional[str] = None) -> List[Dict[str, Any]]:
        """Perform vector search."""
        try:
            search_results = self.search_client.search(
                search_text="",
                vector_queries=[{
                    "vector": query_vector,
                    "k_nearest_neighbors": top_k,
                    "fields": "vector"
                }],
                filter=filters,
                select=["id", "content", "company", "year", "filing_type", "chunk_id"],
                top=top_k
            )
            
            results = []
            for result in search_results:
                results.append({
                    'id': result['id'],
                    'content': result['content'],
                    'metadata': {
                        'company': result.get('company'),
                        'year': result.get('year'),
                        'filing_type': result.get('filing_type'),
                        'chunk_id': result.get('chunk_id')
                    },
                    'score': result.get('@search.score', 0.0)
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error during search: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        try:
            # Get document count
            result = self.search_client.search(search_text="*", include_total_count=True, top=0)
            total_count = result.get_count()
            
            return {
                'total_documents': total_count,
                'index_name': self.index_name,
                'service_name': self.service_name
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {'error': str(e)}


class EmbeddingService:
    """Handles text embedding generation using Azure OpenAI."""
    
    def __init__(self, credential_manager: AzureCredentialManager, deployment_name: str = "text-embedding-ada-002"):
        self.client = credential_manager.get_openai_client()
        self.deployment_name = deployment_name
    
    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text."""
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.deployment_name
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            # Return zero vector as fallback
            return [0.0] * 1536
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        embeddings = []
        for text in texts:
            embedding = self.get_embedding(text)
            embeddings.append(embedding)
        return embeddings


class AzureRAGPipeline:
    """Main Azure RAG Pipeline class."""
    
    def __init__(
        self,
        search_service_name: str,
        search_index_name: str = "financial-documents",
        openai_endpoint: str = None,
        openai_deployment: str = "gpt-4",
        embedding_deployment: str = "text-embedding-ada-002"
    ):
        # Initialize credential manager
        self.credential_manager = AzureCredentialManager()
        
        # Initialize components
        self.document_processor = DocumentProcessor()
        self.search_manager = AzureSearchManager(
            search_service_name, search_index_name, self.credential_manager
        )
        self.embedding_service = EmbeddingService(
            self.credential_manager, embedding_deployment
        )
        self.openai_client = self.credential_manager.get_openai_client()
        self.openai_deployment = openai_deployment
        
        # Create index if it doesn't exist
        self.search_manager.create_index()
    
    def process_directory(self, directory_path: str) -> Dict[str, Any]:
        """Process all files in a directory."""
        directory = Path(directory_path)
        if not directory.exists():
            return {"error": f"Directory {directory_path} does not exist"}
        
        html_files = list(directory.glob("*.htm")) + list(directory.glob("*.html"))
        
        if not html_files:
            return {"error": f"No HTML files found in {directory_path}"}
        
        total_chunks = 0
        processed_files = 0
        uploaded_documents = 0
        
        for file_path in html_files:
            logger.info(f"Processing file: {file_path}")
            
            # Process file into chunks
            chunks = self.document_processor.process_file(file_path)
            
            if chunks:
                # Generate embeddings and prepare documents for upload
                documents = []
                for chunk in chunks:
                    embedding = self.embedding_service.get_embedding(chunk['text'])
                    
                    document = {
                        'id': chunk['id'],
                        'content': chunk['text'],
                        'vector': embedding,
                        'company': chunk['metadata'].get('company'),
                        'year': chunk['metadata'].get('year'),
                        'filing_type': chunk['metadata'].get('filing_type'),
                        'chunk_id': chunk['metadata'].get('chunk_id'),
                        'token_count': chunk['metadata'].get('token_count'),
                        'processed_date': chunk['metadata'].get('processed_date')
                    }
                    documents.append(document)
                
                # Upload to Azure Search
                uploaded = self.search_manager.upload_documents(documents)
                uploaded_documents += uploaded
                total_chunks += len(chunks)
                processed_files += 1
        
        return {
            'total_files': len(html_files),
            'processed_files': processed_files,
            'total_chunks': total_chunks,
            'search_documents': uploaded_documents
        }
    
    def query(self, query_text: str, top_k: int = 5, return_json: bool = False) -> Dict[str, Any]:
        """Query the RAG system."""
        try:
            # Generate query embedding
            query_embedding = self.embedding_service.get_embedding(query_text)
            
            # Search Azure AI Search
            search_results = self.search_manager.search(query_embedding, top_k)
            
            if not search_results:
                if return_json:
                    return {
                        "query": query_text,
                        "answer": "No relevant information found.",
                        "confidence": 0.0,
                        "sources": []
                    }
                else:
                    return {"error": "No relevant information found"}
            
            # Prepare context for LLM
            context = "\n\n".join([
                f"Document {i+1} (Company: {result['metadata'].get('company', 'Unknown')}, Year: {result['metadata'].get('year', 'Unknown')}):\n{result['content']}"
                for i, result in enumerate(search_results)
            ])
            
            # Generate answer using Azure OpenAI
            system_prompt = """You are a financial analyst expert. Use the provided context to answer questions about financial documents. 
            Provide specific, accurate answers based on the context. If the information is not in the context, say so clearly.
            Always cite which company and year you're referencing."""
            
            try:
                response = self.openai_client.chat.completions.create(
                    model=self.openai_deployment,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query_text}"}
                    ],
                    max_tokens=1000,
                    temperature=0.1
                )
                
                answer = response.choices[0].message.content
                
                if return_json:
                    return {
                        "query": query_text,
                        "answer": answer,
                        "confidence": max(result['score'] for result in search_results),
                        "sources": [
                            {
                                "company": result['metadata'].get('company'),
                                "year": result['metadata'].get('year'),
                                "excerpt": result['content'][:200] + "...",
                                "score": result['score']
                            }
                            for result in search_results
                        ]
                    }
                else:
                    return {
                        "answer": answer,
                        "results": [
                            {
                                "text": result['content'],
                                "metadata": result['metadata'],
                                "similarity": result['score']
                            }
                            for result in search_results
                        ]
                    }
                    
            except Exception as e:
                logger.error(f"Error generating answer: {e}")
                if return_json:
                    return {
                        "query": query_text,
                        "answer": "Error generating answer.",
                        "error": str(e),
                        "sources": []
                    }
                else:
                    return {"error": f"Error generating answer: {e}"}
                
        except Exception as e:
            logger.error(f"Error in query processing: {e}")
            return {"error": str(e)}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        return self.search_manager.get_stats()


if __name__ == "__main__":
    # Example usage
    print("Azure RAG Financial System")
    print("This module should be imported and used via main.py")