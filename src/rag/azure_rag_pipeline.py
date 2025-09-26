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
from datetime import datetime, timezone

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
        """Extract structured text from HTML SEC filing with section and table markers."""
        soup = BeautifulSoup(html_content, 'html.parser')
        for script in soup(["script", "style"]):
            script.decompose()
        text_parts = []
        processed_elements = set()
        all_elements = soup.find_all(['h1', 'h2', 'h3', 'h4', 'table', 'p', 'div', 'ul', 'ol'])
        for element in all_elements:
            if id(element) in processed_elements:
                continue
            if element.name in ['h1', 'h2', 'h3', 'h4']:
                level = element.name[1]
                header_text = element.get_text(strip=True)
                if header_text and len(header_text) > 3:
                    marker = "\n" + "=" * max(20, 60 - int(level) * 10) + "\n"
                    text_parts.append(f"{marker}SECTION_{level}: {header_text}{marker}")
                processed_elements.add(id(element))
            elif element.name == 'table':
                parent_table = element.find_parent('table')
                if parent_table and id(parent_table) not in processed_elements:
                    continue
                table_text = self._extract_table_text(element)
                if table_text and len(table_text.strip()) > 50:
                    text_parts.append(f"\n[FINANCIAL_TABLE]\n{table_text}\n[/FINANCIAL_TABLE]\n")
                processed_elements.add(id(element))
            elif element.name in ['p', 'div']:
                parent_processed = False
                for parent in element.parents:
                    if id(parent) in processed_elements:
                        parent_processed = True
                        break
                if not parent_processed:
                    para_text = element.get_text(strip=True)
                    if para_text and len(para_text) > 20:
                        text_parts.append(para_text)
                    processed_elements.add(id(element))
            elif element.name in ['ul', 'ol']:
                parent_processed = False
                for parent in element.parents:
                    if id(parent) in processed_elements:
                        parent_processed = True
                        break
                if not parent_processed:
                    list_text = self._extract_list_text(element)
                    if list_text:
                        text_parts.append(list_text)
                    processed_elements.add(id(element))
        return '\n\n'.join(text_parts)

    def _extract_table_text(self, table) -> str:
        rows = []
        headers = []
        for header in table.find_all(['th']):
            header_text = header.get_text(strip=True)
            if header_text:
                headers.append(header_text)
        if headers:
            rows.append(" | ".join(headers))
            rows.append("-" * 50)
        for row in table.find_all('tr'):
            cells = row.find_all(['td'])
            cell_texts = [cell.get_text(strip=True) for cell in cells]
            if cell_texts:
                rows.append(" | ".join(cell_texts))
        return '\n'.join(rows)

    def _extract_list_text(self, element) -> str:
        items = [li.get_text(strip=True) for li in element.find_all('li')]
        if items:
            return '\n'.join([f"- {item}" for item in items])
        return ""

    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split text into chunks with token limits, respecting 10-K document structure."""
        chunks = []
        sections = self._identify_sections(text)
        for section in sections:
            section_chunks = self._chunk_section(section, metadata)
            chunks.extend(section_chunks)
        return chunks

    def _identify_sections(self, text: str) -> List[Dict[str, Any]]:
        sections = []
        current_section = ""
        current_section_title = ""
        current_section_level = 0
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('SECTION_'):
                if current_section.strip():
                    sections.append({
                        'title': current_section_title,
                        'content': current_section.strip(),
                        'level': current_section_level,
                        'type': self._classify_section_type(current_section_title)
                    })
                level_match = line.split('SECTION_')[1].split(':')[0]
                current_section_level = int(level_match) if level_match.isdigit() else 1
                current_section_title = line.split(':', 1)[1].strip() if ':' in line else line
                current_section = ""
            elif line.startswith('='):
                continue
            else:
                if line:
                    current_section += line + '\n'
        if current_section.strip():
            sections.append({
                'title': current_section_title,
                'content': current_section.strip(),
                'level': current_section_level,
                'type': self._classify_section_type(current_section_title)
            })
        if not sections:
            sections.append({
                'title': 'Document Content',
                'content': text,
                'level': 1,
                'type': 'general'
            })
        return sections

    def _classify_section_type(self, title: str) -> str:
        title_lower = title.lower()
        if any(keyword in title_lower for keyword in ['financial', 'statement', 'income', 'balance', 'cash flow']):
            return 'financial'
        elif any(keyword in title_lower for keyword in ['risk', 'factor']):
            return 'risk'
        elif any(keyword in title_lower for keyword in ['business', 'overview', 'operation']):
            return 'business'
        elif any(keyword in title_lower for keyword in ['legal', 'proceeding', 'litigation']):
            return 'legal'
        elif any(keyword in title_lower for keyword in ['management', 'discussion', 'analysis', 'md&a']):
            return 'mda'
        else:
            return 'general'

    def _chunk_section(self, section: Dict, metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        content = section['content']
        section_chunks = []
        if '[FINANCIAL_TABLE]' in content:
            table_chunks = self._chunk_financial_tables(content, section, metadata)
            section_chunks.extend(table_chunks)
            content = re.sub(r'\[FINANCIAL_TABLE\].*?\[/FINANCIAL_TABLE\]', '', content, flags=re.DOTALL)
        if content.strip():
            regular_chunks = self._chunk_regular_content(content, section, metadata)
            section_chunks.extend(regular_chunks)
        return section_chunks

    def _chunk_financial_tables(self, content: str, section: Dict, metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        chunks = []
        table_pattern = r'\[FINANCIAL_TABLE\](.*?)\[/FINANCIAL_TABLE\]'
        tables = re.findall(table_pattern, content, re.DOTALL)
        for i, table_content in enumerate(tables):
            table_content = table_content.strip()
            if not table_content:
                continue
            chunk_metadata = {
                'section_title': section['title'],
                'section_level': section['level'],
                'section_type': section['type'],
                'content_type': 'financial_table',
                'table_index': i
            }
            if metadata:
                chunk_metadata.update(metadata)
            table_tokens = self.tokenizer.encode(table_content)
            if len(table_tokens) <= self.chunk_size:
                chunks.append(self._create_chunk(table_content, chunk_metadata))
            else:
                rows = table_content.split('\n')
                current_chunk = ""
                current_tokens = 0
                for row in rows:
                    row_tokens = len(self.tokenizer.encode(row))
                    if current_tokens + row_tokens > self.chunk_size:
                        if current_chunk:
                            chunks.append(self._create_chunk(current_chunk.strip(), chunk_metadata))
                        current_chunk = row + '\n'
                        current_tokens = row_tokens
                    else:
                        current_chunk += row + '\n'
                        current_tokens += row_tokens
                if current_chunk.strip():
                    chunks.append(self._create_chunk(current_chunk.strip(), chunk_metadata))
        return chunks

    def _chunk_regular_content(self, content: str, section: Dict, metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        chunks = []
        paragraphs = content.split('\n\n')
        current_chunk = ""
        current_tokens = 0
        chunk_metadata = {
            'section_title': section['title'],
            'section_level': section['level'],
            'section_type': section['type'],
            'content_type': 'text'
        }
        if metadata:
            chunk_metadata.update(metadata)
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            paragraph_tokens = len(self.tokenizer.encode(paragraph))
            if paragraph_tokens > self.chunk_size:
                sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', paragraph)
                for sentence in sentences:
                    sentence_tokens = len(self.tokenizer.encode(sentence))
                    if current_tokens + sentence_tokens > self.chunk_size:
                        if current_chunk:
                            chunks.append(self._create_chunk(current_chunk, chunk_metadata))
                        current_chunk = sentence
                        current_tokens = sentence_tokens
                    else:
                        if current_chunk:
                            current_chunk += " " + sentence
                        else:
                            current_chunk = sentence
                        current_tokens += sentence_tokens
            else:
                if current_tokens + paragraph_tokens > self.chunk_size:
                    if current_chunk:
                        chunks.append(self._create_chunk(current_chunk, chunk_metadata))
                    current_chunk = paragraph
                    current_tokens = paragraph_tokens
                else:
                    if current_chunk:
                        current_chunk += "\n\n" + paragraph
                    else:
                        current_chunk = paragraph
                    current_tokens += paragraph_tokens
        if current_chunk:
            chunks.append(self._create_chunk(current_chunk, chunk_metadata))
        return chunks

    def _create_chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        chunk = {
            'text': text.strip(),
            'token_count': len(self.tokenizer.encode(text))
        }
        if metadata:
            chunk.update(metadata)
        chunk['id'] = f"{metadata.get('company', 'unknown')}_{metadata.get('year', 'unknown')}_{metadata.get('chunk_id', 0)}"
        return chunk
    
    def extract_metadata(self, filename: str) -> Dict[str, Any]:
        """Extract metadata from filename."""
        now = datetime.now(timezone.utc).replace(microsecond=0)
        metadata = {
            'filename': filename,
            'processed_date': now.isoformat()
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
                    "fields": "vector",
                    "kind": "vector"
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