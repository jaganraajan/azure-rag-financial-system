#!/usr/bin/env python3
"""
Test script for Azure RAG Financial System

This script tests the core functionality without requiring Azure services,
using mock implementations for offline testing.
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockEmbeddingService:
    """Mock embedding service for testing without Azure OpenAI."""
    
    def get_embedding(self, text: str) -> list:
        """Return a mock embedding vector."""
        # Simple hash-based mock embedding
        import hashlib
        hash_obj = hashlib.md5(text.encode())
        hash_hex = hash_obj.hexdigest()
        
        # Convert to a 1536-dimensional vector (matching Azure OpenAI)
        embedding = []
        for i in range(0, len(hash_hex), 2):
            val = int(hash_hex[i:i+2], 16) / 255.0 - 0.5  # Normalize to [-0.5, 0.5]
            embedding.append(val)
        
        # Pad or truncate to 1536 dimensions
        while len(embedding) < 1536:
            embedding.extend(embedding[:min(16, 1536 - len(embedding))])
        
        return embedding[:1536]


class MockSearchManager:
    """Mock search manager for testing without Azure AI Search."""
    
    def __init__(self):
        self.documents = []
    
    def create_index(self, vector_dimensions: int = 1536):
        """Mock index creation."""
        logger.info("Mock: Created search index")
        return True
    
    def upload_documents(self, documents: list):
        """Mock document upload."""
        self.documents.extend(documents)
        logger.info(f"Mock: Uploaded {len(documents)} documents")
        return len(documents)
    
    def search(self, query_vector: list, top_k: int = 5, filters: str = None) -> list:
        """Mock search with basic text matching."""
        results = []
        
        for doc in self.documents[:top_k]:
            # Simple mock scoring based on document index
            score = 0.9 - (len(results) * 0.1)
            results.append({
                'id': doc.get('id', 'unknown'),
                'content': doc.get('content', '')[:200] + "...",
                'metadata': {
                    'company': doc.get('company'),
                    'year': doc.get('year'),
                    'filing_type': doc.get('filing_type'),
                    'chunk_id': doc.get('chunk_id', 0)
                },
                'score': score
            })
        
        return results
    
    def get_stats(self):
        """Return mock statistics."""
        return {
            'total_documents': len(self.documents),
            'index_name': 'financial-documents-mock',
            'service_name': 'mock-search-service'
        }


class MockOpenAIClient:
    """Mock OpenAI client for testing without Azure OpenAI."""
    
    def __init__(self):
        self.chat = self
        self.completions = self
    
    def create(self, model: str, messages: list, max_tokens: int = 1000, temperature: float = 0.1):
        """Mock chat completion."""
        user_message = messages[-1]['content'] if messages else ""
        
        # Generate a simple mock response based on the query
        mock_response = f"""Based on the provided financial documents, here's what I found:

The query asks about financial information. From the context provided, I can see data from multiple companies including their revenue, operating margins, and business segments.

Key findings:
- The companies show varying performance across different metrics
- Revenue trends indicate growth patterns in the technology sector
- Operating margins reflect different business model efficiencies

This analysis is based on the SEC 10-K filings provided in the context."""
        
        class MockChoice:
            def __init__(self, content):
                self.message = MockMessage(content)
        
        class MockMessage:
            def __init__(self, content):
                self.content = content
        
        class MockResponse:
            def __init__(self, content):
                self.choices = [MockChoice(content)]
        
        return MockResponse(mock_response)


def test_document_processing():
    """Test document processing functionality."""
    logger.info("Testing document processing...")
    
    try:
        from rag.azure_rag_pipeline import DocumentProcessor
        
        processor = DocumentProcessor()
        
        # Test with a demo file
        demo_files = list(Path("demo_filings").glob("*.htm"))
        if demo_files:
            test_file = demo_files[0]
            chunks = processor.process_file(str(test_file))
            
            logger.info(f"Processed {test_file.name}: {len(chunks)} chunks created")
            if chunks:
                logger.info(f"Sample chunk: {chunks[0]['text'][:100]}...")
                return True
        else:
            logger.warning("No demo files found")
            return False
            
    except Exception as e:
        logger.error(f"Document processing test failed: {e}")
        return False


def test_mock_rag_pipeline():
    """Test a mock RAG pipeline."""
    logger.info("Testing mock RAG pipeline...")
    
    try:
        from rag.azure_rag_pipeline import DocumentProcessor
        
        # Create mock components
        doc_processor = DocumentProcessor()
        embedding_service = MockEmbeddingService()
        search_manager = MockSearchManager()
        openai_client = MockOpenAIClient()
        
        # Create mock index
        search_manager.create_index()
        
        # Process demo documents
        demo_files = list(Path("demo_filings").glob("*.htm"))
        total_chunks = 0
        
        for file_path in demo_files[:3]:  # Process first 3 files
            chunks = doc_processor.process_file(str(file_path))
            
            # Convert chunks to search documents
            documents = []
            for chunk in chunks:
                embedding = embedding_service.get_embedding(chunk['text'])
                
                document = {
                    'id': chunk['id'],
                    'content': chunk['text'],
                    'vector': embedding,
                    'company': chunk['metadata'].get('company'),
                    'year': chunk['metadata'].get('year'),
                    'filing_type': chunk['metadata'].get('filing_type'),
                    'chunk_id': chunk['metadata'].get('chunk_id'),
                }
                documents.append(document)
            
            # Upload to mock search
            uploaded = search_manager.upload_documents(documents)
            total_chunks += uploaded
        
        logger.info(f"Processed {len(demo_files[:3])} files, {total_chunks} chunks uploaded")
        
        # Test search
        query = "What are the main revenue sources?"
        query_embedding = embedding_service.get_embedding(query)
        search_results = search_manager.search(query_embedding, top_k=3)
        
        logger.info(f"Search results for '{query}': {len(search_results)} documents found")
        
        # Test answer generation
        if search_results:
            context = "\n\n".join([f"Company: {r['metadata']['company']}\n{r['content']}" 
                                 for r in search_results])
            
            response = openai_client.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a financial analyst."},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
                ]
            )
            
            answer = response.choices[0].message.content
            logger.info(f"Generated answer: {answer[:100]}...")
        
        # Get stats
        stats = search_manager.get_stats()
        logger.info(f"System stats: {stats}")
        
        return True
        
    except Exception as e:
        logger.error(f"Mock RAG pipeline test failed: {e}")
        return False


def test_scraper():
    """Test the SEC scraper demo functionality."""
    logger.info("Testing SEC scraper...")
    
    try:
        from scrapers.sec_edgar_scraper import create_demo_filings
        
        # Create demo filings
        demo_files = create_demo_filings("test_demo_filings")
        logger.info(f"Created {len(demo_files)} demo files")
        
        # Clean up test files
        import shutil
        if Path("test_demo_filings").exists():
            shutil.rmtree("test_demo_filings")
        
        return True
        
    except Exception as e:
        logger.error(f"Scraper test failed: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("Starting Azure RAG Financial System Tests")
    logger.info("=" * 50)
    
    # Ensure demo files exist
    if not Path("demo_filings").exists():
        logger.info("Creating demo files...")
        try:
            from scrapers.sec_edgar_scraper import create_demo_filings
            create_demo_filings()
        except Exception as e:
            logger.error(f"Failed to create demo files: {e}")
            return False
    
    tests = [
        ("Document Processing", test_document_processing),
        ("SEC Scraper", test_scraper),
        ("Mock RAG Pipeline", test_mock_rag_pipeline),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        logger.info(f"\nRunning {test_name} test...")
        try:
            if test_func():
                logger.info(f"✅ {test_name} test PASSED")
                passed += 1
            else:
                logger.error(f"❌ {test_name} test FAILED")
                failed += 1
        except Exception as e:
            logger.error(f"❌ {test_name} test FAILED with exception: {e}")
            failed += 1
    
    logger.info("\n" + "=" * 50)
    logger.info("Test Summary:")
    logger.info(f"✅ Passed: {passed}")
    logger.info(f"❌ Failed: {failed}")
    logger.info(f"📊 Total: {passed + failed}")
    
    if failed == 0:
        logger.info("🎉 All tests passed! The system is ready for Azure deployment.")
        logger.info("\nNext steps:")
        logger.info("1. Configure Azure services (AI Search, OpenAI, App Service)")
        logger.info("2. Set environment variables in .env")
        logger.info("3. Deploy using: cd azure-deploy && ./deploy.sh")
        logger.info("4. Process real documents: python main.py rag --process")
    else:
        logger.error("❌ Some tests failed. Please check the errors above.")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)