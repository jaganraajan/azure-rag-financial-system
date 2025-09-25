#!/usr/bin/env python3
"""
Azure RAG Financial System Web Interface

A Flask-based web application for querying financial documents using Azure AI Search
and Azure OpenAI. Provides an intuitive interface for financial analysis and document exploration.
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS

try:
    from rag.azure_rag_pipeline import AzureRAGPipeline
    RAG_AVAILABLE = True
except ImportError as e:
    print(f"Azure RAG pipeline not available: {e}")
    RAG_AVAILABLE = False

try:
    from scrapers.sec_edgar_scraper import SECEdgarScraper
    SCRAPER_AVAILABLE = True
except ImportError as e:
    print(f"SEC scraper not available: {e}")
    SCRAPER_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global RAG pipeline instance
rag_pipeline = None


def initialize_rag_pipeline():
    """Initialize the Azure RAG pipeline."""
    global rag_pipeline
    
    if not RAG_AVAILABLE:
        logger.error("Azure RAG pipeline not available")
        return False
    
    try:
        search_service = os.getenv('AZURE_SEARCH_SERVICE_NAME')
        if not search_service:
            logger.error("AZURE_SEARCH_SERVICE_NAME environment variable not set")
            return False
        
        rag_pipeline = AzureRAGPipeline(
            search_service_name=search_service,
            search_index_name=os.getenv('AZURE_SEARCH_INDEX_NAME', 'financial-documents'),
            openai_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
            openai_deployment=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', 'gpt-4'),
            embedding_deployment=os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT', 'text-embedding-ada-002')
        )
        
        logger.info("Azure RAG pipeline initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize Azure RAG pipeline: {e}")
        return False


@app.route('/')
def index():
    """Main dashboard page."""
    try:
        stats = rag_pipeline.get_stats() if rag_pipeline else {'error': 'RAG pipeline not initialized'}
        return render_template('index.html', stats=stats, rag_available=rag_pipeline is not None)
    except Exception as e:
        logger.error(f"Error loading index page: {e}")
        return render_template('index.html', stats={'error': str(e)}, rag_available=False)


@app.route('/query', methods=['GET', 'POST'])
def query_page():
    """Query interface page."""
    if request.method == 'POST':
        return handle_query()
    
    return render_template('query.html', rag_available=rag_pipeline is not None)


@app.route('/api/query', methods=['POST'])
def handle_query():
    """Handle query requests."""
    if not rag_pipeline:
        return jsonify({'error': 'RAG pipeline not initialized'}), 500
    
    try:
        data = request.get_json()
        query_text = data.get('query', '').strip()
        top_k = data.get('top_k', 5)
        
        if not query_text:
            return jsonify({'error': 'Query text is required'}), 400
        
        # Execute query
        result = rag_pipeline.query(query_text, top_k=top_k, return_json=True)
        
        # Add timestamp
        result['timestamp'] = datetime.now().isoformat()
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats')
def api_stats():
    """API endpoint for system statistics."""
    if not rag_pipeline:
        return jsonify({'error': 'RAG pipeline not initialized'}), 500
    
    try:
        stats = rag_pipeline.get_stats()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/search', methods=['POST'])
def api_search():
    """API endpoint for document search without LLM generation."""
    if not rag_pipeline:
        return jsonify({'error': 'RAG pipeline not initialized'}), 500
    
    try:
        data = request.get_json()
        query_text = data.get('query', '').strip()
        top_k = data.get('top_k', 5)
        
        if not query_text:
            return jsonify({'error': 'Query text is required'}), 400
        
        # Get query embedding
        query_embedding = rag_pipeline.embedding_service.get_embedding(query_text)
        
        # Search Azure AI Search
        search_results = rag_pipeline.search_manager.search(query_embedding, top_k)
        
        return jsonify({
            'query': query_text,
            'results': search_results,
            'total_results': len(search_results),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in search: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/docs')
def docs_page():
    """Documentation page."""
    return render_template('docs.html')


@app.route('/admin')
def admin_page():
    """Admin interface for system management."""
    if not rag_pipeline:
        return render_template('admin.html', rag_available=False, stats={})
    
    try:
        stats = rag_pipeline.get_stats()
        return render_template('admin.html', rag_available=True, stats=stats)
    except Exception as e:
        logger.error(f"Error loading admin page: {e}")
        return render_template('admin.html', rag_available=False, stats={'error': str(e)})


@app.route('/api/process', methods=['POST'])
def api_process():
    """API endpoint to process documents."""
    if not rag_pipeline:
        return jsonify({'error': 'RAG pipeline not initialized'}), 500
    
    try:
        data = request.get_json()
        input_dir = data.get('input_dir', 'demo_filings')
        
        if not os.path.exists(input_dir):
            return jsonify({'error': f'Directory {input_dir} does not exist'}), 400
        
        # Process documents
        results = rag_pipeline.process_directory(input_dir)
        
        return jsonify({
            'success': True,
            'results': results,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error processing documents: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/admin/add-company', methods=['POST'])
def api_admin_add_company():
    """API endpoint to add a new company to the scraper."""
    if not SCRAPER_AVAILABLE:
        return jsonify({'error': 'SEC scraper not available'}), 500
        
    try:
        data = request.get_json()
        symbol = data.get('symbol', '').strip().upper()
        name = data.get('name', '').strip()
        cik = data.get('cik', '').strip()
        
        if not symbol or not name or not cik:
            return jsonify({'error': 'Symbol, name, and CIK are required'}), 400
        
        # Add company to scraper configuration
        SECEdgarScraper.COMPANIES[symbol] = {
            'name': name,
            'cik': cik
        }
        
        logger.info(f"Added new company: {symbol} - {name} (CIK: {cik})")
        
        return jsonify({
            'success': True,
            'message': f'Company {symbol} added successfully',
            'company': {
                'symbol': symbol,
                'name': name,
                'cik': cik
            }
        })
        
    except Exception as e:
        logger.error(f"Error adding company: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/admin/add-years', methods=['POST'])
def api_admin_add_years():
    """API endpoint to add years for companies and trigger scraping/processing."""
    if not SCRAPER_AVAILABLE:
        return jsonify({'error': 'SEC scraper not available'}), 500
        
    try:
        data = request.get_json()
        companies = data.get('companies', [])
        years = data.get('years', [])
        
        if not companies or not years:
            return jsonify({'error': 'Companies and years are required'}), 400
        
        # Initialize scraper
        scraper = SECEdgarScraper()
        
        # Create output directory
        output_dir = 'admin_filings'
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Starting scraping for companies: {companies}, years: {years}")
        
        # Scrape new filings
        scrape_results = scraper.scrape_all_companies(companies, years, output_dir)
        
        # Count successfully downloaded files
        total_files = sum(len(files) for files in scrape_results.values())
        
        if total_files == 0:
            logger.warning("No files were downloaded")
            return jsonify({
                'success': False,
                'error': 'No filings were downloaded. This might be normal if no filings exist for the specified years.'
            })
        
        # Process the downloaded files with RAG pipeline
        processing_results = {'processed_files': 0, 'total_chunks': 0, 'search_documents': 0}
        
        if rag_pipeline and os.path.exists(output_dir) and os.listdir(output_dir):
            try:
                processing_results = rag_pipeline.process_directory(output_dir)
                logger.info(f"Processing completed: {processing_results}")
            except Exception as e:
                logger.error(f"Error during RAG processing: {e}")
                # Continue even if RAG processing fails
        
        return jsonify({
            'success': True,
            'message': f'Successfully processed {total_files} files',
            'scrape_results': scrape_results,
            'processed_files': processing_results.get('processed_files', 0),
            'total_chunks': processing_results.get('total_chunks', 0),
            'search_documents': processing_results.get('search_documents', 0),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in add years process: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/admin/companies', methods=['GET'])
def api_admin_companies():
    """API endpoint to get available companies."""
    if not SCRAPER_AVAILABLE:
        return jsonify({'error': 'SEC scraper not available'}), 500
        
    try:
        companies = SECEdgarScraper.COMPANIES
        return jsonify({
            'success': True,
            'companies': companies
        })
        
    except Exception as e:
        logger.error(f"Error getting companies: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/admin/storage-files', methods=['GET'])
def api_admin_storage_files():
    """API endpoint to list files in Azure Blob Storage."""
    try:
        # Load environment variables from .env if present
        load_dotenv()

        # Get Azure Storage connection string from environment
        connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        if not connection_string:
            return jsonify({
                'success': False,
                'error': 'Azure Storage connection string not configured'
            }), 400
            
        container_name = "filings"
        
        # Initialize Azure Storage client
        from azure.storage.blob import BlobServiceClient
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(container_name)
        
        # List all blobs in the container
        files = []
        try:
            blob_list = container_client.list_blobs()
            for blob in blob_list:
                # Get file info
                blob_properties = container_client.get_blob_client(blob.name).get_blob_properties()
                files.append({
                    'name': blob.name,
                    'size': blob.size,
                    'last_modified': blob.last_modified.isoformat() if blob.last_modified else None,
                    'content_type': blob_properties.content_settings.content_type if hasattr(blob_properties, 'content_settings') else 'application/octet-stream',
                    'url': f"https://{blob_service_client.account_name}.blob.core.windows.net/{container_name}/{blob.name}"
                })
        except Exception as container_error:
            # Container might not exist or might be empty
            if "ContainerNotFound" in str(container_error):
                # Try to create the container
                container_client.create_container()
                files = []
            else:
                raise container_error
                
        return jsonify({
            'success': True,
            'files': files,
            'container': container_name,
            'total_files': len(files)
        })
        
    except Exception as e:
        logger.error(f"Error listing storage files: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/admin/trigger-scraper', methods=['POST'])
def api_admin_trigger_scraper():
    """API endpoint to trigger scraper job for 10K filings."""
    if not SCRAPER_AVAILABLE:
        return jsonify({'error': 'SEC scraper not available'}), 500
    
    try:
        data = request.get_json()
        companies = data.get('companies', [])
        years = data.get('years', [])
        
        if not companies:
            return jsonify({'error': 'Companies are required'}), 400
            
        if not years:
            return jsonify({'error': 'Years are required'}), 400
            
        # Initialize scraper with Azure Storage
        connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        scraper = SECEdgarScraper(
            azure_storage_connection=connection_string,
            azure_container_name="filings"
        )
        
        # Create output directory for local backup
        output_dir = "scraped_filings"
        os.makedirs(output_dir, exist_ok=True)
        
        # Scrape documents
        scrape_results = scraper.scrape_all_companies(companies, years, output_dir)
        
        # Count total files scraped
        total_files = sum(len(files) for files in scrape_results.values())
        
        return jsonify({
            'success': True,
            'message': f'Successfully scraped {total_files} files',
            'scrape_results': scrape_results,
            'total_files': total_files,
            'companies': companies,
            'years': years,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error triggering scraper: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/admin/create-embeddings', methods=['POST'])
def api_admin_create_embeddings():
    """API endpoint to create vector embeddings for selected files."""
    if not rag_pipeline:
        return jsonify({'error': 'RAG pipeline not initialized'}), 500
    
    try:
        data = request.get_json()
        selected_files = data.get('files', [])
        
        if not selected_files:
            return jsonify({'error': 'No files selected'}), 400
            
        # Get Azure Storage connection string from environment
        connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        if not connection_string:
            return jsonify({
                'error': 'Azure Storage connection string not configured'
            }), 400
            
        container_name = "filings"
        
        # Initialize Azure Storage client
        from azure.storage.blob import BlobServiceClient
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(container_name)
        
        # Download selected files and process them
        temp_dir = "/tmp/embeddings_processing"
        os.makedirs(temp_dir, exist_ok=True)
        
        downloaded_files = []
        
        for file_name in selected_files:
            try:
                # Download file from Azure Storage
                blob_client = container_client.get_blob_client(file_name)
                file_content = blob_client.download_blob().readall()
                
                # Save to temporary file
                temp_file_path = os.path.join(temp_dir, file_name)
                with open(temp_file_path, 'wb') as f:
                    f.write(file_content)
                
                downloaded_files.append(temp_file_path)
                logger.info(f"Downloaded file for processing: {file_name}")
                
            except Exception as file_error:
                logger.error(f"Error downloading file {file_name}: {file_error}")
                continue
        
        if not downloaded_files:
            return jsonify({
                'error': 'Failed to download any files for processing'
            }), 400
        
        # Process files through RAG pipeline to create embeddings
        processing_results = rag_pipeline.process_directory(temp_dir)
        
        # Clean up temporary files
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        return jsonify({
            'success': True,
            'message': f'Successfully created embeddings for {len(downloaded_files)} files',
            'processed_files': processing_results.get('processed_files', 0),
            'total_chunks': processing_results.get('total_chunks', 0),
            'search_documents': processing_results.get('search_documents', 0),
            'selected_files': selected_files,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error creating embeddings: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/health')
def health_check():
    """Health check endpoint for Azure App Service."""
    health_status = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'rag_pipeline': rag_pipeline is not None,
        'azure_search': False,
        'azure_openai': False
    }
    
    if rag_pipeline:
        try:
            # Test Azure Search
            stats = rag_pipeline.get_stats()
            health_status['azure_search'] = 'error' not in stats
            
            # Test Azure OpenAI (simple embedding test)
            test_embedding = rag_pipeline.embedding_service.get_embedding("test")
            health_status['azure_openai'] = len(test_embedding) > 0
            
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            health_status['status'] = 'degraded'
            health_status['error'] = str(e)
    
    status_code = 200 if health_status['status'] == 'healthy' else 503
    return jsonify(health_status), status_code


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return render_template('error.html', 
                         error_code=404, 
                         error_message="Page not found"), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return render_template('error.html', 
                         error_code=500, 
                         error_message="Internal server error"), 500


# Template creation functions for deployment
def create_templates():
    """Create HTML templates for the Flask app."""
    templates_dir = Path(__file__).parent / 'templates'
    templates_dir.mkdir(exist_ok=True)
    
    # Base template
    base_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}Azure RAG Financial System{% endblock %}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .navbar-brand { font-weight: bold; }
        .card-metric { font-size: 2rem; font-weight: bold; color: #0066cc; }
        .query-result { background-color: #f8f9fa; border-left: 4px solid #0066cc; padding: 1rem; margin: 1rem 0; }
        .source-card { background-color: #e3f2fd; border: 1px solid #bbdefb; }
        .confidence-high { color: #4caf50; }
        .confidence-medium { color: #ff9800; }
        .confidence-low { color: #f44336; }
        footer { background-color: #f8f9fa; margin-top: 3rem; padding: 2rem 0; }
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-primary">
        <div class="container">
            <a class="navbar-brand" href="/">
                <i class="fas fa-chart-line me-2"></i>Azure RAG Financial System
            </a>
            <div class="navbar-nav ms-auto">
                <a class="nav-link" href="/">Dashboard</a>
                <a class="nav-link" href="/query">Query</a>
                <a class="nav-link" href="/docs">Docs</a>
                <a class="nav-link" href="/admin">Admin</a>
            </div>
        </div>
    </nav>

    <main class="container my-4">
        {% block content %}{% endblock %}
    </main>

    <footer class="bg-light text-center text-muted">
        <div class="container">
            <p>&copy; 2024 Azure RAG Financial System. Powered by Azure AI Search and Azure OpenAI.</p>
        </div>
    </footer>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    {% block scripts %}{% endblock %}
</body>
</html>
    """
    
    # Index template
    index_template = """
{% extends "base.html" %}

{% block title %}Dashboard - Azure RAG Financial System{% endblock %}

{% block content %}
<div class="row">
    <div class="col-md-8">
        <h1><i class="fas fa-tachometer-alt me-2"></i>System Dashboard</h1>
        <p class="lead">Azure-powered financial document analysis and querying system.</p>
    </div>
    <div class="col-md-4">
        <div class="card">
            <div class="card-body text-center">
                {% if rag_available %}
                    <i class="fas fa-check-circle text-success fa-3x"></i>
                    <h5 class="card-title mt-2">System Online</h5>
                    <p class="text-success">Azure RAG pipeline is active</p>
                {% else %}
                    <i class="fas fa-exclamation-triangle text-warning fa-3x"></i>
                    <h5 class="card-title mt-2">System Offline</h5>
                    <p class="text-warning">Azure RAG pipeline not available</p>
                {% endif %}
            </div>
        </div>
    </div>
</div>

<div class="row mt-4">
    {% if stats and 'error' not in stats %}
    <div class="col-md-4">
        <div class="card text-center">
            <div class="card-body">
                <div class="card-metric">{{ stats.total_documents }}</div>
                <h5 class="card-title">Documents</h5>
                <p class="card-text">Indexed in Azure AI Search</p>
            </div>
        </div>
    </div>
    <div class="col-md-4">
        <div class="card text-center">
            <div class="card-body">
                <div class="card-metric">{{ stats.index_name }}</div>
                <h5 class="card-title">Search Index</h5>
                <p class="card-text">Azure AI Search Index</p>
            </div>
        </div>
    </div>
    <div class="col-md-4">
        <div class="card text-center">
            <div class="card-body">
                <div class="card-metric">{{ stats.service_name }}</div>
                <h5 class="card-title">Search Service</h5>
                <p class="card-text">Azure Search Service</p>
            </div>
        </div>
    </div>
    {% else %}
    <div class="col-12">
        <div class="alert alert-warning">
            <i class="fas fa-exclamation-triangle me-2"></i>
            Unable to load system statistics. Please check your Azure configuration.
            {% if stats and stats.error %}
            <br><strong>Error:</strong> {{ stats.error }}
            {% endif %}
        </div>
    </div>
    {% endif %}
</div>

<div class="row mt-4">
    <div class="col-md-6">
        <div class="card">
            <div class="card-header">
                <h5><i class="fas fa-search me-2"></i>Quick Query</h5>
            </div>
            <div class="card-body">
                <p>Try asking questions about financial data:</p>
                <ul>
                    <li>"What are the main revenue sources for Microsoft?"</li>
                    <li>"Compare Google and NVIDIA operating margins"</li>
                    <li>"What risk factors does NVIDIA mention?"</li>
                </ul>
                <a href="/query" class="btn btn-primary">Start Querying</a>
            </div>
        </div>
    </div>
    <div class="col-md-6">
        <div class="card">
            <div class="card-header">
                <h5><i class="fas fa-cogs me-2"></i>System Features</h5>
            </div>
            <div class="card-body">
                <ul>
                    <li><strong>Azure AI Search:</strong> Vector embeddings for semantic search</li>
                    <li><strong>Azure OpenAI:</strong> GPT-4 for intelligent answers</li>
                    <li><strong>SEC Filings:</strong> Real 10-K financial documents</li>
                    <li><strong>Multi-Company:</strong> Google, Microsoft, NVIDIA</li>
                </ul>
                <a href="/docs" class="btn btn-outline-primary">View Documentation</a>
            </div>
        </div>
    </div>
</div>
{% endblock %}
    """
    
    # Query template
    query_template = """
{% extends "base.html" %}

{% block title %}Query - Azure RAG Financial System{% endblock %}

{% block content %}
<div class="row">
    <div class="col-md-8">
        <h1><i class="fas fa-search me-2"></i>Financial Query Interface</h1>
        <p class="lead">Ask questions about financial documents using natural language.</p>
    </div>
    <div class="col-md-4">
        <div class="card">
            <div class="card-body text-center">
                {% if rag_available %}
                    <i class="fas fa-robot text-primary fa-2x"></i>
                    <h6 class="mt-2">Azure AI Ready</h6>
                {% else %}
                    <i class="fas fa-robot text-muted fa-2x"></i>
                    <h6 class="mt-2">AI Unavailable</h6>
                {% endif %}
            </div>
        </div>
    </div>
</div>

{% if not rag_available %}
<div class="alert alert-warning">
    <i class="fas fa-exclamation-triangle me-2"></i>
    Azure RAG pipeline is not available. Please check your configuration.
</div>
{% endif %}

<div class="row mt-4">
    <div class="col-md-8">
        <div class="card">
            <div class="card-header">
                <h5>Ask a Question</h5>
            </div>
            <div class="card-body">
                <form id="queryForm">
                    <div class="mb-3">
                        <textarea 
                            class="form-control" 
                            id="queryText" 
                            rows="3" 
                            placeholder="Enter your question about financial documents..."
                            {% if not rag_available %}disabled{% endif %}
                        ></textarea>
                    </div>
                    <div class="row">
                        <div class="col-md-6">
                            <label for="topK" class="form-label">Number of sources:</label>
                            <select class="form-select" id="topK" {% if not rag_available %}disabled{% endif %}>
                                <option value="3">3</option>
                                <option value="5" selected>5</option>
                                <option value="10">10</option>
                            </select>
                        </div>
                        <div class="col-md-6 d-flex align-items-end">
                            <button 
                                type="submit" 
                                class="btn btn-primary w-100"
                                {% if not rag_available %}disabled{% endif %}
                            >
                                <i class="fas fa-search me-2"></i>Search
                            </button>
                        </div>
                    </div>
                </form>
            </div>
        </div>
        
        <div id="loadingIndicator" class="text-center mt-3" style="display: none;">
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <p class="mt-2">Searching financial documents...</p>
        </div>
        
        <div id="queryResults" class="mt-4"></div>
    </div>
    
    <div class="col-md-4">
        <div class="card">
            <div class="card-header">
                <h5>Example Questions</h5>
            </div>
            <div class="card-body">
                <div class="list-group list-group-flush">
                    <button class="list-group-item list-group-item-action example-query" 
                            data-query="What are Microsoft's main revenue sources?">
                        Microsoft revenue sources
                    </button>
                    <button class="list-group-item list-group-item-action example-query" 
                            data-query="Compare operating margins for Google and NVIDIA">
                        Compare operating margins
                    </button>
                    <button class="list-group-item list-group-item-action example-query" 
                            data-query="What risk factors does NVIDIA mention in their 10-K?">
                        NVIDIA risk factors
                    </button>
                    <button class="list-group-item list-group-item-action example-query" 
                            data-query="How did Google's revenue change from 2022 to 2023?">
                        Google revenue trends
                    </button>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}

{% block scripts %}
<script>
document.addEventListener('DOMContentLoaded', function() {
    const queryForm = document.getElementById('queryForm');
    const queryText = document.getElementById('queryText');
    const topK = document.getElementById('topK');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const queryResults = document.getElementById('queryResults');
    const exampleButtons = document.querySelectorAll('.example-query');
    
    // Handle example query clicks
    exampleButtons.forEach(button => {
        button.addEventListener('click', function() {
            queryText.value = this.dataset.query;
        });
    });
    
    // Handle form submission
    queryForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const query = queryText.value.trim();
        if (!query) {
            alert('Please enter a question');
            return;
        }
        
        // Show loading indicator
        loadingIndicator.style.display = 'block';
        queryResults.innerHTML = '';
        
        // Send query to API
        fetch('/api/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: query,
                top_k: parseInt(topK.value)
            })
        })
        .then(response => response.json())
        .then(data => {
            loadingIndicator.style.display = 'none';
            displayResults(data);
        })
        .catch(error => {
            loadingIndicator.style.display = 'none';
            console.error('Error:', error);
            queryResults.innerHTML = '<div class="alert alert-danger">Error processing query: ' + error + '</div>';
        });
    });
    
    function displayResults(data) {
        if (data.error) {
            queryResults.innerHTML = '<div class="alert alert-danger">' + data.error + '</div>';
            return;
        }
        
        const confidence = data.confidence || 0;
        const confidenceClass = confidence > 0.7 ? 'confidence-high' : confidence > 0.4 ? 'confidence-medium' : 'confidence-low';
        
        let html = '<div class="query-result">';
        html += '<h4><i class="fas fa-lightbulb me-2"></i>Answer</h4>';
        html += '<p class="lead">' + (data.answer || 'No answer generated') + '</p>';
        html += '<div class="row mt-3">';
        html += '<div class="col-md-6">';
        html += '<small class="text-muted">Confidence: <span class="' + confidenceClass + '">' + (confidence * 100).toFixed(1) + '%</span></small>';
        html += '</div>';
        html += '<div class="col-md-6 text-end">';
        html += '<small class="text-muted">Sources: ' + (data.sources ? data.sources.length : 0) + '</small>';
        html += '</div>';
        html += '</div>';
        html += '</div>';
        
        if (data.sources && data.sources.length > 0) {
            html += '<h5 class="mt-4"><i class="fas fa-file-alt me-2"></i>Sources</h5>';
            data.sources.forEach((source, index) => {
                html += '<div class="card source-card mt-2">';
                html += '<div class="card-body">';
                html += '<h6 class="card-title">' + (source.company || 'Unknown') + ' (' + (source.year || 'Unknown') + ')</h6>';
                html += '<p class="card-text">' + (source.excerpt || '') + '</p>';
                html += '<small class="text-muted">Score: ' + (source.score || 0).toFixed(3) + '</small>';
                html += '</div>';
                html += '</div>';
            });
        }
        
        queryResults.innerHTML = html;
    }
});
</script>
{% endblock %}
    """
    
    # Write templates
    with open(templates_dir / 'base.html', 'w') as f:
        f.write(base_template)
    
    with open(templates_dir / 'index.html', 'w') as f:
        f.write(index_template)
    
    with open(templates_dir / 'query.html', 'w') as f:
        f.write(query_template)
    
    logger.info("Created HTML templates")


if __name__ == '__main__':
    # Create templates
    create_templates()
    
    # Initialize RAG pipeline
    if initialize_rag_pipeline():
        logger.info("Starting Flask web application")
        port = int(os.getenv('FLASK_PORT', 8080))
        app.run(
            host='0.0.0.0',
            port=port,
            debug=os.getenv('FLASK_ENV') == 'development'
        )
    else:
        logger.error("Failed to initialize RAG pipeline. Please check your Azure configuration.")
        print("\nPlease ensure you have set the following environment variables:")
        print("- AZURE_SEARCH_SERVICE_NAME")
        print("- AZURE_SEARCH_ADMIN_KEY")
        print("- AZURE_OPENAI_ENDPOINT")
        print("- AZURE_OPENAI_API_KEY")