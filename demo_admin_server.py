#!/usr/bin/env python3
"""
Simple mock web server to demonstrate admin page functionality
"""
import os
import json
import tempfile
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import html

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

class MockAdminHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = urlparse(self.path).path
        
        if path == '/admin':
            self.serve_admin_page()
        elif path == '/api/admin/companies':
            self.serve_companies_api()
        elif path.startswith('/static/'):
            self.serve_static()
        else:
            self.serve_404()
    
    def do_POST(self):
        path = urlparse(self.path).path
        
        if path == '/api/admin/add-company':
            self.handle_add_company()
        elif path == '/api/admin/add-years':
            self.handle_add_years()
        else:
            self.serve_404()
    
    def serve_admin_page(self):
        """Serve the admin page with mock data"""
        admin_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Admin - Azure RAG Financial System (DEMO)</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .navbar-brand { font-weight: bold; }
        .card-metric { font-size: 2rem; font-weight: bold; color: #0066cc; }
        .demo-badge { position: fixed; top: 10px; right: 10px; z-index: 1000; }
    </style>
</head>
<body>
    <div class="alert alert-info demo-badge">
        <i class="fas fa-flask me-2"></i>DEMO MODE
    </div>
    
    <nav class="navbar navbar-expand-lg navbar-dark bg-primary">
        <div class="container">
            <a class="navbar-brand" href="/">
                <i class="fas fa-chart-line me-2"></i>Azure RAG Financial System
            </a>
            <div class="navbar-nav ms-auto">
                <a class="nav-link" href="/">Dashboard</a>
                <a class="nav-link" href="/query">Query</a>
                <a class="nav-link" href="/docs">Docs</a>
                <a class="nav-link active" href="/admin">Admin</a>
            </div>
        </div>
    </nav>

    <main class="container my-4">
        <div class="row">
            <div class="col-md-12">
                <h1><i class="fas fa-cogs me-2"></i>Admin Panel</h1>
                <p class="lead">Manage companies and 10-K filings for the RAG system.</p>
            </div>
        </div>

        <!-- System Status -->
        <div class="row mt-4">
            <div class="col-md-12">
                <div class="card">
                    <div class="card-header">
                        <h5><i class="fas fa-info-circle me-2"></i>System Status</h5>
                    </div>
                    <div class="card-body">
                        <div class="alert alert-success">
                            <i class="fas fa-check-circle me-2"></i>Azure RAG pipeline is active and ready (Demo Mode)
                        </div>
                        <div class="row">
                            <div class="col-md-3">
                                <div class="text-center">
                                    <div class="h3 text-primary">150</div>
                                    <small class="text-muted">Total Documents</small>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="text-center">
                                    <div class="h3 text-primary">demo-search</div>
                                    <small class="text-muted">Search Service</small>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="text-center">
                                    <div class="h3 text-primary">financial-docs</div>
                                    <small class="text-muted">Search Index</small>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="text-center">
                                    <div class="h3 text-primary">Active</div>
                                    <small class="text-muted">Status</small>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Company Management -->
        <div class="row mt-4">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5><i class="fas fa-building me-2"></i>Add New Company</h5>
                    </div>
                    <div class="card-body">
                        <form id="addCompanyForm">
                            <div class="mb-3">
                                <label for="companySymbol" class="form-label">Company Symbol</label>
                                <input type="text" class="form-control" id="companySymbol" placeholder="e.g., AAPL" required>
                                <div class="form-text">Stock ticker symbol (e.g., GOOGL, MSFT, AAPL)</div>
                            </div>
                            <div class="mb-3">
                                <label for="companyName" class="form-label">Company Name</label>
                                <input type="text" class="form-control" id="companyName" placeholder="e.g., Apple Inc." required>
                            </div>
                            <div class="mb-3">
                                <label for="companyCik" class="form-label">CIK Code</label>
                                <input type="text" class="form-control" id="companyCik" placeholder="e.g., 0000320193" required>
                                <div class="form-text">SEC Central Index Key (CIK) for the company</div>
                            </div>
                            <button type="submit" class="btn btn-primary">
                                <i class="fas fa-plus me-2"></i>Add Company
                            </button>
                        </form>
                    </div>
                </div>
            </div>

            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5><i class="fas fa-calendar me-2"></i>Add Years for Existing Companies</h5>
                    </div>
                    <div class="card-body">
                        <form id="addYearsForm">
                            <div class="mb-3">
                                <label for="selectedCompanies" class="form-label">Select Companies</label>
                                <select class="form-select" id="selectedCompanies" multiple required>
                                    <!-- Companies will be loaded dynamically -->
                                </select>
                                <div class="form-text">Hold Ctrl/Cmd to select multiple companies</div>
                            </div>
                            <div class="mb-3">
                                <label for="selectedYears" class="form-label">Years to Add</label>
                                <div class="row">
                                    <div class="col-md-6">
                                        <div class="form-check">
                                            <input class="form-check-input" type="checkbox" value="2021" id="year2021">
                                            <label class="form-check-label" for="year2021">2021</label>
                                        </div>
                                        <div class="form-check">
                                            <input class="form-check-input" type="checkbox" value="2020" id="year2020">
                                            <label class="form-check-label" for="year2020">2020</label>
                                        </div>
                                        <div class="form-check">
                                            <input class="form-check-input" type="checkbox" value="2019" id="year2019">
                                            <label class="form-check-label" for="year2019">2019</label>
                                        </div>
                                    </div>
                                    <div class="col-md-6">
                                        <div class="form-check">
                                            <input class="form-check-input" type="checkbox" value="2018" id="year2018">
                                            <label class="form-check-label" for="year2018">2018</label>
                                        </div>
                                        <div class="form-check">
                                            <input class="form-check-input" type="checkbox" value="2017" id="year2017">
                                            <label class="form-check-label" for="year2017">2017</label>
                                        </div>
                                        <div class="form-check">
                                            <input class="form-check-input" type="checkbox" value="2016" id="year2016">
                                            <label class="form-check-label" for="year2016">2016</label>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <button type="submit" class="btn btn-success">
                                <i class="fas fa-download me-2"></i>Add Years & Process
                            </button>
                        </form>
                    </div>
                </div>
            </div>
        </div>

        <!-- Processing Status -->
        <div class="row mt-4">
            <div class="col-md-12">
                <div class="card">
                    <div class="card-header">
                        <h5><i class="fas fa-tasks me-2"></i>Processing Status</h5>
                    </div>
                    <div class="card-body">
                        <div id="processingStatus" class="d-none">
                            <div class="alert alert-info">
                                <div class="d-flex align-items-center">
                                    <div class="spinner-border spinner-border-sm me-3" role="status"></div>
                                    <div>
                                        <strong>Processing...</strong>
                                        <div id="processingMessage">Preparing to scrape and process filings...</div>
                                    </div>
                                </div>
                            </div>
                            <div class="progress mb-3">
                                <div class="progress-bar progress-bar-striped progress-bar-animated" 
                                     id="processingProgress" role="progressbar" style="width: 0%"></div>
                            </div>
                        </div>
                        <div id="processingResults"></div>
                    </div>
                </div>
            </div>
        </div>
    </main>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        const addCompanyForm = document.getElementById('addCompanyForm');
        const addYearsForm = document.getElementById('addYearsForm');
        const processingStatus = document.getElementById('processingStatus');
        const processingMessage = document.getElementById('processingMessage');
        const processingProgress = document.getElementById('processingProgress');
        const processingResults = document.getElementById('processingResults');

        // Load available companies on page load
        loadAvailableCompanies();

        function loadAvailableCompanies() {
            fetch('/api/admin/companies')
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    const select = document.getElementById('selectedCompanies');
                    select.innerHTML = ''; // Clear existing options
                    
                    Object.entries(data.companies).forEach(([symbol, company]) => {
                        const option = document.createElement('option');
                        option.value = symbol;
                        option.textContent = `${symbol} - ${company.name}`;
                        select.appendChild(option);
                    });
                }
            })
            .catch(error => {
                console.error('Error loading companies:', error);
            });
        }

        // Add Company Form Handler
        addCompanyForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const symbol = document.getElementById('companySymbol').value.trim().toUpperCase();
            const name = document.getElementById('companyName').value.trim();
            const cik = document.getElementById('companyCik').value.trim();
            
            if (!symbol || !name || !cik) {
                alert('Please fill in all fields');
                return;
            }
            
            showProcessingStatus('Adding new company...');
            
            fetch('/api/admin/add-company', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    symbol: symbol,
                    name: name,
                    cik: cik
                })
            })
            .then(response => response.json())
            .then(data => {
                hideProcessingStatus();
                if (data.success) {
                    showResults(`<div class="alert alert-success">
                        <i class="fas fa-check me-2"></i>Company ${symbol} added successfully! (Demo)
                    </div>`);
                    addCompanyForm.reset();
                    loadAvailableCompanies();
                } else {
                    showResults(`<div class="alert alert-danger">
                        <i class="fas fa-times me-2"></i>Error: ${data.error || 'Failed to add company'}
                    </div>`);
                }
            })
            .catch(error => {
                hideProcessingStatus();
                console.error('Error:', error);
                showResults(`<div class="alert alert-danger">
                    <i class="fas fa-times me-2"></i>Error adding company: ${error.message}
                </div>`);
            });
        });

        // Add Years Form Handler
        addYearsForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const selectedCompanies = Array.from(document.getElementById('selectedCompanies').selectedOptions)
                .map(option => option.value);
            const selectedYears = Array.from(document.querySelectorAll('input[type="checkbox"]:checked'))
                .map(checkbox => parseInt(checkbox.value));
            
            if (selectedCompanies.length === 0) {
                alert('Please select at least one company');
                return;
            }
            
            if (selectedYears.length === 0) {
                alert('Please select at least one year');
                return;
            }
            
            showProcessingStatus('Starting scraping and processing...');
            updateProgress(10, 'Initializing scraper...');
            
            // Simulate processing steps
            setTimeout(() => updateProgress(30, 'Downloading SEC filings...'), 1000);
            setTimeout(() => updateProgress(60, 'Processing documents...'), 2000);
            setTimeout(() => updateProgress(80, 'Generating embeddings...'), 3000);
            setTimeout(() => updateProgress(100, 'Storing in vector database...'), 4000);
            
            setTimeout(() => {
                hideProcessingStatus();
                showResults(`<div class="alert alert-success">
                    <i class="fas fa-check me-2"></i>Successfully processed demo files and generated embeddings! (Demo Mode)
                    <br><small>Companies: ${selectedCompanies.join(', ')} | Years: ${selectedYears.join(', ')}</small>
                </div>`);
                addYearsForm.reset();
            }, 5000);
        });

        function showProcessingStatus(message) {
            processingMessage.textContent = message;
            processingStatus.classList.remove('d-none');
            processingResults.innerHTML = '';
            updateProgress(0);
        }

        function hideProcessingStatus() {
            processingStatus.classList.add('d-none');
        }

        function updateProgress(percent, message) {
            processingProgress.style.width = percent + '%';
            if (message) {
                processingMessage.textContent = message;
            }
        }

        function showResults(html) {
            processingResults.innerHTML = html;
        }
    });
    </script>
</body>
</html>
        """
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(admin_html.encode())
    
    def serve_companies_api(self):
        """Serve mock companies API"""
        # Load companies from scraper
        try:
            from scrapers.sec_edgar_scraper import SECEdgarScraper
            companies = SECEdgarScraper.COMPANIES
        except:
            companies = {
                'GOOGL': {'name': 'Alphabet Inc.', 'cik': '1652044'},
                'MSFT': {'name': 'Microsoft Corporation', 'cik': '789019'},
                'NVDA': {'name': 'NVIDIA Corporation', 'cik': '1045810'}
            }
        
        response = {
            'success': True,
            'companies': companies
        }
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())
    
    def handle_add_company(self):
        """Handle add company request"""
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data.decode())
        
        # Add to scraper in demo mode
        try:
            from scrapers.sec_edgar_scraper import SECEdgarScraper
            symbol = data['symbol'].upper()
            SECEdgarScraper.COMPANIES[symbol] = {
                'name': data['name'],
                'cik': data['cik']
            }
            success = True
        except:
            success = False
        
        response = {
            'success': success,
            'message': f"Company {data['symbol']} added successfully (Demo)"
        }
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())
    
    def handle_add_years(self):
        """Handle add years request"""
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data.decode())
        
        response = {
            'success': True,
            'message': f"Successfully processed demo files for {len(data['companies'])} companies",
            'processed_files': len(data['companies']) * len(data['years']),
            'total_chunks': len(data['companies']) * len(data['years']) * 50  # Mock chunks
        }
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())
    
    def serve_static(self):
        """Serve static files"""
        self.send_response(404)
        self.end_headers()
    
    def serve_404(self):
        """Serve 404 page"""
        self.send_response(404)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(b'<h1>404 Not Found</h1>')

if __name__ == '__main__':
    port = 8080
    server = HTTPServer(('localhost', port), MockAdminHandler)
    print(f"🚀 Mock Admin Demo Server running at http://localhost:{port}/admin")
    print("📝 Features demonstrated:")
    print("   - Company management interface")
    print("   - Year selection for existing companies")  
    print("   - Processing status indicators")
    print("   - Real scraper integration")
    print("\n🛑 Press Ctrl+C to stop")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")
        server.shutdown()