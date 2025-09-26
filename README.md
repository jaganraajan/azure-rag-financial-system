# Azure RAG Financial System

A comprehensive financial analysis system that combines **Azure AI Search** for vector embeddings with **Azure OpenAI** for intelligent query processing. This system processes SEC EDGAR 10-K filings, creates searchable vector embeddings in Azure AI Search, and provides intelligent answers to complex financial questions about Google, Microsoft, and NVIDIA using their recent 10-K filings.

> **Note:** This system is designed for Azure deployment and leverages Azure's AI and cloud services for scalable, production-ready financial document analysis.

## 🚀 Key Features

### 🔍 Azure-Powered Intelligence
- **Azure AI Search**: Vector embeddings for semantic document search
- **Azure OpenAI**: GPT-4 for intelligent answer generation
- **Azure Storage**: Scalable document storage and management
- **Azure Key Vault**: Secure credential management

### 📊 Financial Document Processing
- **SEC EDGAR Integration**: Downloads 10-K filings from SEC database
- **Multi-Company Support**: Google (GOOGL), Microsoft (MSFT), NVIDIA (NVDA)
- **Multi-Year Coverage**: Supports 2022, 2023, and 2024 filings
- **Intelligent Chunking**: Semantic text splitting with overlap for optimal retrieval

### 🤖 Advanced Query Capabilities (Feature available in https://github.com/jaganraajan/agent-rag-financial-system)
- **Natural Language Queries**: Ask questions in plain English
- **Comparative Analysis**: "Which company had higher operating margins?"
- **Temporal Analysis**: "How did revenue change from 2022 to 2023?"
- **Multi-Company Insights**: Cross-company financial comparisons
- **Source Attribution**: Tracks and cites specific document sections

### 🌐 Web Interface
- **Interactive Dashboard**: Real-time system statistics and health monitoring
- **Query Interface**: User-friendly interface for financial questions
- **Search Results**: Detailed answers with confidence scores and sources
- **Admin Panel**: System management and document processing

## Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Interface │    │   Flask App      │    │  Azure AI Search│
│   (Bootstrap)   │◄──►│   (Python)       │◄──►│  (Vector Store) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                               │
                               ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │  Azure OpenAI    │    │  Azure Storage  │
                       │  (GPT-4 + Embed.)│    │  (Documents)    │
                       └──────────────────┘    └─────────────────┘
```

## Quick Start

### 1. Prerequisites

- **Azure Subscription** with the following services:
  - Azure AI Search (Basic tier or higher)
  - Azure OpenAI (with GPT-4 and text-embedding-ada-002 deployments)
  - Azure App Service (for web hosting)
  - Azure Storage Account (optional, for document storage)
  - Azure Key Vault (for secure credential management)

- **Local Development**:
  - Python 3.11+
  - Azure CLI
  - Git

### 2. Azure Deployment (Recommended)

Deploy the entire system to Azure using the provided ARM template:

```bash
# Clone the repository
git clone https://github.com/jaganraajan/azure-rag-financial-system.git
cd azure-rag-financial-system

# Deploy to Azure (creates all required resources)
cd azure-deploy
./deploy.sh
```

The deployment script will:
- Create Azure resource group
- Deploy Azure AI Search, OpenAI, App Service, Storage, and Key Vault
- Configure application settings
- Provide next steps for model deployment

### 3. Local Development Setup

For local development and testing:

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your Azure service details

# Create demo data for testing
python -c "from src.scrapers.sec_edgar_scraper import create_demo_filings; create_demo_filings()"

# Process documents into Azure AI Search
python main.py rag --process --input-dir demo_filings

# Start interactive query mode
python main.py rag

# Or run the web interface
python src/web/flask_app.py
```

## Usage Examples

### Command Line Interface

```bash
# Download real SEC filings
python main.py scrape --companies GOOGL MSFT NVDA --years 2023 2024

# Process documents for search
python main.py rag --process --input-dir filings

# Query examples
python main.py rag --query "What are Microsoft's main revenue sources?"
python main.py rag --query "Compare operating margins for Google and NVIDIA"
python main.py rag --query "What risk factors does NVIDIA mention?"

# Interactive mode
python main.py rag
```

### Web Interface

Access the web interface at `http://localhost:8080` (local) or your Azure App Service URL:

- **Dashboard**: System statistics and health monitoring
- **Query Interface**: Natural language financial queries
- **Search Results**: AI-generated answers with sources and confidence scores

### Example Queries

**Comparative Analysis:**
- "Which company had the highest operating margin in 2023?"
- "Compare Microsoft and Google revenue growth"

**Temporal Analysis:**
- "How did NVIDIA's revenue change from 2022 to 2023?"
- "What was Google's revenue trend over the past three years?"

**Risk Analysis:**
- "What are the main risk factors mentioned by Microsoft?"
- "How do the companies compare in terms of regulatory risks?"

**Business Intelligence:**
- "What are NVIDIA's primary revenue sources?"
- "Which company has the strongest cloud business?"

## Configuration

### Environment Variables

Set these environment variables for Azure integration:

```bash
# Azure AI Search
AZURE_SEARCH_SERVICE_NAME=your-search-service
AZURE_SEARCH_ADMIN_KEY=your-search-key
AZURE_SEARCH_INDEX_NAME=financial-documents

# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://your-openai.openai.azure.com/
AZURE_OPENAI_API_KEY=your-openai-key
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002
AZURE_OPENAI_API_VERSION=2024-02-01

# Azure Storage (optional)
AZURE_STORAGE_CONNECTION_STRING=your-storage-connection

# Application settings
LOG_LEVEL=INFO
FLASK_PORT=8080
```

### Azure OpenAI Model Deployments

Ensure you have deployed these models in Azure OpenAI Studio:

1. **GPT-4** (for answer generation)
   - Deployment name: `gpt-4`
   - Model: `gpt-4` or `gpt-4-32k`

2. **Text Embedding Ada 002** (for vector embeddings)
   - Deployment name: `text-embedding-ada-002`
   - Model: `text-embedding-ada-002`

## Project Structure

```
azure-rag-financial-system/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── main.py                     # Main CLI interface
├── .env.example               # Environment variables template
├── startup.txt                # Azure App Service startup command
├── src/                       # Source code
│   ├── rag/                  # RAG pipeline components
│   │   └── azure_rag_pipeline.py   # Main Azure RAG implementation
│   ├── scrapers/             # SEC EDGAR scraper
│   │   └── sec_edgar_scraper.py    # SEC filing downloader
│   ├── web/                  # Web interface
│   │   └── flask_app.py           # Flask web application
│   └── utils/                # Utility functions
├── azure-deploy/             # Azure deployment files
│   ├── azuredeploy.json          # ARM template
│   ├── azuredeploy.parameters.json # ARM parameters
│   └── deploy.sh                  # Deployment script
├── demo_filings/             # Demo files (created by scraper)
└── filings/                  # Downloaded SEC filings
```

## API Reference

### REST API Endpoints

The web interface provides RESTful API endpoints:

- `GET /` - Dashboard page
- `GET /query` - Query interface page
- `POST /api/query` - Execute financial query
- `POST /api/search` - Document search without LLM
- `GET /api/stats` - System statistics
- `POST /api/process` - Process new documents
- `GET /health` - Health check endpoint

### Example API Usage

```bash
# Query API
curl -X POST http://localhost:8080/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are Microsoft revenue sources?", "top_k": 5}'

# Search API
curl -X POST http://localhost:8080/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "operating margin", "top_k": 3}'
```

## Performance and Scaling

### Azure AI Search Performance
- **Basic Tier**: Up to 2GB storage, 3 replicas
- **Standard Tier**: Up to 25GB storage, 12 replicas
- **Optimized for**: Financial document semantic search

### Azure OpenAI Limits
- **Tokens per minute**: Varies by deployment tier
- **Concurrent requests**: Up to 120 requests per minute (standard)
- **Model capacity**: Configurable based on needs

### Cost Optimization
- Use Azure Cost Management to monitor spending
- Consider reserved instances for predictable workloads
- Implement request caching for frequently asked questions

## Security and Compliance

### Data Security
- **Azure Key Vault**: Secure credential storage
- **HTTPS**: All communications encrypted
- **RBAC**: Role-based access control
- **Private Endpoints**: Optional network isolation

### Compliance
- SOC 2 Type II (Azure services)
- ISO 27001 (Azure infrastructure)
- SEC compliance for financial data usage

## Troubleshooting

### Common Issues

**1. Azure AI Search Connection Failed**
```bash
# Check service name and key
az search service show --name your-service --resource-group your-rg
```

**2. OpenAI Model Not Found**
```bash
# List deployed models
az cognitiveservices account deployment list --name your-openai --resource-group your-rg
```

**3. Web App Not Starting**
- Check Azure App Service logs
- Verify environment variables are set
- Ensure Python runtime is 3.11+

### Monitoring and Logging

- **Application Insights**: Integrated monitoring
- **Azure Monitor**: Resource health and metrics
- **Log Analytics**: Centralized logging

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is for educational and research purposes. Please ensure compliance with:
- SEC terms of service for data usage
- Azure service terms and conditions
- Applicable financial data regulations

## Support and Resources

### Azure Resources
- [Azure AI Search Documentation](https://docs.microsoft.com/en-us/azure/search/)
- [Azure OpenAI Service Documentation](https://docs.microsoft.com/en-us/azure/cognitive-services/openai/)
- [Azure App Service Documentation](https://docs.microsoft.com/en-us/azure/app-service/)

### Financial Data
- [SEC EDGAR Database](https://www.sec.gov/edgar)
- [SEC API Documentation](https://www.sec.gov/developer)

### Contact
For questions and support, please open an issue in the GitHub repository.

---

**Built with Azure AI • Powered by GPT-4 • Financial Data from SEC EDGAR**
