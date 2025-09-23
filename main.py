#!/usr/bin/env python3
"""
Azure RAG Financial System

This script provides a comprehensive financial analysis system using Azure AI Search
for vector embeddings and Azure OpenAI for enhanced query processing.

Supports two main modes:
1. Scraper mode: Downloads 10-K filings from SEC EDGAR
2. RAG mode: Processes filings and provides intelligent query interface
"""

import os
import sys
import argparse
import logging
import json
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from src.rag.azure_rag_pipeline import AzureRAGPipeline
    AZURE_RAG_AVAILABLE = True
except ImportError as e:
    print(f"Azure RAG pipeline not available: {e}")
    AZURE_RAG_AVAILABLE = False

try:
    from src.scrapers.sec_edgar_scraper import SECEdgarScraper
    SCRAPER_AVAILABLE = True
except ImportError as e:
    print(f"SEC scraper not available: {e}")
    SCRAPER_AVAILABLE = False


def setup_logging(level=logging.INFO):
    """Setup logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def run_scraper_mode(args):
    """Run the SEC filing scraper."""
    if not SCRAPER_AVAILABLE:
        print("❌ SEC scraper not available. Please check your installation.")
        sys.exit(1)
    
    print("Azure RAG Financial System - SEC EDGAR Scraper")
    print("=" * 50)
    print(f"Companies: {', '.join(args.companies)}")
    print(f"Years: {', '.join(map(str, args.years))}")
    print(f"Output directory: {args.output_dir}")
    print(f"Expected total files: {len(args.companies) * len(args.years)}")
    print("=" * 50)
    print()
    
    # Create scraper instance
    scraper = SECEdgarScraper(
        user_agent=args.user_agent,
        azure_storage_connection=getattr(args, 'azure_storage_connection', None)
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Track results
    all_results = {}
    total_downloaded = 0
    
    try:
        # Download filings for each company
        for company in args.companies:
            print(f"\nProcessing {company}...")
            files = scraper.scrape_company_10k_filings(company, args.years, args.output_dir)
            all_results[company] = files
            total_downloaded += len(files)
            
        # Print summary
        print("\n" + "=" * 60)
        print("DOWNLOAD SUMMARY")
        print("=" * 60)
        
        for company, files in all_results.items():
            print(f"\n{company}:")
            print(f"  Files downloaded: {len(files)}")
            
            for file_path in files:
                filename = os.path.basename(file_path)
                file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
                print(f"    - {filename} ({file_size:,} bytes)")
        
        print(f"\nTotal files downloaded: {total_downloaded}")
        print(f"Expected files: {len(args.companies) * len(args.years)}")
        
        if total_downloaded == len(args.companies) * len(args.years):
            print("✅ All expected files downloaded successfully!")
        else:
            print("⚠️  Some files may be missing. Check the logs above for details.")
            
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during download: {e}")
        sys.exit(1)


def run_rag_mode(args):
    """Run the Azure RAG pipeline."""
    if not AZURE_RAG_AVAILABLE:
        print("❌ Azure RAG pipeline not available. Please check your Azure configuration.")
        sys.exit(1)
    
    print("Azure RAG Financial System")
    print("=" * 40)
    print(f"Input directory: {args.input_dir}")
    print(f"Azure AI Search service: {args.search_service}")
    print(f"Azure OpenAI endpoint: {args.openai_endpoint}")
    print("=" * 40)
    print()
    
    # Initialize Azure RAG pipeline
    try:
        rag = AzureRAGPipeline(
            search_service_name=args.search_service,
            search_index_name=args.search_index,
            openai_endpoint=args.openai_endpoint,
            openai_deployment=args.openai_deployment,
            embedding_deployment=args.embedding_deployment
        )
        print("🚀 Azure RAG Pipeline initialized successfully")
        
        if args.process:
            # Process documents
            print("Processing documents...")
            results = rag.process_directory(args.input_dir)
            
            print("\n" + "=" * 50)
            print("PROCESSING SUMMARY")
            print("=" * 50)
            print(f"Files processed: {results['processed_files']}/{results['total_files']}")
            print(f"Total chunks created: {results['total_chunks']}")
            print(f"Azure Search documents: {results['search_documents']}")
            print("✅ Documents processed successfully!")
            print()
        
        # Get system stats
        stats = rag.get_stats()
        print(f"Azure AI Search index contains: {stats['total_documents']} documents")
        print()
        
        if args.query:
            # Single query mode
            print(f"Query: {args.query}")
            print("=" * 60)
            
            result = rag.query(args.query, top_k=args.top_k, return_json=True)
            
            print("\n📋 Azure RAG Query Result:")
            print("=" * 40)
            print(json.dumps(result, indent=2))
                
        elif not args.process:
            # Interactive query mode
            print("🤖 Interactive Query Mode (Azure-powered)")
            print("Type your questions about the financial documents.")
            print("Type 'quit' or 'exit' to stop.")
            print()
            
            while True:
                try:
                    query = input("💬 Your question: ").strip()
                    
                    if query.lower() in ['quit', 'exit', 'q']:
                        print("👋 Goodbye!")
                        break
                    
                    if not query:
                        continue
                    
                    print(f"\n🔍 Searching with Azure AI: {query}")
                    print("-" * 50)
                    
                    result = rag.query(query, top_k=args.top_k, return_json=True)
                    
                    if 'error' in result:
                        print(f"❌ Error: {result['error']}")
                        continue
                    
                    print(f"📊 Answer: {result.get('answer', 'No answer generated')}")
                    print(f"🎯 Confidence: {result.get('confidence', 0.0):.2f}")
                    print(f"📚 Sources: {len(result.get('sources', []))}")
                    
                    print("-" * 50)
                    print()
                    
                except KeyboardInterrupt:
                    print("\n\n👋 Goodbye!")
                    break
                except Exception as e:
                    print(f"❌ Error: {e}")
                    print()
            
    except Exception as e:
        print(f"❌ Error initializing Azure RAG system: {e}")
        print("\nPlease check your Azure configuration:")
        print("- Azure AI Search service name")
        print("- Azure OpenAI endpoint")
        print("- Azure credentials")
        sys.exit(1)


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Azure RAG Financial System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download filings (scraper mode)
  python main.py scrape --companies GOOGL MSFT --years 2023 2024
  
  # Process downloaded filings for Azure RAG
  python main.py rag --process --input-dir filings
  
  # Query the Azure RAG system
  python main.py rag --query "What are the main revenue sources?"
  
  # Interactive query mode
  python main.py rag
        """
    )
    
    # Add subcommands
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')
    
    # Scraper mode
    scraper_parser = subparsers.add_parser('scrape', help='Download SEC filings')
    scraper_parser.add_argument('--companies', nargs='+', default=['GOOGL', 'MSFT', 'NVDA'],
                               choices=['GOOGL', 'MSFT', 'NVDA'],
                               help='Company symbols to download (default: all)')
    scraper_parser.add_argument('--years', nargs='+', type=int, default=[2022, 2023, 2024],
                               help='Years to download filings for (default: 2022 2023 2024)')
    scraper_parser.add_argument('--output-dir', default='filings',
                               help='Output directory for downloaded filings (default: filings)')
    scraper_parser.add_argument('--user-agent', 
                               default='Azure Financial Analysis Tool 1.0',
                               help='User agent for SEC requests')
    scraper_parser.add_argument('--azure-storage-connection',
                               help='Azure Storage connection string for cloud storage')
    
    # RAG mode
    rag_parser = subparsers.add_parser('rag', help='Azure RAG pipeline operations')
    rag_parser.add_argument('--input-dir', default='demo_filings',
                           help='Directory containing HTML filings (default: demo_filings)')
    rag_parser.add_argument('--search-service', 
                           default=os.getenv('AZURE_SEARCH_SERVICE_NAME'),
                           help='Azure AI Search service name')
    rag_parser.add_argument('--search-index', default='financial-documents',
                           help='Azure AI Search index name (default: financial-documents)')
    rag_parser.add_argument('--openai-endpoint',
                           default=os.getenv('AZURE_OPENAI_ENDPOINT'),
                           help='Azure OpenAI endpoint')
    rag_parser.add_argument('--openai-deployment', default='gpt-4',
                           help='Azure OpenAI deployment name (default: gpt-4)')
    rag_parser.add_argument('--embedding-deployment', default='text-embedding-ada-002',
                           help='Azure OpenAI embedding deployment name')
    rag_parser.add_argument('--process', action='store_true',
                           help='Process documents and build Azure search index')
    rag_parser.add_argument('--query', type=str,
                           help='Single query to execute')
    rag_parser.add_argument('--top-k', type=int, default=5,
                           help='Number of top results to return (default: 5)')
    
    # Global options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # If no mode specified, show help
    if not args.mode:
        parser.print_help()
        sys.exit(1)
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)
    
    # Route to appropriate mode
    if args.mode == 'scrape':
        run_scraper_mode(args)
    elif args.mode == 'rag':
        run_rag_mode(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()