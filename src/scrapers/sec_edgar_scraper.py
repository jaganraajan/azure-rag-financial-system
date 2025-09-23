#!/usr/bin/env python3
"""
SEC EDGAR Web Scraper for Azure RAG Financial System

Downloads 10-K filings from SEC EDGAR database and optionally stores them in Azure Storage.
Supports local file storage and Azure Blob Storage for cloud deployment.

Features:
- Downloads 10-K filings for specified companies and years
- Rate-limited and SEC-compliant scraping
- Azure Storage integration for cloud storage
- Robust error handling and retry logic
- Detailed logging and progress tracking
"""

import os
import time
import requests
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import json
from azure.storage.blob import BlobServiceClient

# Azure Storage (optional)
try:
    from azure.storage.blob import BlobServiceClient
    from dotenv import load_dotenv
    AZURE_STORAGE_AVAILABLE = True
except ImportError:
    AZURE_STORAGE_AVAILABLE = False

logger = logging.getLogger(__name__)


class SECEdgarScraper:
    """
    SEC EDGAR scraper for downloading 10-K filings.
    
    Supports both local file storage and Azure Blob Storage.
    """
    
    # Company mapping with CIK codes
    COMPANIES = {
        'GOOGL': {
            'name': 'Alphabet Inc.',
            'cik': '1652044'
        },
        'MSFT': {
            'name': 'Microsoft Corporation',
            'cik': '789019'
        },
        'NVDA': {
            'name': 'NVIDIA Corporation',
            'cik': '1045810'
        }
    }
    
    def __init__(self, user_agent: str = "Azure Financial Analysis Tool 1.0", 
                 azure_storage_connection: Optional[str] = None,
                 azure_container_name: str = "financial-filings"):
        """
        Initialize the scraper.
        
        Args:
            user_agent: User agent string for SEC requests
            azure_storage_connection: Azure Storage connection string (optional)
            azure_container_name: Azure container name for blob storage
        """
        self.user_agent = user_agent
        # Use the class-level COMPANIES but allow for dynamic updates
        self.companies = self.COMPANIES.copy()
        
        # Request headers for SEC compliance
        self.headers = {
            'User-Agent': user_agent,
            'Accept-Encoding': 'gzip, deflate',
            'Host': 'data.sec.gov'
        }
        
        # Azure Storage setup (optional)
        self.azure_storage_connection = azure_storage_connection
        self.azure_container_name = azure_container_name
        self.blob_service_client = None
        
        if azure_storage_connection and AZURE_STORAGE_AVAILABLE:
            try:
                self.blob_service_client = BlobServiceClient.from_connection_string(
                    azure_storage_connection
                )
                self._ensure_container_exists()
                logger.info("Azure Storage client initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Azure Storage: {e}")
                self.blob_service_client = None
        elif azure_storage_connection and not AZURE_STORAGE_AVAILABLE:
            logger.warning("Azure Storage requested but azure-storage-blob not installed")
    
    def add_company(self, symbol: str, name: str, cik: str):
        """
        Add a new company to the scraper configuration.
        
        Args:
            symbol: Company stock symbol (e.g., 'AAPL')
            name: Company name (e.g., 'Apple Inc.')
            cik: SEC Central Index Key
        """
        self.companies[symbol.upper()] = {
            'name': name,
            'cik': cik
        }
        # Also update the class-level dictionary for persistence
        SECEdgarScraper.COMPANIES[symbol.upper()] = {
            'name': name,
            'cik': cik
        }
        logger.info(f"Added company: {symbol} - {name} (CIK: {cik})")
    
    def get_available_companies(self) -> Dict[str, Dict[str, str]]:
        """
        Get all available companies.
        
        Returns:
            Dictionary of company data
        """
        return self.companies.copy()
    
    def _ensure_container_exists(self):
        """Ensure the Azure container exists."""
        if self.blob_service_client:
            try:
                self.blob_service_client.create_container(
                    name=self.azure_container_name,
                    public_access=None
                )
                logger.info(f"Created Azure container: {self.azure_container_name}")
            except Exception as e:
                if "ContainerAlreadyExists" in str(e):
                    logger.debug(f"Azure container already exists: {self.azure_container_name}")
                else:
                    logger.error(f"Error creating container: {e}")
    
    def _make_request(self, url: str, retries: int = 3) -> Optional[requests.Response]:
        """
        Make a rate-limited request to SEC with retry logic.
        
        Args:
            url: The URL to request
            retries: Number of retry attempts
            
        Returns:
            Response object or None if failed
        """
        for attempt in range(retries):
            try:
                # SEC rate limiting: 10 requests per second max
                time.sleep(0.1)  # 100ms delay
                
                response = requests.get(url, headers=self.headers, timeout=30)
                response.raise_for_status()
                
                return response
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request attempt {attempt + 1} failed for {url}: {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"All retry attempts failed for {url}")
                    return None
    
    def get_company_filings(self, company_symbol: str) -> Optional[Dict]:
        """
        Get filing information for a company from SEC API.
        
        Args:
            company_symbol: Company symbol (e.g., 'GOOGL')
            
        Returns:
            Dictionary containing filing information or None
        """
        if company_symbol not in self.companies:
            logger.error(f"Unknown company symbol: {company_symbol}")
            return None
        
        cik = self.companies[company_symbol]['cik']
        url = f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json"
        
        logger.info(f"Fetching filing data for {company_symbol} (CIK: {cik})")
        
        response = self._make_request(url)
        if not response:
            return None
        
        try:
            data = response.json()
            logger.info(f"Retrieved filing data for {company_symbol}")
            return data
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response for {company_symbol}: {e}")
            return None
    
    def find_10k_filings(self, filings_data: Dict, years: List[int]) -> List[Dict]:
        """
        Find 10-K filings for specified years.
        
        Args:
            filings_data: Filing data from SEC API
            years: List of years to find filings for
            
        Returns:
            List of filing dictionaries
        """
        recent_filings = filings_data.get('filings', {}).get('recent', {})
        forms = recent_filings.get('form', [])
        filing_dates = recent_filings.get('filingDate', [])
        accession_numbers = recent_filings.get('accessionNumber', [])
        
        found_filings = []
        
        for i, form in enumerate(forms):
            if form == '10-K' and i < len(filing_dates) and i < len(accession_numbers):
                filing_date = filing_dates[i]
                filing_year = int(filing_date.split('-')[0])
                
                if filing_year in years:
                    found_filings.append({
                        'form': form,
                        'filing_date': filing_date,
                        'year': filing_year,
                        'accession_number': accession_numbers[i]
                    })
        
        # Sort by year
        found_filings.sort(key=lambda x: x['year'])
        
        logger.info(f"Found {len(found_filings)} 10-K filings for years {years}")
        return found_filings
    
    def download_filing(self, company_symbol: str, filing: Dict, output_dir: str) -> Optional[str]:
        """
        Download a specific 10-K filing.
        
        Args:
            company_symbol: Company symbol
            filing: Filing information dictionary
            output_dir: Output directory for local storage
            
        Returns:
            Local file path if successful, None otherwise
        """
        cik = self.companies[company_symbol]['cik']
        accession_number = filing['accession_number'].replace('-', '')
        year = filing['year']
        
        # Construct filing URL
        base_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_number}"
        
        # Try to find the main 10-K HTML file
        # Common patterns for 10-K filenames
        possible_filenames = [
            f"{accession_number}.htm",
            f"{accession_number}-10k.htm",
            f"d{accession_number}.htm",
            f"form10k.htm",
            f"10k.htm"
        ]
        
        filing_content = None
        successful_filename = None
        
        for filename in possible_filenames:
            url = f"{base_url}/{filename}"
            logger.debug(f"Trying to download: {url}")
            
            response = self._make_request(url)
            if response and response.status_code == 200:
                filing_content = response.text
                successful_filename = filename
                logger.info(f"Successfully downloaded: {url}")
                break
            else:
                logger.debug(f"Failed to download: {url}")
        
        if not filing_content:
            logger.error(f"Could not download 10-K filing for {company_symbol} {year}")
            return None
        
        # Create local filename
        local_filename = f"{company_symbol}_10K_{year}_{accession_number}.htm"
        local_file_path = os.path.join(output_dir, local_filename)
        
        # Save to local file
        try:
            os.makedirs(output_dir, exist_ok=True)
            with open(local_file_path, 'w', encoding='utf-8') as f:
                f.write(filing_content)
            
            file_size = os.path.getsize(local_file_path)
            logger.info(f"Saved {local_filename} ({file_size:,} bytes)")
            
            # Optionally upload to Azure Storage
            if self.blob_service_client:
                self._upload_to_azure_storage(local_filename, filing_content)
            
            return local_file_path
            
        except Exception as e:
            logger.error(f"Error saving file {local_filename}: {e}")
            return None
    
    def _upload_to_azure_storage(self, filename: str, content: str):
        """
        Upload file content to Azure Blob Storage.
        
        Args:
            filename: Blob name
            content: File content as string
        """
        if not self.blob_service_client:
            return
        
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.azure_container_name,
                blob=filename
            )
            
            blob_client.upload_blob(
                data=content.encode('utf-8'),
                overwrite=True,
                content_type='text/html'
            )
            
            logger.info(f"Uploaded to Azure Storage: {filename}")
            
        except Exception as e:
            logger.error(f"Error uploading to Azure Storage: {e}")
    
    def scrape_company_10k_filings(self, company_symbol: str, years: List[int], 
                                 output_dir: str) -> List[str]:
        """
        Scrape all 10-K filings for a company and specified years.
        
        Args:
            company_symbol: Company symbol (e.g., 'GOOGL')
            years: List of years to download
            output_dir: Output directory for files
            
        Returns:
            List of successfully downloaded file paths
        """
        logger.info(f"Starting 10-K download for {company_symbol}, years: {years}")
        
        # Get company filing data
        filings_data = self.get_company_filings(company_symbol)
        if not filings_data:
            logger.error(f"Could not retrieve filing data for {company_symbol}")
            return []
        
        # Find 10-K filings for specified years
        target_filings = self.find_10k_filings(filings_data, years)
        if not target_filings:
            logger.warning(f"No 10-K filings found for {company_symbol} in years {years}")
            return []
        
        # Download each filing
        downloaded_files = []
        for filing in target_filings:
            logger.info(f"Downloading {company_symbol} 10-K for {filing['year']}...")
            
            file_path = self.download_filing(company_symbol, filing, output_dir)
            if file_path:
                downloaded_files.append(file_path)
            
            # Delay between downloads
            time.sleep(1)
        
        logger.info(f"Completed download for {company_symbol}: {len(downloaded_files)} files")
        return downloaded_files
    
    def scrape_all_companies(self, companies: List[str], years: List[int], 
                           output_dir: str) -> Dict[str, List[str]]:
        """
        Scrape 10-K filings for multiple companies.
        
        Args:
            companies: List of company symbols
            years: List of years to download
            output_dir: Output directory
            
        Returns:
            Dictionary mapping company symbols to downloaded file paths
        """
        all_results = {}
        
        for company in companies:
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing {company}")
            logger.info(f"{'='*50}")
            
            files = self.scrape_company_10k_filings(company, years, output_dir)
            all_results[company] = files
            
            # Delay between companies
            if company != companies[-1]:  # Don't delay after the last company
                logger.info("Waiting 2 seconds before next company...")
                time.sleep(2)
        
        return all_results


def create_demo_filings(output_dir: str = "demo_filings"):
    """
    Create demo filings for testing when SEC access is not available.
    
    Args:
        output_dir: Directory to create demo files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    demo_content_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>{company} Form 10-K {year}</title>
    </head>
    <body>
        <h1>{company_name} - Annual Report (Form 10-K) - {year}</h1>
        
        <h2>Business Overview</h2>
        <p>{company_name} is a leading technology company that operates in various segments including cloud computing, 
        software development, and hardware manufacturing. Our revenue for {year} was driven by strong performance 
        across all business segments.</p>
        
        <h2>Risk Factors</h2>
        <p>Key risk factors include market competition, regulatory changes, cybersecurity threats, and economic 
        uncertainty. We continue to invest in risk mitigation strategies and compliance programs.</p>
        
        <h2>Financial Highlights {year}</h2>
        <p>Total revenue: ${revenue}M</p>
        <p>Operating income: ${operating_income}M</p>
        <p>Operating margin: {operating_margin}%</p>
        <p>Net income: ${net_income}M</p>
        
        <h2>Management Discussion and Analysis</h2>
        <p>Management believes the company is well-positioned for continued growth through innovation, 
        strategic partnerships, and market expansion. We expect continued investment in research and 
        development to drive future performance.</p>
        
        <h2>Future Outlook</h2>
        <p>Looking ahead, we anticipate continued growth in our core business areas. We remain committed 
        to delivering value to shareholders while investing in long-term strategic initiatives.</p>
    </body>
    </html>
    """
    
    # Demo data for each company
    demo_data = {
        'GOOGL': {
            'name': 'Alphabet Inc.',
            'years': {
                2022: {'revenue': 282836, 'operating_income': 74842, 'operating_margin': 26.5, 'net_income': 59972},
                2023: {'revenue': 307394, 'operating_income': 84267, 'operating_margin': 27.4, 'net_income': 73795},
                2024: {'revenue': 334000, 'operating_income': 89000, 'operating_margin': 26.7, 'net_income': 76000}
            }
        },
        'MSFT': {
            'name': 'Microsoft Corporation',
            'years': {
                2022: {'revenue': 198270, 'operating_income': 83383, 'operating_margin': 42.1, 'net_income': 72361},
                2023: {'revenue': 211915, 'operating_income': 89690, 'operating_margin': 42.3, 'net_income': 72361},
                2024: {'revenue': 230000, 'operating_income': 95000, 'operating_margin': 41.3, 'net_income': 78000}
            }
        },
        'NVDA': {
            'name': 'NVIDIA Corporation',
            'years': {
                2022: {'revenue': 26914, 'operating_income': 4368, 'operating_margin': 16.2, 'net_income': 4368},
                2023: {'revenue': 60922, 'operating_income': 19558, 'operating_margin': 32.1, 'net_income': 4368},
                2024: {'revenue': 79000, 'operating_income': 25000, 'operating_margin': 31.6, 'net_income': 20000}
            }
        }
    }
    
    created_files = []
    
    load_dotenv()
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    container_name = "filings"
    blob_service_client = None
    container_client = None
    if connection_string:
        print("Azure Storage connection string found, initializing client...")
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(container_name)
        try:
            print('Creating container if not exists...')
            container_client.create_container()
        except Exception:
            pass  # Container may already exist
    
    for company, company_info in demo_data.items():
        for year, financials in company_info['years'].items():
            content = demo_content_template.format(
                company=company,
                company_name=company_info['name'],
                year=year,
                **financials
            )
            filename = f"{company}_10K_{year}_demo.htm"
            file_path = os.path.join(output_dir, filename)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            created_files.append(file_path)
            logger.info(f"Created demo filing: {filename}")
            # Upload to Azure Blob Storage if configured
            if container_client:
                blob_name = os.path.basename(file_path)
                with open(file_path, "rb") as data:
                    container_client.upload_blob(name=blob_name, data=data, overwrite=True)
                print(f"Uploaded {filename} to Azure Blob Storage container '{container_name}'")
    
    print(f"Created {len(created_files)} demo filings in {output_dir}")
    return created_files


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create demo filings for testing
    demo_files = create_demo_filings()
    print(f"Created {len(demo_files)} demo files for testing")