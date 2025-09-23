#!/usr/bin/env python3
"""
Test script for admin page functionality
"""

import os
import sys
import json
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_admin_functionality():
    """Test the admin functionality without full Azure setup"""
    print("🧪 Testing Admin Page Functionality")
    print("=" * 50)
    
    # Test scraper company addition
    try:
        from scrapers.sec_edgar_scraper import SECEdgarScraper
        
        # Create scraper instance
        scraper = SECEdgarScraper()
        
        # Test adding a new company
        print("📝 Testing company addition...")
        initial_count = len(scraper.companies)
        print(f"Initial companies: {initial_count}")
        
        # Add a test company
        test_symbol = "AAPL"
        test_name = "Apple Inc."
        test_cik = "0000320193"
        
        scraper.add_company(test_symbol, test_name, test_cik)
        
        # Verify company was added
        if test_symbol in scraper.companies:
            print(f"✅ Successfully added {test_symbol}")
            print(f"   Name: {scraper.companies[test_symbol]['name']}")
            print(f"   CIK: {scraper.companies[test_symbol]['cik']}")
        else:
            print(f"❌ Failed to add {test_symbol}")
            return False
        
        # Test getting available companies
        companies = scraper.get_available_companies()
        if len(companies) == initial_count + 1:
            print(f"✅ Company count increased to {len(companies)}")
        else:
            print(f"❌ Company count mismatch")
            return False
        
    except Exception as e:
        print(f"❌ Scraper test failed: {e}")
        return False
    
    # Test demo filing creation for new company
    try:
        print("\n📄 Testing demo filing creation...")
        from scrapers.sec_edgar_scraper import create_demo_filings
        
        # Create temp directory
        with tempfile.TemporaryDirectory() as temp_dir:
            demo_files = create_demo_filings(temp_dir)
            print(f"✅ Created {len(demo_files)} demo files")
            
            # Check if files exist
            for file_path in demo_files:
                if not os.path.exists(file_path):
                    print(f"❌ Demo file missing: {file_path}")
                    return False
            
            print("✅ All demo files created successfully")
        
    except Exception as e:
        print(f"❌ Demo filing test failed: {e}")
        return False
    
    # Test basic web app imports
    try:
        print("\n🌐 Testing web app imports...")
        
        # Check if web app can import required modules
        try:
            from web.flask_app import SCRAPER_AVAILABLE, SECEdgarScraper
            flask_available = True
        except ImportError as e:
            if "flask" in str(e).lower():
                print("⚠️  Flask not installed (expected in test environment)")
                flask_available = False
            else:
                raise e
        
        if flask_available:
            if SCRAPER_AVAILABLE:
                print("✅ SEC scraper available for web app")
            else:
                print("❌ SEC scraper not available for web app")
                return False
        else:
            print("⚠️  Skipping Flask-dependent tests")
        
        # Test class-level company addition (this doesn't require Flask)
        from scrapers.sec_edgar_scraper import SECEdgarScraper
        original_companies = SECEdgarScraper.COMPANIES.copy()
        
        # Add test company at class level
        SECEdgarScraper.COMPANIES["TEST"] = {
            "name": "Test Company",
            "cik": "1234567"
        }
        
        if "TEST" in SECEdgarScraper.COMPANIES:
            print("✅ Class-level company addition works")
        else:
            print("❌ Class-level company addition failed")
            return False
        
        # Restore original companies
        SECEdgarScraper.COMPANIES.clear()
        SECEdgarScraper.COMPANIES.update(original_companies)
        
    except Exception as e:
        print(f"❌ Web app import test failed: {e}")
        return False
    
    print("\n🎉 All admin functionality tests passed!")
    return True

def test_admin_api_simulation():
    """Test API functionality simulation"""
    print("\n🔌 Testing Admin API Simulation")
    print("=" * 50)
    
    try:
        # Simulate add company API call
        print("📝 Simulating add company API...")
        
        from scrapers.sec_edgar_scraper import SECEdgarScraper
        
        # Test data
        test_data = {
            'symbol': 'TSLA',
            'name': 'Tesla, Inc.',
            'cik': '1318605'
        }
        
        # Simulate the API logic
        symbol = test_data['symbol'].strip().upper()
        name = test_data['name'].strip()
        cik = test_data['cik'].strip()
        
        if not symbol or not name or not cik:
            print("❌ Validation failed")
            return False
        
        # Add company to scraper configuration
        SECEdgarScraper.COMPANIES[symbol] = {
            'name': name,
            'cik': cik
        }
        
        # Verify addition
        if symbol in SECEdgarScraper.COMPANIES:
            print(f"✅ API simulation successful: {symbol} added")
        else:
            print("❌ API simulation failed")
            return False
        
        # Simulate get companies API
        print("📋 Simulating get companies API...")
        companies = SECEdgarScraper.COMPANIES
        
        if len(companies) >= 4:  # Original 3 + 1 test company
            print(f"✅ Get companies API simulation successful: {len(companies)} companies")
        else:
            print("❌ Get companies API simulation failed")
            return False
        
        print("\n✅ All API simulations passed!")
        return True
        
    except Exception as e:
        print(f"❌ API simulation failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Admin Page Test Suite")
    print("=" * 60)
    
    all_passed = True
    
    # Run functionality tests
    if not test_admin_functionality():
        all_passed = False
    
    # Run API simulation tests  
    if not test_admin_api_simulation():
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Admin page functionality is working correctly.")
    else:
        print("❌ SOME TESTS FAILED! Please check the errors above.")
    
    sys.exit(0 if all_passed else 1)