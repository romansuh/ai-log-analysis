"""
Setup Script for Log Analysis Notebook
=====================================

This script prepares the environment and tests all dependencies
before running the main analysis notebook.
"""

import sys
import os
import subprocess

def check_imports():
    """Test all required imports"""
    print("🔍 Checking required libraries...")
    
    required_packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 
        'plotly', 'wordcloud', 'jupyter', 'ipykernel'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {missing_packages}")
        print("💡 Run: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All packages available!")
        return True

def setup_directories():
    """Create necessary directories"""
    print("\n📁 Setting up directories...")
    
    directories = ['data', 'results', 'notebooks']
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  ✅ {directory}/")
    
    print("✅ Directory structure ready!")

def test_preprocessing():
    """Test the preprocessing functions"""
    print("\n🧪 Testing preprocessing pipeline...")
    
    try:
        sys.path.append('src')
        from log_preprocessor import clean_log_message, preprocess_logs_dataframe
        import pandas as pd
        
        # Test single message
        test_log = "2024-08-11 12:22:59 INFO User logged in successfully"
        result = clean_log_message(test_log)
        assert 'cleaned' in result
        assert result['cleaned'] == 'user logged in successfully'
        print("  ✅ Single message preprocessing")
        
        # Test DataFrame processing
        df = pd.DataFrame({'log_message': [test_log]})
        processed = preprocess_logs_dataframe(df)
        assert 'cleaned_message' in processed.columns
        assert 'log_level' in processed.columns
        print("  ✅ DataFrame preprocessing")
        
        print("✅ Preprocessing pipeline working!")
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def test_visualizations():
    """Test basic visualization capabilities"""
    print("\n📊 Testing visualization libraries...")
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import plotly.graph_objects as go
        
        # Test matplotlib
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([1, 2, 3], [1, 4, 2])
        ax.set_title('Test Plot')
        plt.close(fig)
        print("  ✅ Matplotlib")
        
        # Test seaborn
        sns.set_style("whitegrid")
        print("  ✅ Seaborn")
        
        # Test plotly
        fig = go.Figure(data=go.Bar(x=['A', 'B'], y=[1, 2]))
        print("  ✅ Plotly")
        
        print("✅ Visualization libraries working!")
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def create_sample_data():
    """Create sample data for the notebook"""
    print("\n📋 Creating sample data...")
    
    sample_logs = [
        "2024-08-11 12:22:59 INFO User admin@company.com logged in from 192.168.1.100",
        "2024-08-11 12:23:01 DEBUG Fetching user profile from /var/lib/users/profiles.db",
        "2024-08-11 12:23:03 ERROR Database connection failed - timeout after 30s",
        "2024-08-11 12:23:05 WARN Invalid SSL certificate for https://api.service.com:8443",
        "2024-08-11 12:23:07 CRITICAL Memory usage exceeded 95% - OOM killer activated"
    ]
    
    # Save to logs.txt if it doesn't exist or is too small
    if not os.path.exists('logs.txt') or os.path.getsize('logs.txt') < 100:
        with open('logs.txt', 'w') as f:
            for log in sample_logs:
                f.write(log + '\n')
        print("  ✅ Sample logs.txt created")
    else:
        print("  ✅ logs.txt already exists")
    
    print("✅ Sample data ready!")

def main():
    """Main setup function"""
    print("🚀 Log Analysis Notebook Setup")
    print("=" * 50)
    
    # Run all checks
    checks = [
        check_imports(),
        test_preprocessing(),
        test_visualizations()
    ]
    
    # Setup environment
    setup_directories()
    create_sample_data()
    
    # Final status
    print("\n" + "=" * 50)
    if all(checks):
        print("🎉 SETUP COMPLETE!")
        print("✅ All systems ready!")
        print("\n🚀 Next steps:")
        print("1. Start Jupyter: jupyter notebook")
        print("2. Open: notebooks/log_analysis_and_preprocessing.ipynb")
        print("3. Run all cells to see your analysis!")
        return True
    else:
        print("❌ SETUP INCOMPLETE!")
        print("🔧 Please fix the issues above before running the notebook.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 