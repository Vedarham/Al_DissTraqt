"""
DissTraqt Web Dashboard - Startup Script

Run this to start the web dashboard and optionally the detector.
"""

import subprocess
import sys
import webbrowser
import time
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed"""
    print("\n📋 Checking dependencies...")
    
    required = ['flask', 'flask_cors', 'cv2', 'ultralytics', 'numpy']
    missing = []
    
    for package in required:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("\nInstall with:")
        print("  pip install -r requirements_web.txt")
        print("  pip install -r requirements_enhanced.txt")
        return False
    
    print("\n✅ All dependencies found!")
    return True


def start_flask_server():
    """Start Flask development server"""
    print("\n🚀 Starting Flask server...")
    print("   Dashboard will be available at: http://localhost:5000")
    
    try:
        subprocess.Popen([
            sys.executable, 'app.py'
        ])
        print("✓ Flask server started")
        return True
    except Exception as e:
        print(f"✗ Error starting Flask: {e}")
        return False


def main():
    """Main startup routine"""
    print("\n" + "="*70)
    print("  DissTraqt Web Dashboard - Startup")
    print("="*70)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Cannot start without dependencies. Install them first.")
        sys.exit(1)
    
    # Start Flask server
    if not start_flask_server():
        print("❌ Could not start Flask server")
        sys.exit(1)
    
    # Wait for server to start
    print("\n⏳ Waiting for server to start...")
    time.sleep(3)
    
    # Try to open browser
    print("\n🌐 Opening dashboard in browser...")
    try:
        webbrowser.open('http://localhost:5000')
        print("✓ Browser opened")
    except Exception as e:
        print(f"⚠️  Could not open browser: {e}")
        print("   Manually visit: http://localhost:5000")
    
    print("\n" + "="*70)
    print("✅ Dashboard is running!")
    print("\nNext steps:")
    print("  1. Open browser to: http://localhost:5000")
    print("  2. Start detector: python detector_web.py")
    print("  3. Or test with: python -c \"import requests; requests.post('http://localhost:5000/api/test/add-violation', json={'person_id': 1, 'event_type': 'phone', 'duration': 10})\"")
    print("\nPress Ctrl+C to stop the server")
    print("="*70 + "\n")
    
    # Keep running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down...")
        print("Dashboard server stopped.")


if __name__ == '__main__':
    main()
