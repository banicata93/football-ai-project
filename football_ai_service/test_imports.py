#!/usr/bin/env python3
"""
Тест скрипт за проверка на всички зависимости
"""

import sys
import traceback

def test_import(module_name, description=""):
    """Тества import на модул"""
    try:
        __import__(module_name)
        print(f"✅ {module_name} - {description}")
        return True
    except ImportError as e:
        print(f"❌ {module_name} - {description}: {e}")
        return False
    except Exception as e:
        print(f"⚠️  {module_name} - {description}: {e}")
        return False

def main():
    """Главна функция за тестване на imports"""
    print("🔍 Тестване на зависимости...\n")
    
    tests = [
        # Core ML
        ("pandas", "Data manipulation"),
        ("numpy", "Numerical computing"),
        ("sklearn", "Machine learning utilities"),
        ("xgboost", "Gradient boosting"),
        ("lightgbm", "Light gradient boosting"),
        ("joblib", "Model serialization"),
        
        # Statistical
        ("scipy", "Statistical functions"),
        ("scipy.stats", "Statistical distributions"),
        
        # API Framework
        ("fastapi", "REST API framework"),
        ("uvicorn", "ASGI server"),
        ("pydantic", "Data validation"),
        
        # Configuration & Utilities
        ("yaml", "YAML configuration"),
        ("tqdm", "Progress bars"),
        ("loguru", "Logging"),
        ("requests", "HTTP requests"),
        
        # Visualization
        ("matplotlib", "Plotting"),
        ("seaborn", "Statistical visualization"),
        
        # Standard library
        ("json", "JSON handling"),
        ("os", "Operating system interface"),
        ("pathlib", "Path handling"),
        ("datetime", "Date and time"),
        ("typing", "Type hints"),
    ]
    
    passed = 0
    total = len(tests)
    
    for module, desc in tests:
        if test_import(module, desc):
            passed += 1
    
    print(f"\n📊 Резултат: {passed}/{total} модула успешно заредени")
    
    if passed == total:
        print("🎉 Всички зависимости са налични!")
        return True
    else:
        print("⚠️  Някои зависимости липсват!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
