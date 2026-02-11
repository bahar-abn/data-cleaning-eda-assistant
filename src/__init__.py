"""
دستیار هوشمند پاکسازی داده و EDA
این پکیج شامل ابزارهای کامل برای تحلیل و پاکسازی خودکار داده است
"""

__version__ = "1.0.0"
__author__ = "Your Name"

# ایمپورت کلاس‌های اصلی برای دسترسی آسان
from src.data_loader import DataLoader
from src.data_cleaner import DataCleaner
from src.eda_analyzer import EDAAnalyzer
from src.visualizer import Visualizer
from src.report_generator import ReportGenerator

__all__ = [
    'DataLoader',
    'DataCleaner',
    'EDAAnalyzer',
    'Visualizer',
    'ReportGenerator'
]