"""
ماژول توابع کمکی و ابزارهای عمومی
این ماژول شامل توابع کاربردی برای کل پروژه است
"""

import os
import json
import yaml
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

# -------------------- توابع مدیریت لاگ --------------------

def setup_logger(name: str, log_file: Optional[str] = None, level=logging.INFO) -> logging.Logger:
    """
    تنظیم و پیکربندی سیستم لاگ‌گیری
    
    پارامترها:
        name: نام logger
        log_file: مسیر فایل لاگ (اختیاری)
        level: سطح لاگ‌گیری
    
    بازگشت:
        logger: شیء logger پیکربندی شده
    """
    # ایجاد فرمت لاگ
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # ایجاد logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # اگر logger قبلاً تنظیم شده، از اضافه کردن دوباره هندلر جلوگیری می‌کنیم
    if not logger.handlers:
        # هندلر کنسول
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # هندلر فایل (اگر مسیر داده شده باشد)
        if log_file:
            # ایجاد پوشه لاگ اگر وجود ندارد
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
                
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    return logger

# -------------------- توابع بارگذاری تنظیمات --------------------

def load_config(config_path: str = 'config/config.yaml') -> Dict[str, Any]:
    """
    بارگذاری فایل تنظیمات YAML
    
    پارامترها:
        config_path: مسیر فایل تنظیمات
    
    بازگشت:
        config: دیکشنری حاوی تنظیمات
    """
    try:
        # بررسی وجود فایل
        if not os.path.exists(config_path):
            print(f"⚠️ فایل تنظیمات در {config_path} یافت نشد. استفاده از تنظیمات پیش‌فرض...")
            return get_default_config()
            
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
            
        # بررسی اینکه config خالی نباشد
        if config is None:
            print("⚠️ فایل تنظیمات خالی است. استفاده از تنظیمات پیش‌فرض...")
            return get_default_config()
            
        return config
        
    except FileNotFoundError:
        print(f"⚠️ فایل تنظیمات در {config_path} یافت نشد. استفاده از تنظیمات پیش‌فرض...")
        return get_default_config()
    except Exception as e:
        print(f"⚠️ خطا در بارگذاری فایل تنظیمات: {e}. استفاده از تنظیمات پیش‌فرض...")
        return get_default_config()
    
    
def get_default_config() -> Dict[str, Any]:
    """
    برگرداندن تنظیمات پیش‌فرض
    
    بازگشت:
        config: دیکشنری تنظیمات پیش‌فرض
    """
    return {
        'data': {
            'max_file_size_mb': 200,
            'allowed_formats': ['csv', 'xlsx', 'xls', 'json'],
            'encoding': 'utf-8'
        },
        'cleaning': {
            'missing_values': {
                'threshold_percent': 50,
                'numeric_strategy': 'mean',
                'categorical_strategy': 'mode'
            },
            'outliers': {
                'method': 'iqr',
                'iqr_multiplier': 1.5,
                'zscore_threshold': 3
            }
        },
        'visualization': {
            'figure_size': [12, 8],
            'dpi': 100,
            'color_palette': 'viridis',
            'heatmap_cmap': 'coolwarm'
        }
    }

# -------------------- توابع ذخیره‌سازی --------------------

def save_json_report(data: Dict, filepath: str) -> None:
    """
    ذخیره گزارش به فرمت JSON
    
    پارامترها:
        data: دیکشنری حاوی داده‌های گزارش
        filepath: مسیر فایل خروجی
    """
    # ایجاد پوشه خروجی اگر وجود نداشته باشد
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # تبدیل داده‌های numpy و pandas به نوع‌های قابل ذخیره
    def convert_to_serializable(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif pd.isna(obj):
            return None
        return obj
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, default=convert_to_serializable, ensure_ascii=False, indent=4)

def save_text_report(content: str, filepath: str) -> None:
    """
    ذخیره گزارش متنی
    
    پارامترها:
        content: محتوای متنی گزارش
        filepath: مسیر فایل خروجی
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

# -------------------- توابع تحلیل داده --------------------

def detect_data_type(series: pd.Series) -> str:
    """
    تشخیص نوع داده یک ستون
    
    پارامترها:
        series: سری پانداس
    
    بازگشت:
        data_type: 'numeric', 'categorical', 'datetime', یا 'text'
    """
    if pd.api.types.is_numeric_dtype(series):
        return 'numeric'
    elif pd.api.types.is_datetime64_any_dtype(series):
        return 'datetime'
    elif series.dtype == 'object':
        # اگر تعداد مقادیر یکتا کم باشد، categorical است
        if series.nunique() < 20:
            return 'categorical'
        else:
            return 'text'
    else:
        return 'unknown'

def get_memory_usage(df: pd.DataFrame) -> Dict[str, float]:
    """
    محاسبه میزان حافظه مصرفی دیتافریم
    
    پارامترها:
        df: دیتافریم پانداس
    
    بازگشت:
        memory_info: دیکشنری حاوی اطلاعات حافظه
    """
    memory_bytes = df.memory_usage(deep=True).sum()
    memory_mb = memory_bytes / (1024 * 1024)
    memory_kb = memory_bytes / 1024
    
    return {
        'bytes': int(memory_bytes),
        'kb': round(memory_kb, 2),
        'mb': round(memory_mb, 2)
    }

# -------------------- توابع زمان و تاریخ --------------------

def get_timestamp() -> str:
    """
    دریافت زمان فعلی به فرمت رشته
    
    بازگشت:
        timestamp: زمان فعلی فرمت شده
    """
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def format_datetime(dt: datetime) -> str:
    """
    فرمت کردن زمان به فارسی/انگلیسی
    
    پارامترها:
        dt: شیء datetime
    
    بازگشت:
        formatted_date: تاریخ فرمت شده
    """
    return dt.strftime('%Y-%m-%d %H:%M:%S')

# -------------------- توابع اعتبارسنجی --------------------

def validate_file_extension(filename: str, allowed_extensions: list) -> bool:
    """
    اعتبارسنجی پسوند فایل
    
    پارامترها:
        filename: نام فایل
        allowed_extensions: لیست پسوندهای مجاز
    
    بازگشت:
        is_valid: آیا پسوند مجاز است؟
    """
    ext = filename.split('.')[-1].lower() if '.' in filename else ''
    return ext in allowed_extensions

def validate_file_size(file_path: str, max_size_mb: int = 200) -> bool:
    """
    اعتبارسنجی حجم فایل
    
    پارامترها:
        file_path: مسیر فایل
        max_size_mb: حداکثر حجم مجاز به مگابایت
    
    بازگشت:
        is_valid: آیا حجم مجاز است؟
    """
    try:
        size_bytes = os.path.getsize(file_path)
        size_mb = size_bytes / (1024 * 1024)
        return size_mb <= max_size_mb
    except:
        return False

# -------------------- توابع کمکی برای Streamlit --------------------

def format_number(num: float, decimals: int = 2) -> str:
    """
    فرمت کردن اعداد با جدا کننده هزارگان
    
    پارامترها:
        num: عدد ورودی
        decimals: تعداد اعشار
    
    بازگشت:
        formatted: عدد فرمت شده
    """
    if isinstance(num, (int, float)):
        if num == int(num):
            return f"{int(num):,}"
        else:
            return f"{num:,.{decimals}f}"
    return str(num)

def create_download_link(content: str, filename: str, link_text: str) -> str:
    """
    ایجاد لینک دانلود برای HTML
    
    پارامترها:
        content: محتوای فایل
        filename: نام فایل
        link_text: متن لینک
    
    بازگشت:
        html_link: تگ HTML لینک دانلود
    """
    import base64
    b64 = base64.b64encode(content.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{filename}">{link_text}</a>'