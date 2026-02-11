"""
ماژول بارگذاری داده
این ماژول مسئول خواندن داده از فرمت‌های مختلف و اعتبارسنجی اولیه است
"""

import os
import pandas as pd
import json
from typing import Optional, Union, Dict, Any
from pathlib import Path
import logging

from src.utils import setup_logger, validate_file_extension, validate_file_size, load_config

class DataLoader:
    """
    کلاس بارگذاری داده از منابع مختلف
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        سازنده کلاس DataLoader
        
        پارامترها:
            config_path: مسیر فایل تنظیمات (اختیاری)
        """
        # بارگذاری تنظیمات با مدیریت خطا
        try:
            self.config = load_config(config_path) if config_path else load_config()
        except Exception as e:
            print(f"⚠️ خطا در بارگذاری تنظیمات: {e}. استفاده از تنظیمات پیش‌فرض...")
            from src.utils import get_default_config
            self.config = get_default_config()
        
        self.data_config = self.config.get('data', {})
        
        # اگر data_config خالی بود، مقدار پیش‌فرض بده
        if not self.data_config:
            self.data_config = {
                'max_file_size_mb': 200,
                'allowed_formats': ['csv', 'xlsx', 'xls', 'json'],
                'encoding': 'utf-8'
            }
        
        # تنظیم logger
        self.logger = setup_logger(
            'data_loader',
            log_file='outputs/logs/data_loader.log'
        )
        
        # تنظیمات پیش‌فرض
        self.max_file_size_mb = self.data_config.get('max_file_size_mb', 200)
        self.allowed_formats = self.data_config.get('allowed_formats', ['csv', 'xlsx', 'xls', 'json'])
        self.encoding = self.data_config.get('encoding', 'utf-8')
        
        self.logger.info(f"✅ DataLoader initialized with config: {self.data_config}")
    
    def load_csv(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        بارگذاری فایل CSV
        
        پارامترها:
            file_path: مسیر فایل CSV
            **kwargs: پارامترهای اضافی برای pandas read_csv
        
        بازگشت:
            df: دیتافریم پانداس
        """
        try:
            self.logger.info(f"📂 Loading CSV file: {file_path}")
            
            # اعتبارسنجی فایل
            self._validate_file(file_path, 'csv')
            
            # تنظیمات پیش‌فرض برای خواندن CSV
            csv_params = {
                'encoding': self.encoding,
                'low_memory': False
            }
            csv_params.update(kwargs)
            
            # خواندن فایل
            df = pd.read_csv(file_path, **csv_params)
            
            self.logger.info(f"✅ CSV loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Error loading CSV file: {e}")
            raise Exception(f"خطا در بارگذاری فایل CSV: {e}")
    
    def load_excel(self, file_path: str, sheet_name: Optional[Union[str, int]] = 0, **kwargs) -> pd.DataFrame:
        """
        بارگذاری فایل Excel
        
        پارامترها:
            file_path: مسیر فایل Excel
            sheet_name: نام یا شماره شیت
            **kwargs: پارامترهای اضافی برای pandas read_excel
        
        بازگشت:
            df: دیتافریم پانداس
        """
        try:
            self.logger.info(f"📂 Loading Excel file: {file_path}, sheet: {sheet_name}")
            
            # اعتبارسنجی فایل
            self._validate_file(file_path, ['xlsx', 'xls'])
            
            # خواندن فایل
            df = pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
            
            self.logger.info(f"✅ Excel loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Error loading Excel file: {e}")
            raise Exception(f"خطا در بارگذاری فایل Excel: {e}")
    
    def load_json(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        بارگذاری فایل JSON
        
        پارامترها:
            file_path: مسیر فایل JSON
            **kwargs: پارامترهای اضافی برای pandas read_json
        
        بازگشت:
            df: دیتافریم پانداس
        """
        try:
            self.logger.info(f"📂 Loading JSON file: {file_path}")
            
            # اعتبارسنجی فایل
            self._validate_file(file_path, 'json')
            
            # خواندن فایل
            df = pd.read_json(file_path, **kwargs)
            
            self.logger.info(f"✅ JSON loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Error loading JSON file: {e}")
            raise Exception(f"خطا در بارگذاری فایل JSON: {e}")
    
    def load_data(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        بارگذاری خودکار داده بر اساس پسوند فایل
        
        پارامترها:
            file_path: مسیر فایل
            **kwargs: پارامترهای اضافی
        
        بازگشت:
            df: دیتافریم پانداس
        """
        self.logger.info(f"📂 Auto-detecting file format: {file_path}")
        
        # تشخیص پسوند فایل
        file_extension = Path(file_path).suffix.lower().replace('.', '')
        
        # انتخاب روش بارگذاری مناسب
        if file_extension == 'csv':
            return self.load_csv(file_path, **kwargs)
        elif file_extension in ['xlsx', 'xls']:
            return self.load_excel(file_path, **kwargs)
        elif file_extension == 'json':
            return self.load_json(file_path, **kwargs)
        else:
            error_msg = f"فرمت فایل {file_extension} پشتیبانی نمی‌شود. فرمت‌های مجاز: {self.allowed_formats}"
            self.logger.error(f"❌ {error_msg}")
            raise ValueError(error_msg)
    
    def _validate_file(self, file_path: str, expected_format: Union[str, list]) -> None:
        """
        اعتبارسنجی فایل قبل از بارگذاری
        
        پارامترها:
            file_path: مسیر فایل
            expected_format: فرمت(های) مورد انتظار
        """
        # بررسی وجود فایل
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"فایل {file_path} پیدا نشد")
        
        # بررسی پسوند فایل
        if not validate_file_extension(file_path, 
                                      [expected_format] if isinstance(expected_format, str) else expected_format):
            raise ValueError(f"فرمت فایل نامعتبر است. فرمت مورد انتظار: {expected_format}")
        
        # بررسی حجم فایل
        if not validate_file_size(file_path, self.max_file_size_mb):
            raise ValueError(f"حجم فایل بیش از حد مجاز ({self.max_file_size_mb} مگابایت) است")
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """
        دریافت اطلاعات فایل
        
        پارامترها:
            file_path: مسیر فایل
        
        بازگشت:
            info: دیکشنری حاوی اطلاعات فایل
        """
        info = {
            'filename': os.path.basename(file_path),
            'extension': Path(file_path).suffix,
            'size_bytes': os.path.getsize(file_path),
            'size_mb': round(os.path.getsize(file_path) / (1024 * 1024), 2),
            'modified_time': pd.Timestamp.fromtimestamp(os.path.getmtime(file_path))
        }
        return info
    
    def save_processed_data(self, df: pd.DataFrame, file_path: str, format: str = 'csv') -> None:
        """
        ذخیره داده‌های پردازش شده
        
        پارامترها:
            df: دیتافریم
            file_path: مسیر ذخیره
            format: فرمت خروجی
        """
        try:
            self.logger.info(f"💾 Saving processed data to: {file_path}")
            
            # ایجاد پوشه خروجی اگر وجود ندارد
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            if format == 'csv':
                df.to_csv(file_path, index=False, encoding=self.encoding)
            elif format == 'excel':
                df.to_excel(file_path, index=False)
            elif format == 'json':
                df.to_json(file_path, orient='records', date_format='iso')
            else:
                raise ValueError(f"فرمت {format} پشتیبانی نمی‌شود")
            
            self.logger.info(f"✅ Data saved successfully to {file_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Error saving data: {e}")
            raise Exception(f"خطا در ذخیره داده: {e}")