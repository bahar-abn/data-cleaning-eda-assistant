"""
ماژول پاکسازی داده
این ماژول مسئول شناسایی و مدیریت مقادیر گمشده، داده‌های پرت و ناهنجاری‌ها است
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Tuple
from scipy import stats
import logging

from src.utils import setup_logger, detect_data_type, load_config

class DataCleaner:
    """
    کلاس پاکسازی و پیش‌پردازش داده
    """
    
    def __init__(self, df: pd.DataFrame, config_path: Optional[str] = None):
        """
        سازنده کلاس DataCleaner
        
        پارامترها:
            df: دیتافریم ورودی
            config_path: مسیر فایل تنظیمات
        """
        self.df = df.copy()
        self.original_shape = df.shape
        
        # بارگذاری تنظیمات
        self.config = load_config(config_path) if config_path else load_config()
        self.cleaning_config = self.config.get('cleaning', {})
        
        # تنظیم logger
        self.logger = setup_logger(
            'data_cleaner',
            log_file='outputs/logs/data_cleaner.log'
        )
        
        self.logger.info(f"✅ DataCleaner initialized with dataframe shape: {df.shape}")
        
        # ذخیره گزارش پاکسازی
        self.cleaning_report = {
            'initial_shape': self.original_shape,
            'final_shape': None,
            'missing_values': {},
            'outliers': {},
            'duplicates': 0,
            'operations': []
        }
    
    # -------------------- بخش مدیریت مقادیر گمشده --------------------
    
    def analyze_missing_values(self) -> Dict[str, Dict]:
        """
        تحلیل مقادیر گمشده در دیتافریم
        
        بازگشت:
            missing_report: گزارش کامل از مقادیر گمشده
        """
        self.logger.info("🔍 Analyzing missing values...")
        
        missing_report = {}
        
        for column in self.df.columns:
            missing_count = self.df[column].isnull().sum()
            missing_percentage = (missing_count / len(self.df)) * 100
            
            missing_report[column] = {
                'count': int(missing_count),
                'percentage': round(missing_percentage, 2),
                'dtype': str(self.df[column].dtype)
            }
        
        self.cleaning_report['missing_values'] = missing_report
        self.logger.info(f"✅ Missing values analysis completed")
        
        return missing_report
    
    def handle_missing_values(self, 
                            threshold: Optional[float] = None,
                            numeric_strategy: str = 'mean',
                            categorical_strategy: str = 'mode',
                            fill_values: Optional[Dict[str, Union[int, float, str]]] = None) -> pd.DataFrame:
        """
        مدیریت مقادیر گمشده
        
        پارامترها:
            threshold: آستانه درصد مقادیر گمشده برای حذف ستون
            numeric_strategy: روش پرکردن مقادیر عددی ('mean', 'median', 'mode', 'constant')
            categorical_strategy: روش پرکردن مقادیر دسته‌بندی ('mode', 'constant')
            fill_values: دیکشنری مقادیر جایگزین برای هر ستون
        
        بازگشت:
            df: دیتافریم پاکسازی شده
        """
        self.logger.info("🧹 Handling missing values...")
        
        # استفاده از تنظیمات پیش‌فرض اگر مقداری داده نشده
        if threshold is None:
            threshold = self.cleaning_config.get('missing_values', {}).get('threshold_percent', 50)
        
        # 1. حذف ستون‌هایی که بیش از آستانه مقادیر گمشده دارند
        cols_to_drop = []
        for column in self.df.columns:
            missing_percentage = (self.df[column].isnull().sum() / len(self.df)) * 100
            if missing_percentage > threshold:
                cols_to_drop.append(column)
        
        if cols_to_drop:
            self.df = self.df.drop(columns=cols_to_drop)
            self.cleaning_report['operations'].append({
                'operation': 'drop_columns_high_missing',
                'columns': cols_to_drop,
                'threshold': threshold
            })
            self.logger.info(f"✅ Dropped columns with >{threshold}% missing values: {cols_to_drop}")
        
        # 2. پرکردن مقادیر گمشده در ستون‌های باقیمانده
        for column in self.df.columns:
            if self.df[column].isnull().sum() > 0:
                # اگر مقدار جایگزین مشخص شده باشد
                if fill_values and column in fill_values:
                    fill_value = fill_values[column]
                    self.df[column] = self.df[column].fillna(fill_value)
                    method = 'constant'
                else:
                    # تشخیص نوع داده و انتخاب روش مناسب
                    data_type = detect_data_type(self.df[column])
                    
                    if data_type == 'numeric':
                        if numeric_strategy == 'mean':
                            fill_value = self.df[column].mean()
                        elif numeric_strategy == 'median':
                            fill_value = self.df[column].median()
                        elif numeric_strategy == 'mode':
                            fill_value = self.df[column].mode()[0] if not self.df[column].mode().empty else 0
                        else:
                            fill_value = 0
                        method = numeric_strategy
                        
                    else:  # categorical, text, datetime
                        if categorical_strategy == 'mode':
                            fill_value = self.df[column].mode()[0] if not self.df[column].mode().empty else 'Unknown'
                        else:
                            fill_value = 'Unknown'
                        method = categorical_strategy
                    
                    self.df[column] = self.df[column].fillna(fill_value)
                
                self.cleaning_report['operations'].append({
                    'operation': 'fill_missing_values',
                    'column': column,
                    'method': method,
                    'fill_value': str(fill_value) if isinstance(fill_value, (int, float)) else fill_value
                })
                
                self.logger.info(f"✅ Filled missing values in '{column}' using {method}")
        
        return self.df
    
    # -------------------- بخش مدیریت داده‌های پرت --------------------
    
    def detect_outliers_iqr(self, column: str, multiplier: float = 1.5) -> pd.Series:
        """
        تشخیص داده‌های پرت با روش IQR
        
        پارامترها:
            column: نام ستون
            multiplier: ضریب IQR
        
        بازگشت:
            outlier_mask: ماسک بولی برای داده‌های پرت
        """
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        outliers = (self.df[column] < lower_bound) | (self.df[column] > upper_bound)
        
        return outliers, lower_bound, upper_bound
    
    def detect_outliers_zscore(self, column: str, threshold: float = 3) -> pd.Series:
        """
        تشخیص داده‌های پرت با روش Z-Score
        
        پارامترها:
            column: نام ستون
            threshold: آستانه Z-Score
        
        بازگشت:
            outlier_mask: ماسک بولی برای داده‌های پرت
        """
        z_scores = np.abs(stats.zscore(self.df[column].dropna()))
        outliers = pd.Series(False, index=self.df.index)
        outliers[self.df[column].dropna().index] = z_scores > threshold
        
        return outliers
    
    def analyze_outliers(self, method: str = 'iqr') -> Dict[str, Dict]:
        """
        تحلیل داده‌های پرت در تمام ستون‌های عددی
        
        پارامترها:
            method: روش تشخیص ('iqr' یا 'zscore')
        
        بازگشت:
            outliers_report: گزارش داده‌های پرت
        """
        self.logger.info(f"🔍 Analyzing outliers using {method} method...")
        
        outliers_report = {}
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        
        config = self.cleaning_config.get('outliers', {})
        iqr_multiplier = config.get('iqr_multiplier', 1.5)
        zscore_threshold = config.get('zscore_threshold', 3)
        
        for column in numeric_columns:
            try:
                if method == 'iqr':
                    outliers, lower, upper = self.detect_outliers_iqr(column, iqr_multiplier)
                else:
                    outliers = self.detect_outliers_zscore(column, zscore_threshold)
                    lower = upper = None
                
                outlier_count = outliers.sum()
                outlier_percentage = (outlier_count / len(self.df)) * 100
                
                outliers_report[column] = {
                    'count': int(outlier_count),
                    'percentage': round(outlier_percentage, 2),
                    'method': method,
                    'bounds': {
                        'lower': round(lower, 2) if lower is not None else None,
                        'upper': round(upper, 2) if upper is not None else None
                    }
                }
                
            except Exception as e:
                self.logger.warning(f"⚠️ Could not detect outliers for {column}: {e}")
                outliers_report[column] = {'error': str(e)}
        
        self.cleaning_report['outliers'] = outliers_report
        self.logger.info(f"✅ Outliers analysis completed")
        
        return outliers_report
    
    def handle_outliers(self, 
                       method: str = 'cap', 
                       columns: Optional[List[str]] = None,
                       iqr_multiplier: Optional[float] = None) -> pd.DataFrame:
        """
        مدیریت داده‌های پرت
        
        پارامترها:
            method: روش مدیریت ('remove', 'cap', 'winsorize')
            columns: لیست ستون‌های هدف
            iqr_multiplier: ضریب IQR
        
        بازگشت:
            df: دیتافریم پاکسازی شده
        """
        self.logger.info(f"🧹 Handling outliers using {method} method...")
        
        if iqr_multiplier is None:
            iqr_multiplier = self.cleaning_config.get('outliers', {}).get('iqr_multiplier', 1.5)
        
        # اگر ستونی مشخص نشده، همه ستون‌های عددی را انتخاب کن
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        for column in columns:
            if column not in self.df.columns:
                continue
                
            if method == 'remove':
                # حذف سطرهای دارای داده پرت
                outliers, _, _ = self.detect_outliers_iqr(column, iqr_multiplier)
                self.df = self.df[~outliers]
                
                self.cleaning_report['operations'].append({
                    'operation': 'remove_outliers',
                    'column': column,
                    'removed_count': int(outliers.sum()),
                    'method': 'iqr'
                })
                
            elif method == 'cap':
                # محدود کردن داده‌های پرت به کران‌های بالا و پایین
                outliers, lower, upper = self.detect_outliers_iqr(column, iqr_multiplier)
                self.df.loc[self.df[column] < lower, column] = lower
                self.df.loc[self.df[column] > upper, column] = upper
                
                self.cleaning_report['operations'].append({
                    'operation': 'cap_outliers',
                    'column': column,
                    'lower_bound': round(lower, 2),
                    'upper_bound': round(upper, 2)
                })
            
            self.logger.info(f"✅ Handled outliers in '{column}' using {method}")
        
        return self.df
    
    # -------------------- بخش مدیریت داده‌های تکراری --------------------
    
    def handle_duplicates(self, keep: str = 'first') -> pd.DataFrame:
        """
        مدیریت رکوردهای تکراری
        
        پارامترها:
            keep: کدام رکورد نگه داشته شود ('first', 'last', False)
        
        بازگشت:
            df: دیتافریم پاکسازی شده
        """
        self.logger.info("🔍 Checking for duplicate rows...")
        
        duplicate_count = self.df.duplicated().sum()
        self.cleaning_report['duplicates'] = int(duplicate_count)
        
        if duplicate_count > 0:
            self.df = self.df.drop_duplicates(keep=keep)
            self.cleaning_report['operations'].append({
                'operation': 'remove_duplicates',
                'removed_count': duplicate_count,
                'keep': keep
            })
            self.logger.info(f"✅ Removed {duplicate_count} duplicate rows")
        else:
            self.logger.info("✅ No duplicate rows found")
        
        return self.df
    
    # -------------------- بخش تغییر نوع داده --------------------
    
    def optimize_dtypes(self) -> pd.DataFrame:
        """
        بهینه‌سازی نوع داده‌ها برای کاهش حافظه مصرفی
        
        بازگشت:
            df: دیتافریم با انواع بهینه شده
        """
        self.logger.info("🔄 Optimizing data types...")
        
        before_memory = self.df.memory_usage(deep=True).sum() / (1024 * 1024)
        
        for column in self.df.columns:
            col_type = self.df[column].dtype
            
            if col_type == 'object':
                # تبدیل ستون‌های متنی به categorical اگر تعداد مقادیر یکتا کم باشد
                num_unique = self.df[column].nunique()
                if num_unique / len(self.df) < 0.5:  # اگر کمتر از 50% مقادیر یکتا باشند
                    self.df[column] = self.df[column].astype('category')
                    self.cleaning_report['operations'].append({
                        'operation': 'optimize_dtype',
                        'column': column,
                        'from': 'object',
                        'to': 'category'
                    })
            
            elif 'int' in str(col_type):
                # کاهش سایز اعداد صحیح
                c_min = self.df[column].min()
                c_max = self.df[column].max()
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    self.df[column] = self.df[column].astype('int8')
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    self.df[column] = self.df[column].astype('int16')
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    self.df[column] = self.df[column].astype('int32')
                    
            elif 'float' in str(col_type):
                # کاهش سایز اعداد اعشاری
                self.df[column] = self.df[column].astype('float32')
        
        after_memory = self.df.memory_usage(deep=True).sum() / (1024 * 1024)
        memory_reduced = ((before_memory - after_memory) / before_memory) * 100
        
        self.logger.info(f"✅ Data types optimized: Memory usage reduced from {before_memory:.2f} MB to {after_memory:.2f} MB ({memory_reduced:.1f}% reduction)")
        
        return self.df
    
    # -------------------- بخش پاکسازی کامل --------------------
    
    def clean_all(self) -> Tuple[pd.DataFrame, Dict]:
        """
        اجرای تمام مراحل پاکسازی به صورت خودکار
        
        بازگشت:
            df: دیتافریم پاکسازی شده
            report: گزارش کامل پاکسازی
        """
        self.logger.info("🚀 Starting full data cleaning pipeline...")
        
        # 1. تحلیل مقادیر گمشده
        self.analyze_missing_values()
        
        # 2. مدیریت مقادیر گمشده
        self.handle_missing_values()
        
        # 3. مدیریت داده‌های تکراری
        self.handle_duplicates()
        
        # 4. تحلیل داده‌های پرت
        self.analyze_outliers()
        
        # 5. مدیریت داده‌های پرت
        self.handle_outliers(method='cap')
        
        # 6. بهینه‌سازی نوع داده‌ها
        self.optimize_dtypes()
        
        # به‌روزرسانی گزارش نهایی
        self.cleaning_report['final_shape'] = self.df.shape
        
        rows_removed = self.original_shape[0] - self.df.shape[0]
        cols_removed = self.original_shape[1] - self.df.shape[1]
        
        self.logger.info(f"✅ Data cleaning completed: {rows_removed} rows removed, {cols_removed} columns removed")
        
        return self.df, self.cleaning_report
    
    def get_cleaning_summary(self) -> str:
        """
        دریافت خلاصه عملیات پاکسازی به صورت متنی
        
        بازگشت:
            summary: خلاصه گزارش
        """
        summary = []
        summary.append("="*50)
        summary.append("📊 گزارش خلاصه پاکسازی داده")
        summary.append("="*50)
        summary.append(f"📏 ابعاد اولیه: {self.cleaning_report['initial_shape'][0]} سطر و {self.cleaning_report['initial_shape'][1]} ستون")
        summary.append(f"📏 ابعاد نهایی: {self.cleaning_report['final_shape'][0]} سطر و {self.cleaning_report['final_shape'][1]} ستون")
        
        if self.cleaning_report['duplicates'] > 0:
            summary.append(f"🗑️ رکوردهای تکراری حذف شده: {self.cleaning_report['duplicates']}")
        
        missing_cols = [col for col, info in self.cleaning_report['missing_values'].items() if info['percentage'] > 0]
        if missing_cols:
            summary.append(f"⚠️ ستون‌های دارای مقادیر گمشده: {len(missing_cols)} ستون")
        
        outlier_cols = [col for col, info in self.cleaning_report['outliers'].items() 
                       if isinstance(info, dict) and info.get('count', 0) > 0]
        if outlier_cols:
            summary.append(f"📈 ستون‌های دارای داده پرت: {len(outlier_cols)} ستون")
        
        summary.append("="*50)
        
        return "\n".join(summary)