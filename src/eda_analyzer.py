"""
ماژول تحلیل اکتشافی داده (EDA)
این ماژول مسئول محاسبه آمار توصیفی، ماتریس همبستگی و تحلیل توزیع داده‌ها است
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Any
from scipy import stats
from scipy.stats import normaltest, shapiro, kstest
import warnings

from src.utils import setup_logger, detect_data_type, get_memory_usage

class EDAAnalyzer:
    """
    کلاس تحلیل اکتشافی داده
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        سازنده کلاس EDAAnalyzer
        
        پارامترها:
            df: دیتافریم ورودی
        """
        self.df = df.copy()
        self.logger = setup_logger(
            'eda_analyzer',
            log_file='outputs/logs/eda_analyzer.log'
        )
        
        self.logger.info(f"✅ EDAAnalyzer initialized with dataframe shape: {df.shape}")
        
        # ذخیره نتایج تحلیل
        self.analysis_results = {
            'basic_info': {},
            'descriptive_stats': {},
            'missing_values': {},
            'correlation': {},
            'distribution': {},
            'categorical_analysis': {},
            'memory_usage': {}
        }
    
    # -------------------- بخش اطلاعات پایه --------------------
    
    def get_basic_info(self) -> Dict[str, Any]:
        """
        دریافت اطلاعات پایه دیتافریم
        
        بازگشت:
            basic_info: دیکشنری حاوی اطلاعات پایه
        """
        self.logger.info("📋 Getting basic dataframe information...")
        
        # اطلاعات کلی
        info = {
            'shape': {
                'rows': self.df.shape[0],
                'columns': self.df.shape[1]
            },
            'columns': self.df.columns.tolist(),
            'dtypes': {col: str(dtype) for col, dtype in self.df.dtypes.items()},
            'memory_usage': get_memory_usage(self.df),
            'index_info': {
                'start': str(self.df.index[0]) if len(self.df) > 0 else None,
                'end': str(self.df.index[-1]) if len(self.df) > 0 else None,
                'length': len(self.df.index)
            }
        }
        
        self.analysis_results['basic_info'] = info
        self.logger.info("✅ Basic information collected")
        
        return info
    
    def get_descriptive_stats(self) -> pd.DataFrame:
        """
        محاسبه آمار توصیفی برای ستون‌های عددی
        
        بازگشت:
            stats_df: دیتافریم حاوی آمار توصیفی
        """
        self.logger.info("📊 Calculating descriptive statistics...")
        
        # ستون‌های عددی
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            self.logger.warning("⚠️ No numeric columns found for descriptive statistics")
            return pd.DataFrame()
        
        # آمار توصیفی پایه
        desc_stats = self.df[numeric_cols].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).T
        
        # اضافه کردن آمار اضافی
        for col in numeric_cols:
            desc_stats.loc[col, 'variance'] = self.df[col].var()
            desc_stats.loc[col, 'skewness'] = self.df[col].skew()
            desc_stats.loc[col, 'kurtosis'] = self.df[col].kurtosis()
            desc_stats.loc[col, 'missing_count'] = self.df[col].isnull().sum()
            desc_stats.loc[col, 'missing_percentage'] = (self.df[col].isnull().sum() / len(self.df)) * 100
            desc_stats.loc[col, 'unique_values'] = self.df[col].nunique()
        
        self.analysis_results['descriptive_stats'] = desc_stats.to_dict()
        self.logger.info(f"✅ Descriptive statistics calculated for {len(numeric_cols)} numeric columns")
        
        return desc_stats
    
    # -------------------- بخش تحلیل همبستگی --------------------
    
    def calculate_correlation(self, method: str = 'pearson', threshold: float = 0.7) -> Dict[str, Any]:
        """
        محاسبه ماتریس همبستگی
        
        پارامترها:
            method: روش محاسبه همبستگی ('pearson', 'spearman', 'kendall')
            threshold: آستانه همبستگی قوی
        
        بازگشت:
            correlation_info: دیکشنری حاوی اطلاعات همبستگی
        """
        self.logger.info(f"🔄 Calculating {method} correlation matrix...")
        
        # انتخاب ستون‌های عددی
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            self.logger.warning("⚠️ Need at least 2 numeric columns for correlation analysis")
            return {'matrix': None, 'high_correlations': []}
        
        # محاسبه ماتریس همبستگی
        corr_matrix = self.df[numeric_cols].corr(method=method)
        
        # یافتن همبستگی‌های قوی
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > threshold:
                    high_corr.append({
                        'variable1': corr_matrix.columns[i],
                        'variable2': corr_matrix.columns[j],
                        'correlation': round(corr_value, 3),
                        'strength': 'strong' if abs(corr_value) > 0.7 else 'moderate',
                        'direction': 'positive' if corr_value > 0 else 'negative'
                    })
        
        correlation_info = {
            'matrix': corr_matrix.to_dict(),
            'high_correlations': high_corr,
            'method': method,
            'threshold': threshold
        }
        
        self.analysis_results['correlation'] = correlation_info
        self.logger.info(f"✅ Correlation analysis completed. Found {len(high_corr)} high correlations")
        
        return correlation_info
    
    # -------------------- بخش تحلیل توزیع --------------------
    
    def test_normality(self, column: str) -> Dict[str, Any]:
        """
        آزمون نرمال بودن توزیع داده‌ها
        
        پارامترها:
            column: نام ستون
        
        بازگشت:
            normality_test: نتایج آزمون نرمال بودن
        """
        data = self.df[column].dropna()
        
        if len(data) < 3:
            return {'is_normal': False, 'error': 'Insufficient data'}
        
        results = {}
        
        try:
            # Shapiro-Wilk test (برای داده‌های کمتر از 5000)
            if len(data) < 5000:
                shapiro_stat, shapiro_p = shapiro(data)
                results['shapiro'] = {
                    'statistic': round(shapiro_stat, 4),
                    'p_value': round(shapiro_p, 4),
                    'is_normal': shapiro_p > 0.05
                }
            
            # D'Agostino's K^2 test
            dagostino_stat, dagostino_p = normaltest(data)
            results['dagostino'] = {
                'statistic': round(dagostino_stat, 4),
                'p_value': round(dagostino_p, 4),
                'is_normal': dagostino_p > 0.05
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Normality test failed for {column}: {e}")
            results['error'] = str(e)
        
        return results
    
    def analyze_distribution(self) -> Dict[str, Dict]:
        """
        تحلیل توزیع تمام ستون‌ها
        
        بازگشت:
            distribution_info: اطلاعات توزیع برای هر ستون
        """
        self.logger.info("📈 Analyzing data distributions...")
        
        distribution_info = {}
        
        for column in self.df.columns:
            col_info = {}
            data_type = detect_data_type(self.df[column])
            col_info['data_type'] = data_type
            
            # آمار پایه
            col_info['unique_count'] = int(self.df[column].nunique())
            col_info['unique_percentage'] = round((col_info['unique_count'] / len(self.df)) * 100, 2)
            col_info['missing_count'] = int(self.df[column].isnull().sum())
            col_info['missing_percentage'] = round((col_info['missing_count'] / len(self.df)) * 100, 2)
            
            # آمار مخصوص هر نوع داده
            if data_type == 'numeric':
                col_info.update({
                    'min': float(self.df[column].min()) if not pd.isna(self.df[column].min()) else None,
                    'max': float(self.df[column].max()) if not pd.isna(self.df[column].max()) else None,
                    'mean': float(self.df[column].mean()) if not pd.isna(self.df[column].mean()) else None,
                    'median': float(self.df[column].median()) if not pd.isna(self.df[column].median()) else None,
                    'std': float(self.df[column].std()) if not pd.isna(self.df[column].std()) else None,
                    'skewness': float(self.df[column].skew()) if not pd.isna(self.df[column].skew()) else None,
                    'kurtosis': float(self.df[column].kurtosis()) if not pd.isna(self.df[column].kurtosis()) else None,
                    'q1': float(self.df[column].quantile(0.25)),
                    'q3': float(self.df[column].quantile(0.75)),
                    'iqr': float(self.df[column].quantile(0.75) - self.df[column].quantile(0.25))
                })
                
                # آزمون نرمال بودن
                normality_results = self.test_normality(column)
                col_info['normality_tests'] = normality_results
                
            elif data_type == 'categorical':
                # فراوانی مقادیر
                value_counts = self.df[column].value_counts().head(10).to_dict()
                # تبدیل مقادیر به رشته
                col_info['top_values'] = {str(k): int(v) for k, v in value_counts.items()}
                col_info['mode'] = str(self.df[column].mode()[0]) if not self.df[column].mode().empty else None
                
            elif data_type == 'datetime':
                col_info.update({
                    'min_date': str(self.df[column].min()) if not pd.isna(self.df[column].min()) else None,
                    'max_date': str(self.df[column].max()) if not pd.isna(self.df[column].max()) else None,
                    'range_days': (self.df[column].max() - self.df[column].min()).days if not pd.isna(self.df[column].min()) else None
                })
            
            distribution_info[column] = col_info
        
        self.analysis_results['distribution'] = distribution_info
        self.logger.info(f"✅ Distribution analysis completed for {len(self.df.columns)} columns")
        
        return distribution_info
    
    # -------------------- بخش تحلیل ستون‌های دسته‌بندی --------------------
    
    def analyze_categorical(self) -> Dict[str, Dict]:
        """
        تحلیل عمیق ستون‌های دسته‌بندی
        
        بازگشت:
            categorical_info: اطلاعات تحلیل ستون‌های دسته‌بندی
        """
        self.logger.info("🏷️ Analyzing categorical columns...")
        
        categorical_info = {}
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        
        for column in categorical_cols:
            col_info = {}
            
            # آمار کلی
            col_info['total_values'] = len(self.df[column])
            col_info['unique_values'] = self.df[column].nunique()
            col_info['missing_values'] = int(self.df[column].isnull().sum())
            col_info['missing_percentage'] = round((col_info['missing_values'] / len(self.df)) * 100, 2)
            
            # پرتکرارترین مقادیر
            top_values = self.df[column].value_counts().head(10)
            col_info['top_values'] = {str(k): int(v) for k, v in top_values.items()}
            col_info['top_value_frequency'] = float(top_values.iloc[0] / len(self.df) * 100) if len(top_values) > 0 else 0
            
            # کم‌تکرارترین مقادیر
            rare_threshold = len(self.df) * 0.01  # 1%
            rare_values = self.df[column].value_counts()
            rare_values = rare_values[rare_values < rare_threshold]
            col_info['rare_values_count'] = len(rare_values)
            col_info['rare_values_percentage'] = (len(rare_values) / col_info['unique_values'] * 100) if col_info['unique_values'] > 0 else 0
            
            # آنتروپی و تنوع
            probabilities = self.df[column].value_counts(normalize=True)
            entropy = -sum(p * np.log2(p) for p in probabilities)
            max_entropy = np.log2(col_info['unique_values']) if col_info['unique_values'] > 0 else 0
            col_info['entropy'] = round(entropy, 2)
            col_info['normalized_entropy'] = round(entropy / max_entropy, 2) if max_entropy > 0 else 0
            
            categorical_info[column] = col_info
        
        self.analysis_results['categorical_analysis'] = categorical_info
        self.logger.info(f"✅ Categorical analysis completed for {len(categorical_cols)} columns")
        
        return categorical_info
    
    # -------------------- بخش گزارش کامل EDA --------------------
    
    def generate_full_report(self) -> Dict[str, Any]:
        """
        تولید گزارش کامل EDA
        
        بازگشت:
            full_report: دیکشنری حاوی تمام تحلیل‌ها
        """
        self.logger.info("🚀 Generating complete EDA report...")
        
        # اجرای تمام تحلیل‌ها
        self.get_basic_info()
        self.get_descriptive_stats()
        self.calculate_correlation()
        self.analyze_distribution()
        self.analyze_categorical()
        
        # خلاصه کلی
        summary = {
            'dataset_name': 'Dataset Analysis',
            'total_rows': self.df.shape[0],
            'total_columns': self.df.shape[1],
            'numeric_columns': len(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(self.df.select_dtypes(include=['object', 'category']).columns),
            'datetime_columns': len(self.df.select_dtypes(include=['datetime64']).columns),
            'total_missing_values': int(self.df.isnull().sum().sum()),
            'total_duplicates': int(self.df.duplicated().sum()),
            'memory_usage_mb': get_memory_usage(self.df)['mb']
        }
        
        self.analysis_results['summary'] = summary
        
        self.logger.info("✅ EDA report generated successfully")
        
        return self.analysis_results
    
    def get_insights(self) -> List[str]:
        """
        استخراج insights از تحلیل‌ها
        
        بازگشت:
            insights: لیستی از insights کشف شده
        """
        insights = []
        
        # 1. Insight درباره ابعاد داده
        insights.append(f"📏 این مجموعه داده شامل {self.df.shape[0]:,} رکورد و {self.df.shape[1]} ویژگی است.")
        
        # 2. Insight درباره مقادیر گمشده
        missing_total = self.df.isnull().sum().sum()
        if missing_total > 0:
            missing_percent = (missing_total / (self.df.shape[0] * self.df.shape[1])) * 100
            insights.append(f"⚠️ {missing_total:,} مقدار گمشده در داده‌ها وجود دارد ({missing_percent:.1f}% از کل داده‌ها).")
        else:
            insights.append("✅ هیچ مقدار گمشده‌ای در داده‌ها وجود ندارد.")
        
        # 3. Insight درباره داده‌های تکراری
        duplicate_count = self.df.duplicated().sum()
        if duplicate_count > 0:
            duplicate_percent = (duplicate_count / self.df.shape[0]) * 100
            insights.append(f"🔄 {duplicate_count:,} رکورد تکراری در داده‌ها وجود دارد ({duplicate_percent:.1f}%).")
        
        # 4. Insight درباره همبستگی‌ها
        if 'correlation' in self.analysis_results and self.analysis_results['correlation']:
            high_corr = self.analysis_results['correlation'].get('high_correlations', [])
            if high_corr:
                strong_corr = [c for c in high_corr if c['strength'] == 'strong']
                insights.append(f"📊 {len(strong_corr)} همبستگی قوی بین ویژگی‌ها یافت شد.")
        
        # 5. Insight درباره توزیع داده‌ها
        if 'distribution' in self.analysis_results:
            normal_cols = 0
            for col, info in self.analysis_results['distribution'].items():
                if 'normality_tests' in info and info['normality_tests']:
                    tests = info['normality_tests']
                    if 'dagostino' in tests and tests['dagostino'].get('is_normal', False):
                        normal_cols += 1
            if normal_cols > 0:
                insights.append(f"📈 {normal_cols} ویژگی دارای توزیع نرمال هستند.")
        
        # 6. Insight درباره ستون‌های دسته‌بندی
        if 'categorical_analysis' in self.analysis_results:
            high_cardinality = 0
            for col, info in self.analysis_results['categorical_analysis'].items():
                if info['unique_values'] > 100:
                    high_cardinality += 1
            if high_cardinality > 0:
                insights.append(f"🏷️ {high_cardinality} ستون دسته‌بندی با تنوع بالا (بیش از ۱۰۰ مقدار یکتا) وجود دارد.")
        
        return insights