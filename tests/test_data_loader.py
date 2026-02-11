"""
تست‌های ماژول DataLoader
"""

import unittest
import pandas as pd
import numpy as np
import os
import tempfile
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import DataLoader

class TestDataLoader(unittest.TestCase):
    """کلاس تست DataLoader"""
    
    def setUp(self):
        """مقداردهی اولیه قبل از هر تست"""
        self.loader = DataLoader()
        
        # ایجاد فایل CSV نمونه
        self.temp_dir = tempfile.mkdtemp()
        self.csv_path = os.path.join(self.temp_dir, 'test.csv')
        
        df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': ['a', 'b', 'c', 'd', 'e'],
            'col3': [1.1, 2.2, 3.3, 4.4, 5.5]
        })
        df.to_csv(self.csv_path, index=False)
    
    def tearDown(self):
        """پاکسازی بعد از هر تست"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_load_csv(self):
        """تست بارگذاری فایل CSV"""
        df = self.loader.load_csv(self.csv_path)
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape, (5, 3))
        self.assertEqual(list(df.columns), ['col1', 'col2', 'col3'])
    
    def test_get_file_info(self):
        """تست دریافت اطلاعات فایل"""
        info = self.loader.get_file_info(self.csv_path)
        
        self.assertEqual(info['filename'], 'test.csv')
        self.assertEqual(info['extension'], '.csv')
        self.assertGreater(info['size_bytes'], 0)
    
    def test_validate_file_extension(self):
        """تست اعتبارسنجی پسوند فایل"""
        # تست پسوند صحیح
        from src.utils import validate_file_extension
        self.assertTrue(validate_file_extension(self.csv_path, ['csv']))
        
        # تست پسوند غلط
        self.assertFalse(validate_file_extension(self.csv_path, ['xlsx']))
    
    def test_save_processed_data(self):
        """تست ذخیره داده‌های پردازش شده"""
        df = pd.DataFrame({'test': [1, 2, 3]})
        save_path = os.path.join(self.temp_dir, 'output.csv')
        
        self.loader.save_processed_data(df, save_path)
        
        self.assertTrue(os.path.exists(save_path))
        
        # تست محتوای فایل ذخیره شده
        df_loaded = pd.read_csv(save_path)
        self.assertEqual(df_loaded.shape, (3, 1))

if __name__ == '__main__':
    unittest.main()