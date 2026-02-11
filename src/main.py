"""
فایل اصلی پروژه
این ماژول مسئول اجرای کل pipeline پردازش داده است
"""

import argparse
import pandas as pd
from pathlib import Path
import sys
import os

# اضافه کردن مسیر پروژه به PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import DataLoader
from src.data_cleaner import DataCleaner
from src.eda_analyzer import EDAAnalyzer
from src.visualizer import Visualizer
from src.report_generator import ReportGenerator
from src.utils import setup_logger, load_config

def main():
    """
    تابع اصلی اجرای پروژه
    """
    # تنظیم logger
    logger = setup_logger('main', log_file='outputs/logs/main.log')
    logger.info("🚀 Starting Data Cleaning and EDA Assistant...")
    
    # پارس کردن آرگومان‌های خط فرمان
    parser = argparse.ArgumentParser(
        description='دستیار هوشمند پاکسازی داده و EDA',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='مسیر فایل ورودی (CSV, Excel, JSON)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='outputs/reports',
        help='پوشه خروجی گزارش‌ها'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config/config.yaml',
        help='مسیر فایل تنظیمات'
    )
    
    parser.add_argument(
        '--skip-cleaning',
        action='store_true',
        help='پرش از مرحله پاکسازی'
    )
    
    parser.add_argument(
        '--save-processed',
        action='store_true',
        help='ذخیره داده‌های پردازش شده'
    )
    
    args = parser.parse_args()
    
    try:
        # 1. بارگذاری داده
        logger.info(f"📂 Loading data from: {args.input}")
        loader = DataLoader(args.config)
        df = loader.load_data(args.input)
        logger.info(f"✅ Data loaded successfully: {df.shape}")
        
        # 2. پاکسازی داده
        if not args.skip_cleaning:
            logger.info("🧹 Starting data cleaning...")
            cleaner = DataCleaner(df, args.config)
            df_clean, cleaning_report = cleaner.clean_all()
            logger.info(cleaner.get_cleaning_summary())
        else:
            logger.info("⏭️ Skipping data cleaning step")
            df_clean = df
            cleaning_report = {}
        
        # 3. ذخیره داده‌های پاکسازی شده
        if args.save_processed:
            output_path = Path('data/processed')
            output_path.mkdir(parents=True, exist_ok=True)
            processed_file = output_path / f"processed_{Path(args.input).stem}.csv"
            loader.save_processed_data(df_clean, str(processed_file))
            logger.info(f"💾 Processed data saved to: {processed_file}")
        
        # 4. تحلیل EDA
        logger.info("📊 Starting EDA analysis...")
        analyzer = EDAAnalyzer(df_clean)
        analysis_results = analyzer.generate_full_report()
        
        # اضافه کردن دیتافریم به نتایج برای استفاده در گزارش
        analysis_results['df'] = df_clean
        
        # اضافه کردن بینش‌ها
        insights = analyzer.get_insights()
        analysis_results['insights'] = insights
        
        logger.info("✅ EDA analysis completed")
        
        # 5. مصورسازی
        logger.info("🎨 Creating visualizations...")
        visualizer = Visualizer(df_clean, args.config)
        
        # ایجاد داشبورد
        figures = visualizer.create_dashboard()
        
        # ذخیره نمودارها
        for name, fig in figures.items():
            visualizer.save_figure(fig, name)
        
        logger.info(f"✅ Created {len(figures)} visualizations")
        
        # 6. تولید گزارش
        logger.info("📄 Generating reports...")
        report_gen = ReportGenerator(analysis_results, cleaning_report)
        saved_files = report_gen.save_all_reports(args.output)
        
        logger.info("✅ Reports generated successfully:")
        for format_type, filepath in saved_files.items():
            logger.info(f"   - {format_type}: {filepath}")
        
        # 7. نمایش خلاصه
        print("\n" + "="*60)
        print("✅ پردازش با موفقیت انجام شد!")
        print("="*60)
        print(f"📊 تعداد رکوردها: {df_clean.shape[0]:,}")
        print(f"📏 تعداد ویژگی‌ها: {df_clean.shape[1]}")
        print(f"📁 گزارش‌ها در پوشه {args.output} ذخیره شدند")
        print("="*60)
        
    except Exception as e:
        logger.error(f"❌ Error in main pipeline: {e}")
        print(f"\n❌ خطا در اجرای برنامه: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()