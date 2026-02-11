"""
ماژول تولید گزارش
این ماژول مسئول تولید گزارش‌های تحلیلی در فرمت‌های مختلف است
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
import json
import os
from pathlib import Path
import jinja2

from src.utils import setup_logger, save_json_report, save_text_report, format_number

class ReportGenerator:
    """
    کلاس تولید گزارش‌های تحلیلی
    """
    
    def __init__(self, 
                 analysis_results: Dict[str, Any],
                 cleaning_report: Optional[Dict[str, Any]] = None):
        """
        سازنده کلاس ReportGenerator
        
        پارامترها:
            analysis_results: نتایج تحلیل EDA
            cleaning_report: گزارش پاکسازی داده
        """
        self.analysis_results = analysis_results
        self.cleaning_report = cleaning_report or {}
        
        self.logger = setup_logger(
            'report_generator',
            log_file='outputs/logs/report_generator.log'
        )
        
        # تنظیم Jinja2 برای قالب‌های HTML
        self.template_dir = Path('templates')
        self.template_dir.mkdir(exist_ok=True)
        
        self.logger.info("✅ ReportGenerator initialized")
    
    # -------------------- گزارش HTML --------------------
    
    def generate_html_report(self) -> str:
        """
        تولید گزارش HTML
        
        بازگشت:
            html_content: محتوای HTML گزارش
        """
        self.logger.info("📄 Generating HTML report...")
        
        # قالب ساده HTML
        html_template = """
        <!DOCTYPE html>
        <html dir="rtl" lang="fa">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>گزارش تحلیل داده - {{ timestamp }}</title>
            <style>
                body {
                    font-family: 'Vazir', 'Tahoma', sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f5f5f5;
                }
                .header {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 30px;
                    border-radius: 10px;
                    margin-bottom: 30px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                }
                .section {
                    background: white;
                    padding: 25px;
                    border-radius: 10px;
                    margin-bottom: 25px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                h1 {
                    margin: 0;
                    font-size: 28px;
                }
                h2 {
                    color: #4a5568;
                    border-bottom: 2px solid #667eea;
                    padding-bottom: 10px;
                    margin-top: 0;
                }
                .stats-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin-top: 20px;
                }
                .stat-card {
                    background: #f8fafc;
                    padding: 20px;
                    border-radius: 8px;
                    border-left: 4px solid #667eea;
                }
                .stat-value {
                    font-size: 28px;
                    font-weight: bold;
                    color: #2d3748;
                    margin: 10px 0;
                }
                .stat-label {
                    font-size: 14px;
                    color: #718096;
                    text-transform: uppercase;
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 20px;
                }
                th {
                    background: #667eea;
                    color: white;
                    padding: 12px;
                    text-align: right;
                }
                td {
                    padding: 12px;
                    border-bottom: 1px solid #e2e8f0;
                }
                tr:hover {
                    background: #f7fafc;
                }
                .insight {
                    background: #ebf4ff;
                    padding: 15px;
                    border-radius: 8px;
                    margin-top: 20px;
                    border-right: 4px solid #4299e1;
                }
                .footer {
                    text-align: center;
                    padding: 20px;
                    color: #718096;
                    font-size: 14px;
                }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📊 گزارش تحلیل اکتشافی داده (EDA)</h1>
                <p>تاریخ تولید: {{ timestamp }}</p>
                <p>نام مجموعه داده: {{ dataset_name }}</p>
            </div>
            
            <div class="section">
                <h2>📋 خلاصه اطلاعات</h2>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-label">تعداد رکوردها</div>
                        <div class="stat-value">{{ summary.total_rows|format_number }}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">تعداد ویژگی‌ها</div>
                        <div class="stat-value">{{ summary.total_columns }}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">مقادیر گمشده</div>
                        <div class="stat-value">{{ summary.total_missing_values|format_number }}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">حافظه مصرفی</div>
                        <div class="stat-value">{{ summary.memory_usage_mb }} MB</div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>🔍 اطلاعات پایه</h2>
                <table>
                    <tr>
                        <th>ویژگی</th>
                        <th>نوع داده</th>
                        <th>مقادیر یکتا</th>
                        <th>مقادیر گمشده (%)</th>
                    </tr>
                    {% for col, info in distribution.items() %}
                    <tr>
                        <td><strong>{{ col }}</strong></td>
                        <td>{{ info.data_type }}</td>
                        <td>{{ info.unique_count }}</td>
                        <td>{{ info.missing_percentage }}%</td>
                    </tr>
                    {% endfor %}
                </table>
            </div>
            
            <div class="section">
                <h2>💡 بینش‌های کلیدی</h2>
                {% for insight in insights %}
                <div class="insight">
                    {{ insight }}
                </div>
                {% endfor %}
            </div>
            
            <div class="section">
                <h2>🧹 گزارش پاکسازی</h2>
                {% if cleaning_report %}
                <p><strong>ابعاد اولیه:</strong> {{ cleaning_report.initial_shape[0] }} سطر × {{ cleaning_report.initial_shape[1] }} ستون</p>
                <p><strong>ابعاد نهایی:</strong> {{ cleaning_report.final_shape[0] }} سطر × {{ cleaning_report.final_shape[1] }} ستون</p>
                
                {% if cleaning_report.duplicates > 0 %}
                <p><strong>رکوردهای تکراری حذف شده:</strong> {{ cleaning_report.duplicates }}</p>
                {% endif %}
                
                <h3>عملیات انجام شده:</h3>
                <ul>
                    {% for op in cleaning_report.operations %}
                    <li>{{ op.operation }} - {{ op.column if op.column else '' }}</li>
                    {% endfor %}
                </ul>
                {% else %}
                <p>گزارش پاکسازی موجود نیست</p>
                {% endif %}
            </div>
            
            <div class="footer">
                <p>تولید شده توسط دستیار هوشمند پاکسازی داده و EDA</p>
                <p>© 2024 - تمامی حقوق محفوظ است</p>
            </div>
        </body>
        </html>
        """
        
        # آماده‌سازی داده‌ها برای قالب
        template = jinja2.Template(html_template)
        
        # اضافه کردن تابع format_number به Jinja2
        def format_number_filter(value):
            if isinstance(value, (int, float)):
                return f"{value:,}"
            return str(value)
        
        template.environment.filters['format_number'] = format_number_filter
        
        # استخراج داده‌ها
        summary = self.analysis_results.get('summary', {})
        distribution = self.analysis_results.get('distribution', {})
        insights = self.analysis_results.get('insights', [])
        
        if not insights:
            # اگر insights وجود نداشت، از eda_analyzer دریافت کنیم
            from src.eda_analyzer import EDAAnalyzer
            if 'df' in self.analysis_results:
                analyzer = EDAAnalyzer(self.analysis_results['df'])
                insights = analyzer.get_insights()
        
        html_content = template.render(
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            dataset_name=summary.get('dataset_name', 'Unknown'),
            summary=summary,
            distribution=distribution,
            insights=insights,
            cleaning_report=self.cleaning_report
        )
        
        self.logger.info("✅ HTML report generated successfully")
        
        return html_content
    
    # -------------------- گزارش JSON --------------------
    
    def generate_json_report(self) -> Dict[str, Any]:
        """
        تولید گزارش JSON
        
        بازگشت:
            json_data: دیکشنری کامل گزارش
        """
        self.logger.info("📄 Generating JSON report...")
        
        report = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'version': '1.0.0'
            },
            'analysis': self.analysis_results,
            'cleaning': self.cleaning_report,
            'summary': self.analysis_results.get('summary', {})
        }
        
        # اضافه کردن insights
        if 'insights' not in report['analysis']:
            from src.eda_analyzer import EDAAnalyzer
            if 'df' in self.analysis_results:
                analyzer = EDAAnalyzer(self.analysis_results['df'])
                report['analysis']['insights'] = analyzer.get_insights()
        
        self.logger.info("✅ JSON report generated successfully")
        
        return report
    
    # -------------------- گزارش متنی ساده --------------------
    
    def generate_text_report(self) -> str:
        """
        تولید گزارش متنی ساده
        
        بازگشت:
            text_content: محتوای متنی گزارش
        """
        self.logger.info("📄 Generating text report...")
        
        lines = []
        lines.append("="*60)
        lines.append("📊 گزارش تحلیل اکتشافی داده (EDA)")
        lines.append("="*60)
        lines.append(f"تاریخ تولید: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # خلاصه اطلاعات
        summary = self.analysis_results.get('summary', {})
        lines.append("📋 خلاصه اطلاعات:")
        lines.append("-"*40)
        lines.append(f"تعداد رکوردها: {summary.get('total_rows', 0):,}")
        lines.append(f"تعداد ویژگی‌ها: {summary.get('total_columns', 0)}")
        lines.append(f"ستون‌های عددی: {summary.get('numeric_columns', 0)}")
        lines.append(f"ستون‌های دسته‌بندی: {summary.get('categorical_columns', 0)}")
        lines.append(f"مقادیر گمشده: {summary.get('total_missing_values', 0):,}")
        lines.append(f"رکوردهای تکراری: {summary.get('total_duplicates', 0):,}")
        lines.append(f"حافظه مصرفی: {summary.get('memory_usage_mb', 0)} MB")
        lines.append("")
        
        # اطلاعات ویژگی‌ها
        distribution = self.analysis_results.get('distribution', {})
        if distribution:
            lines.append("🔍 اطلاعات ویژگی‌ها:")
            lines.append("-"*40)
            for col, info in list(distribution.items())[:20]:  # حداکثر 20 ستون
                lines.append(f"• {col}:")
                lines.append(f"  - نوع: {info.get('data_type', 'نامشخص')}")
                lines.append(f"  - مقادیر یکتا: {info.get('unique_count', 0)}")
                lines.append(f"  - مقادیر گمشده: {info.get('missing_percentage', 0)}%")
            if len(distribution) > 20:
                lines.append(f"... و {len(distribution) - 20} ستون دیگر")
            lines.append("")
        
        # بینش‌ها
        insights = self.analysis_results.get('insights', [])
        if not insights:
            from src.eda_analyzer import EDAAnalyzer
            if 'df' in self.analysis_results:
                analyzer = EDAAnalyzer(self.analysis_results['df'])
                insights = analyzer.get_insights()
        
        if insights:
            lines.append("💡 بینش‌های کلیدی:")
            lines.append("-"*40)
            for insight in insights:
                lines.append(f"• {insight}")
            lines.append("")
        
        # گزارش پاکسازی
        if self.cleaning_report:
            lines.append("🧹 گزارش پاکسازی:")
            lines.append("-"*40)
            initial = self.cleaning_report.get('initial_shape', (0, 0))
            final = self.cleaning_report.get('final_shape', (0, 0))
            lines.append(f"ابعاد اولیه: {initial[0]} سطر × {initial[1]} ستون")
            lines.append(f"ابعاد نهایی: {final[0]} سطر × {final[1]} ستون")
            lines.append(f"رکوردهای تکراری حذف شده: {self.cleaning_report.get('duplicates', 0)}")
            
            if 'operations' in self.cleaning_report:
                lines.append("\nعملیات انجام شده:")
                for op in self.cleaning_report['operations'][:10]:
                    lines.append(f"  - {op.get('operation', '')}")
        
        lines.append("")
        lines.append("="*60)
        lines.append("✅ پایان گزارش")
        lines.append("="*60)
        
        self.logger.info("✅ Text report generated successfully")
        
        return "\n".join(lines)
    
    # -------------------- ذخیره گزارش‌ها --------------------
    
    def save_all_reports(self, output_dir: str = 'outputs/reports') -> Dict[str, str]:
        """
        ذخیره تمام فرمت‌های گزارش
        
        پارامترها:
            output_dir: پوشه خروجی
        
        بازگشت:
            saved_files: دیکشنری مسیر فایل‌های ذخیره شده
        """
        self.logger.info(f"💾 Saving all reports to {output_dir}...")
        
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        saved_files = {}
        
        # ذخیره گزارش HTML
        try:
            html_content = self.generate_html_report()
            html_path = f"{output_dir}/eda_report_{timestamp}.html"
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            saved_files['html'] = html_path
            self.logger.info(f"✅ HTML report saved: {html_path}")
        except Exception as e:
            self.logger.error(f"❌ Failed to save HTML report: {e}")
        
        # ذخیره گزارش JSON
        try:
            json_data = self.generate_json_report()
            json_path = f"{output_dir}/eda_report_{timestamp}.json"
            save_json_report(json_data, json_path)
            saved_files['json'] = json_path
            self.logger.info(f"✅ JSON report saved: {json_path}")
        except Exception as e:
            self.logger.error(f"❌ Failed to save JSON report: {e}")
        
        # ذخیره گزارش متنی
        try:
            text_content = self.generate_text_report()
            text_path = f"{output_dir}/eda_report_{timestamp}.txt"
            save_text_report(text_content, text_path)
            saved_files['txt'] = text_path
            self.logger.info(f"✅ Text report saved: {text_path}")
        except Exception as e:
            self.logger.error(f"❌ Failed to save text report: {e}")
        
        return saved_files