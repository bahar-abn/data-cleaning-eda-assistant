"""
ماژول مصورسازی داده
این ماژول مسئول ایجاد انواع نمودارها و بصری‌سازی‌های داده است
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import io
import base64
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.utils import setup_logger, load_config

class Visualizer:
    """
    کلاس مصورسازی داده
    پشتیبانی از هر دو کتابخانه matplotlib و plotly
    """
    
    def __init__(self, df: pd.DataFrame, config_path: Optional[str] = None):
        """
        سازنده کلاس Visualizer
        
        پارامترها:
            df: دیتافریم ورودی
            config_path: مسیر فایل تنظیمات
        """
        self.df = df.copy()
        
        # بارگذاری تنظیمات
        self.config = load_config(config_path) if config_path else load_config()
        self.viz_config = self.config.get('visualization', {})
        
        # تنظیم logger
        self.logger = setup_logger(
            'visualizer',
            log_file='outputs/logs/visualizer.log'
        )
        
        # تنظیمات پیش‌فرض مصورسازی
        plt.style.use('default')
        sns.set_palette(self.viz_config.get('color_palette', 'viridis'))
        
        self.figure_size = tuple(self.viz_config.get('figure_size', [12, 8]))
        self.dpi = self.viz_config.get('dpi', 100)
        
        self.logger.info(f"✅ Visualizer initialized with dataframe shape: {df.shape}")
    
    # -------------------- توابع کمکی --------------------
    
    def save_figure(self, fig, filename: str, format: str = 'png') -> str:
        """
        ذخیره نمودار در فایل
        
        پارامترها:
            fig: شیء figure
            filename: نام فایل
            format: فرمت خروجی
        
        بازگشت:
            filepath: مسیر فایل ذخیره شده
        """
        output_dir = Path('outputs/figures')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = output_dir / f"{filename}.{format}"
        
        if isinstance(fig, plt.Figure):
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        elif hasattr(fig, 'write_image'):  # plotly figure
            fig.write_image(filepath)
        
        self.logger.info(f"💾 Figure saved: {filepath}")
        
        return str(filepath)
    
    def fig_to_base64(self, fig) -> str:
        """
        تبدیل figure به رشته base64 برای نمایش در HTML
        
        پارامترها:
            fig: شیء figure
        
        بازگشت:
            base64_string: رشته base64
        """
        buf = io.BytesIO()
        
        if isinstance(fig, plt.Figure):
            fig.savefig(buf, format='png', dpi=self.dpi, bbox_inches='tight')
        elif hasattr(fig, 'write_image'):
            fig.write_image(buf, format='png')
        
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        
        return img_str
    
    # -------------------- هیستوگرام و توزیع --------------------
    
    def plot_histogram(self, 
                      column: str, 
                      bins: int = 30,
                      kde: bool = True,
                      title: Optional[str] = None,
                      use_plotly: bool = False) -> Union[plt.Figure, go.Figure]:
        """
        رسم هیستوگرام برای یک ستون
        
        پارامترها:
            column: نام ستون
            bins: تعداد bins
            kde: نمایش منحنی توزیع
            title: عنوان نمودار
            use_plotly: استفاده از plotly
        
        بازگشت:
            fig: شیء figure
        """
        self.logger.info(f"📊 Plotting histogram for column: {column}")
        
        if column not in self.df.columns:
            raise ValueError(f"ستون {column} در دیتافریم وجود ندارد")
        
        data = self.df[column].dropna()
        
        if use_plotly:
            # رسم با plotly
            fig = px.histogram(
                data,
                x=column,
                nbins=bins,
                title=title or f'توزیع {column}',
                marginal='box' if kde else None,
                opacity=0.8
            )
            
            fig.update_layout(
                xaxis_title=column,
                yaxis_title='فراوانی',
                showlegend=True
            )
            
        else:
            # رسم با matplotlib
            fig, ax = plt.subplots(figsize=self.figure_size)
            
            if kde:
                sns.histplot(data=data, kde=True, bins=bins, ax=ax)
            else:
                ax.hist(data, bins=bins, alpha=0.7, edgecolor='black')
            
            ax.set_xlabel(column)
            ax.set_ylabel('فراوانی')
            ax.set_title(title or f'توزیع {column}')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
        
        return fig
    
    # -------------------- باکس پلات --------------------
    
    def plot_boxplot(self,
                    columns: Optional[List[str]] = None,
                    by: Optional[str] = None,
                    title: Optional[str] = None,
                    use_plotly: bool = False) -> Union[plt.Figure, go.Figure]:
        """
        رسم باکس پلات
        
        پارامترها:
            columns: لیست ستون‌ها
            by: ستون دسته‌بندی
            title: عنوان نمودار
            use_plotly: استفاده از plotly
        
        بازگشت:
            fig: شیء figure
        """
        self.logger.info("📦 Plotting boxplot...")
        
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns[:5].tolist()
        
        if use_plotly:
            fig = go.Figure()
            
            for col in columns:
                if by and by in self.df.columns:
                    # گروه‌بندی بر اساس ستون دسته‌بندی
                    for category in self.df[by].unique():
                        data = self.df[self.df[by] == category][col].dropna()
                        fig.add_trace(go.Box(
                            y=data,
                            name=f"{col} - {category}",
                            boxmean='sd'
                        ))
                else:
                    fig.add_trace(go.Box(
                        y=self.df[col].dropna(),
                        name=col,
                        boxmean='sd'
                    ))
            
            fig.update_layout(
                title=title or 'باکس پلات',
                yaxis_title='مقدار',
                showlegend=True
            )
            
        else:
            fig, ax = plt.subplots(figsize=self.figure_size)
            
            if by and by in self.df.columns:
                data_to_plot = [self.df[self.df[by] == cat][col].dropna() 
                              for cat in self.df[by].unique() 
                              for col in columns]
                labels = [f"{col}\n{cat}" for cat in self.df[by].unique() for col in columns]
                ax.boxplot(data_to_plot, labels=labels)
            else:
                self.df[columns].boxplot(ax=ax)
            
            ax.set_title(title or 'باکس پلات')
            ax.set_ylabel('مقدار')
            ax.grid(True, alpha=0.3)
            
            plt.xticks(rotation=45)
            plt.tight_layout()
        
        return fig
    
    # -------------------- هیتمپ همبستگی --------------------
    
    def plot_correlation_heatmap(self,
                               method: str = 'pearson',
                               annot: bool = True,
                               title: Optional[str] = None,
                               use_plotly: bool = False) -> Union[plt.Figure, go.Figure]:
        """
        رسم هیتمپ ماتریس همبستگی
        
        پارامترها:
            method: روش همبستگی
            annot: نمایش مقادیر
            title: عنوان نمودار
            use_plotly: استفاده از plotly
        
        بازگشت:
            fig: شیء figure
        """
        self.logger.info(f"🔥 Plotting correlation heatmap using {method} method...")
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            self.logger.warning("⚠️ Need at least 2 numeric columns for correlation heatmap")
            return None
        
        corr_matrix = self.df[numeric_cols].corr(method=method)
        
        if use_plotly:
            fig = px.imshow(
                corr_matrix,
                text_auto=annot,
                aspect="auto",
                color_continuous_scale=self.viz_config.get('heatmap_cmap', 'RdBu_r'),
                title=title or f'ماتریس همبستگی ({method})'
            )
            
            fig.update_layout(
                xaxis_title='ویژگی‌ها',
                yaxis_title='ویژگی‌ها'
            )
            
        else:
            fig, ax = plt.subplots(figsize=self.figure_size)
            
            cmap = self.viz_config.get('heatmap_cmap', 'coolwarm')
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            
            sns.heatmap(
                corr_matrix,
                mask=mask,
                annot=annot,
                fmt='.2f',
                cmap=cmap,
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={'shrink': 0.8},
                ax=ax
            )
            
            ax.set_title(title or f'ماتریس همبستگی ({method})')
            plt.tight_layout()
        
        return fig
    
    # -------------------- نمودار پراکندگی --------------------
    
    def plot_scatter(self,
                    x: str,
                    y: str,
                    color: Optional[str] = None,
                    size: Optional[str] = None,
                    title: Optional[str] = None,
                    use_plotly: bool = False) -> Union[plt.Figure, go.Figure]:
        """
        رسم نمودار پراکندگی
        
        پارامترها:
            x: ستون محور x
            y: ستون محور y
            color: ستون برای رنگ‌بندی
            size: ستون برای اندازه نقاط
            title: عنوان نمودار
            use_plotly: استفاده از plotly
        
        بازگشت:
            fig: شیء figure
        """
        self.logger.info(f"📈 Plotting scatter plot: {x} vs {y}")
        
        if use_plotly:
            fig = px.scatter(
                self.df,
                x=x,
                y=y,
                color=color,
                size=size,
                title=title or f'{x} vs {y}',
                opacity=0.7,
                trendline='ols' if len(self.df) > 10 else None
            )
            
        else:
            fig, ax = plt.subplots(figsize=self.figure_size)
            
            if color and color in self.df.columns:
                # رنگ‌بندی بر اساس ستون دسته‌بندی
                categories = self.df[color].unique()
                for cat in categories[:10]:  # حداکثر 10 دسته
                    subset = self.df[self.df[color] == cat]
                    ax.scatter(subset[x], subset[y], label=cat, alpha=0.6, s=30)
                ax.legend()
            else:
                ax.scatter(self.df[x], self.df[y], alpha=0.6, s=30)
            
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            ax.set_title(title or f'{x} vs {y}')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
        
        return fig
    
    # -------------------- نمودار ستونی --------------------
    
    def plot_bar(self,
                x: str,
                y: Optional[str] = None,
                title: Optional[str] = None,
                horizontal: bool = False,
                use_plotly: bool = False) -> Union[plt.Figure, go.Figure]:
        """
        رسم نمودار ستونی
        
        پارامترها:
            x: ستون محور x (یا مقادیر)
            y: ستون محور y (اگر None باشد، count محاسبه می‌شود)
            title: عنوان نمودار
            horizontal: نمودار افقی
            use_plotly: استفاده از plotly
        
        بازگشت:
            fig: شیء figure
        """
        self.logger.info(f"📊 Plotting bar chart for {x}")
        
        if y is None:
            # محاسبه فراوانی
            data = self.df[x].value_counts().head(20)
            x_values = data.index
            y_values = data.values
            x_label = x
            y_label = 'تعداد'
        else:
            data = self.df.groupby(x)[y].mean().head(20)
            x_values = data.index
            y_values = data.values
            x_label = x
            y_label = f'میانگین {y}'
        
        if use_plotly:
            fig = px.bar(
                x=x_values if not horizontal else y_values,
                y=y_values if not horizontal else x_values,
                orientation='v' if not horizontal else 'h',
                title=title or f'نمودار {x}',
                labels={'x': x_label, 'y': y_label}
            )
            
        else:
            fig, ax = plt.subplots(figsize=self.figure_size)
            
            if horizontal:
                ax.barh(range(len(x_values)), y_values)
                ax.set_yticks(range(len(x_values)))
                ax.set_yticklabels([str(val) for val in x_values])
                ax.set_xlabel(y_label)
                ax.set_ylabel(x_label)
            else:
                ax.bar(range(len(x_values)), y_values)
                ax.set_xticks(range(len(x_values)))
                ax.set_xticklabels([str(val) for val in x_values], rotation=45, ha='right')
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)
            
            ax.set_title(title or f'نمودار {x}')
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
        
        return fig
    
    # -------------------- داشبورد کامل --------------------
    
    def create_dashboard(self, 
                        numeric_cols: Optional[List[str]] = None,
                        categorical_cols: Optional[List[str]] = None) -> Dict[str, Union[plt.Figure, go.Figure]]:
        """
        ایجاد داشبورد کامل از نمودارها
        
        پارامترها:
            numeric_cols: لیست ستون‌های عددی
            categorical_cols: لیست ستون‌های دسته‌بندی
        
        بازگشت:
            figures: دیکشنری نمودارها
        """
        self.logger.info("🚀 Creating complete visualization dashboard...")
        
        figures = {}
        
        # انتخاب ستون‌های پیش‌فرض
        if numeric_cols is None:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns[:4].tolist()
        
        if categorical_cols is None:
            categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns[:3].tolist()
        
        # 1. هیستوگرام برای ستون‌های عددی
        for col in numeric_cols[:3]:  # حداکثر 3 نمودار
            try:
                fig = self.plot_histogram(col, use_plotly=True)
                figures[f'histogram_{col}'] = fig
            except Exception as e:
                self.logger.warning(f"⚠️ Could not create histogram for {col}: {e}")
        
        # 2. باکس پلات برای ستون‌های عددی
        if numeric_cols:
            try:
                fig = self.plot_boxplot(numeric_cols[:5], use_plotly=True)
                figures['boxplot'] = fig
            except Exception as e:
                self.logger.warning(f"⚠️ Could not create boxplot: {e}")
        
        # 3. هیتمپ همبستگی
        try:
            fig = self.plot_correlation_heatmap(use_plotly=True)
            if fig:
                figures['correlation_heatmap'] = fig
        except Exception as e:
            self.logger.warning(f"⚠️ Could not create correlation heatmap: {e}")
        
        # 4. نمودار ستونی برای ستون‌های دسته‌بندی
        for col in categorical_cols[:2]:  # حداکثر 2 نمودار
            try:
                fig = self.plot_bar(col, use_plotly=True)
                figures[f'bar_{col}'] = fig
            except Exception as e:
                self.logger.warning(f"⚠️ Could not create bar chart for {col}: {e}")
        
        self.logger.info(f"✅ Dashboard created with {len(figures)} figures")
        
        return figures