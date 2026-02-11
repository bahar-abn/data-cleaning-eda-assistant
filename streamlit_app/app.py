"""
برنامه اصلی Streamlit
این فایل رابط کاربری گرافیکی پروژه را پیاده‌سازی می‌کند
"""

import streamlit as st
import pandas as pd
import sys
import os
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

# اضافه کردن مسیر پروژه
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import DataLoader
from src.data_cleaner import DataCleaner
from src.eda_analyzer import EDAAnalyzer
from src.visualizer import Visualizer
from src.report_generator import ReportGenerator
from src.utils import format_number

# تنظیمات صفحه
st.set_page_config(
    page_title="دستیار هوشمند پاکسازی داده و EDA",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# استایل CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@300;400;700&display=swap');
    
    * {
        font-family: 'Vazirmatn', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 30px;
        border-radius: 15px;
        margin-bottom: 30px;
        text-align: center;
    }
    
    .stat-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .insight-box {
        background: #ebf4ff;
        padding: 15px;
        border-radius: 8px;
        border-right: 4px solid #4299e1;
        margin: 10px 0;
    }
    
    .stProgress > div > div {
        background-color: #667eea;
    }
    
    .css-1v3fvcr {
        background-color: #f7fafc;
    }
    
    h1, h2, h3 {
        color: #2d3748;
    }
</style>
""", unsafe_allow_html=True)

# -------------------- Session State Initialization --------------------

def init_session_state():
    """
    مقداردهی اولیه session state
    """
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'df_clean' not in st.session_state:
        st.session_state.df_clean = None
    if 'cleaner' not in st.session_state:
        st.session_state.cleaner = None
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'cleaning_report' not in st.session_state:
        st.session_state.cleaning_report = None
    if 'filename' not in st.session_state:
        st.session_state.filename = None

init_session_state()

# -------------------- Header --------------------

st.markdown("""
<div class="main-header">
    <h1>📊 دستیار هوشمند پاکسازی داده و EDA</h1>
    <p style="font-size: 18px; margin-top: 10px;">تحلیل خودکار داده، پاکسازی هوشمند و مصورسازی پیشرفته</p>
</div>
""", unsafe_allow_html=True)

# -------------------- Sidebar --------------------

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/data-configuration.png", width=100)
    st.title("📁 کنترل پنل")
    st.markdown("---")
    
    # آپلود فایل
    st.subheader("📂 بارگذاری داده")
    uploaded_file = st.file_uploader(
        "فایل داده خود را انتخاب کنید",
        type=['csv', 'xlsx', 'xls', 'json'],
        help="فرمت‌های پشتیبانی شده: CSV, Excel, JSON"
    )
    
    if uploaded_file is not None:
        st.session_state.filename = uploaded_file.name
        
        # دکمه بارگذاری
        if st.button("🚀 بارگذاری و تحلیل", type="primary", use_container_width=True):
            with st.spinner("در حال بارگذاری داده..."):
                try:
                    loader = DataLoader()
                    
                    # ذخیره فایل موقت
                    temp_path = f"data/raw/{uploaded_file.name}"
                    os.makedirs("data/raw", exist_ok=True)
                    
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # بارگذاری داده
                    st.session_state.df = loader.load_data(temp_path)
                    st.success(f"✅ داده با موفقیت بارگذاری شد! {st.session_state.df.shape[0]:,} رکورد و {st.session_state.df.shape[1]} ستون")
                    
                except Exception as e:
                    st.error(f"❌ خطا در بارگذاری فایل: {e}")
    
    st.markdown("---")
    
    # تنظیمات پیشرفته
    with st.expander("⚙️ تنظیمات پیشرفته"):
        st.session_state.missing_threshold = st.slider(
            "آستانه حذف ستون‌های دارای مقادیر گمشده (%)",
            min_value=10,
            max_value=90,
            value=50,
            step=5
        )
        
        st.session_state.outlier_method = st.selectbox(
            "روش تشخیص داده‌های پرت",
            options=["IQR", "Z-Score"],
            index=0
        )
        
        st.session_state.save_processed = st.checkbox(
            "ذخیره داده‌های پاکسازی شده",
            value=True
        )
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #718096; font-size: 12px;">
        توسعه داده شده با ❤️ برای جامعه داده کاوی ایران
    </div>
    """, unsafe_allow_html=True)

# -------------------- Main Content --------------------

if st.session_state.df is not None:
    # ایجاد تب‌ها
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 پیش‌نمایش داده",
        "🧹 پاکسازی داده",
        "📊 تحلیل EDA",
        "🎨 مصورسازی",
        "📄 گزارش"
    ])
    
    # -------------------- Tab 1: Data Preview --------------------
    
    with tab1:
        st.header("📋 پیش‌نمایش داده‌ها")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("تعداد رکوردها", f"{st.session_state.df.shape[0]:,}")
        with col2:
            st.metric("تعداد ستون‌ها", st.session_state.df.shape[1])
        with col3:
            missing_total = st.session_state.df.isnull().sum().sum()
            st.metric("مقادیر گمشده", f"{missing_total:,}", delta_color="inverse")
        with col4:
            memory_mb = st.session_state.df.memory_usage(deep=True).sum() / (1024 * 1024)
            st.metric("حافظه مصرفی", f"{memory_mb:.1f} MB")
        
        st.subheader("نمایش 100 رکورد اول")
        st.dataframe(st.session_state.df.head(100), use_container_width=True)
        
        st.subheader("اطلاعات ستون‌ها")
        col_info = []
        for col in st.session_state.df.columns:
            col_info.append({
                "ستون": col,
                "نوع داده": str(st.session_state.df[col].dtype),
                "مقادیر غیر خالی": st.session_state.df[col].count(),
                "مقادیر یکتا": st.session_state.df[col].nunique(),
                "مقادیر گمشده": st.session_state.df[col].isnull().sum(),
                "درصد گمشده": f"{(st.session_state.df[col].isnull().sum() / len(st.session_state.df) * 100):.1f}%"
            })
        
        st.dataframe(pd.DataFrame(col_info), use_container_width=True)
    
    # -------------------- Tab 2: Data Cleaning --------------------
    
    with tab2:
        st.header("🧹 پاکسازی هوشمند داده")
        
        if st.button("🚀 اجرای پاکسازی خودکار", type="primary", use_container_width=True):
            with st.spinner("در حال پاکسازی داده..."):
                try:
                    cleaner = DataCleaner(st.session_state.df)
                    st.session_state.df_clean, st.session_state.cleaning_report = cleaner.clean_all()
                    st.session_state.cleaner = cleaner
                    st.success("✅ پاکسازی داده با موفقیت انجام شد!")
                except Exception as e:
                    st.error(f"❌ خطا در پاکسازی داده: {e}")
        
        if st.session_state.df_clean is not None:
            st.subheader("📊 گزارش پاکسازی")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**ابعاد قبل از پاکسازی:**")
                st.info(f"{st.session_state.cleaning_report['initial_shape'][0]:,} سطر × {st.session_state.cleaning_report['initial_shape'][1]} ستون")
            
            with col2:
                st.markdown("**ابعاد بعد از پاکسازی:**")
                st.info(f"{st.session_state.cleaning_report['final_shape'][0]:,} سطر × {st.session_state.cleaning_report['final_shape'][1]} ستون")
            
            st.subheader("عملیات انجام شده:")
            
            ops_df = []
            for op in st.session_state.cleaning_report.get('operations', [])[:10]:
                ops_df.append({
                    "عملیات": op.get('operation', ''),
                    "ستون": op.get('column', '-'),
                    "جزئیات": str(op.get('method', op.get('fill_value', '-')))
                })
            
            if ops_df:
                st.dataframe(pd.DataFrame(ops_df), use_container_width=True)
            
            st.subheader("داده‌های پاکسازی شده")
            st.dataframe(st.session_state.df_clean.head(100), use_container_width=True)
    
    # -------------------- Tab 3: EDA Analysis --------------------
    
    with tab3:
        st.header("📊 تحلیل اکتشافی داده (EDA)")
        
        df_for_analysis = st.session_state.df_clean if st.session_state.df_clean is not None else st.session_state.df
        
        if st.button("🔍 اجرای تحلیل کامل", type="primary", use_container_width=True):
            with st.spinner("در حال تحلیل داده..."):
                try:
                    analyzer = EDAAnalyzer(df_for_analysis)
                    st.session_state.analysis_results = analyzer.generate_full_report()
                    st.session_state.analysis_results['insights'] = analyzer.get_insights()
                    st.session_state.analyzer = analyzer
                    st.success("✅ تحلیل با موفقیت انجام شد!")
                except Exception as e:
                    st.error(f"❌ خطا در تحلیل داده: {e}")
        
        if st.session_state.analysis_results:
            # نمایش بینش‌ها
            st.subheader("💡 بینش‌های کلیدی")
            insights = st.session_state.analysis_results.get('insights', [])
            
            for insight in insights[:5]:
                st.markdown(f"""
                <div class="insight-box">
                    {insight}
                </div>
                """, unsafe_allow_html=True)
            
            # آمار توصیفی
            st.subheader("📈 آمار توصیفی")
            
            desc_stats = st.session_state.analyzer.get_descriptive_stats()
            if not desc_stats.empty:
                st.dataframe(desc_stats.round(2), use_container_width=True)
            
            # ماتریس همبستگی
            st.subheader("🔄 ماتریس همبستگی")
            
            correlation_info = st.session_state.analysis_results.get('correlation', {})
            high_corr = correlation_info.get('high_correlations', [])
            
            if high_corr:
                st.markdown("**همبستگی‌های قوی یافت شده:**")
                corr_df = pd.DataFrame(high_corr)
                st.dataframe(corr_df, use_container_width=True)
    
    # -------------------- Tab 4: Visualization --------------------
    
    with tab4:
        st.header("🎨 مصورسازی داده")
        
        df_for_viz = st.session_state.df_clean if st.session_state.df_clean is not None else st.session_state.df
        
        if st.button("🎨 ایجاد داشبورد تصویری", type="primary", use_container_width=True):
            with st.spinner("در حال ایجاد نمودارها..."):
                try:
                    visualizer = Visualizer(df_for_viz)
                    st.session_state.visualizer = visualizer
                    
                    # انتخاب ستون‌ها
                    numeric_cols = df_for_viz.select_dtypes(include=['number']).columns.tolist()
                    categorical_cols = df_for_viz.select_dtypes(include=['object', 'category']).columns.tolist()
                    
                    # ایجاد نمودارها
                    st.session_state.figures = visualizer.create_dashboard(
                        numeric_cols[:4],
                        categorical_cols[:2]
                    )
                    
                    st.success(f"✅ {len(st.session_state.figures)} نمودار با موفقیت ایجاد شد!")
                    
                except Exception as e:
                    st.error(f"❌ خطا در ایجاد نمودارها: {e}")
        
        if 'figures' in st.session_state and st.session_state.figures:
            # نمایش نمودارها
            for name, fig in st.session_state.figures.items():
                with st.expander(f"📊 {name}", expanded=True):
                    if hasattr(fig, 'write_html'):  # Plotly figure
                        st.plotly_chart(fig, use_container_width=True)
                    else:  # Matplotlib figure
                        st.pyplot(fig)
        
        # نمودار سفارشی
        st.subheader("📈 ایجاد نمودار سفارشی")
        
        col1, col2 = st.columns(2)
        
        with col1:
            chart_type = st.selectbox(
                "نوع نمودار",
                ["هیستوگرام", "باکس پلات", "نمودار پراکندگی", "نمودار ستونی"]
            )
        
        with col2:
            columns = df_for_viz.columns.tolist()
            x_col = st.selectbox("محور X", columns)
        
        if chart_type in ["باکس پلات", "نمودار پراکندگی"]:
            y_col = st.selectbox("محور Y", [col for col in columns if col != x_col])
        
        if st.button("رسم نمودار", use_container_width=True):
            try:
                visualizer = Visualizer(df_for_viz)
                
                if chart_type == "هیستوگرام":
                    fig = visualizer.plot_histogram(x_col, use_plotly=True)
                elif chart_type == "باکس پلات":
                    fig = visualizer.plot_boxplot([x_col], use_plotly=True)
                elif chart_type == "نمودار پراکندگی":
                    fig = visualizer.plot_scatter(x_col, y_col, use_plotly=True)
                elif chart_type == "نمودار ستونی":
                    fig = visualizer.plot_bar(x_col, use_plotly=True)
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ خطا در رسم نمودار: {e}")
    
    # -------------------- Tab 5: Report --------------------
    
    with tab5:
        st.header("📄 گزارش تحلیلی")
        
        if st.session_state.analysis_results:
            st.subheader("فرمت گزارش")
            
            report_format = st.radio(
                "فرمت مورد نظر را انتخاب کنید:",
                ["HTML", "JSON", "متن ساده"],
                horizontal=True
            )
            
            if st.button("📥 تولید و دانلود گزارش", type="primary", use_container_width=True):
                with st.spinner("در حال تولید گزارش..."):
                    try:
                        report_gen = ReportGenerator(
                            st.session_state.analysis_results,
                            st.session_state.cleaning_report
                        )
                        
                        if report_format == "HTML":
                            report_content = report_gen.generate_html_report()
                            mime_type = "text/html"
                            file_ext = "html"
                        elif report_format == "JSON":
                            report_content = report_gen.generate_json_report()
                            import json
                            report_content = json.dumps(report_content, indent=4, ensure_ascii=False)
                            mime_type = "application/json"
                            file_ext = "json"
                        else:
                            report_content = report_gen.generate_text_report()
                            mime_type = "text/plain"
                            file_ext = "txt"
                        
                        st.download_button(
                            label=f"📥 دانلود گزارش {report_format}",
                            data=report_content,
                            file_name=f"eda_report.{file_ext}",
                            mime=mime_type,
                            use_container_width=True
                        )
                        
                        st.success("✅ گزارش با موفقیت تولید شد!")
                        
                    except Exception as e:
                        st.error(f"❌ خطا در تولید گزارش: {e}")
            
            # پیش‌نمایش گزارش
            with st.expander("📋 پیش‌نمایش گزارش"):
                report_gen = ReportGenerator(
                    st.session_state.analysis_results,
                    st.session_state.cleaning_report
                )
                preview = report_gen.generate_text_report()
                st.text(preview[:2000] + "..." if len(preview) > 2000 else preview)
        
        else:
            st.info("لطفاً ابتدا تحلیل داده را در تب EDA اجرا کنید.")

else:
    # صفحه خوش‌آمدگویی
    st.markdown("""
    <div style="text-align: center; padding: 50px;">
        <h2 style="color: #4a5568;">👋 به دستیار هوشمند پاکسازی داده و EDA خوش آمدید!</h2>
        <p style="color: #718096; font-size: 18px; margin-top: 20px;">
            برای شروع، فایل داده خود را از طریق منوی کناری آپلود کنید.
        </p>
        <div style="margin-top: 40px;">
            <img src="https://img.icons8.com/fluency/96/000000/data-sheet.png" style="margin: 10px;">
            <img src="https://img.icons8.com/fluency/96/000000/bar-chart.png" style="margin: 10px;">
            <img src="https://img.icons8.com/fluency/96/000000/report.png" style="margin: 10px;">
        </div>
        <div style="margin-top: 40px; background: #f7fafc; padding: 30px; border-radius: 15px;">
            <h3 style="color: #2d3748;">✨ قابلیت‌های اصلی:</h3>
            <ul style="list-style: none; padding: 0; color: #4a5568; font-size: 16px;">
                <li style="margin: 10px 0;">✓ تشخیص و مدیریت مقادیر گمشده</li>
                <li style="margin: 10px 0;">✓ شناسایی داده‌های پرت</li>
                <li style="margin: 10px 0;">✓ تحلیل توزیع و همبستگی</li>
                <li style="margin: 10px 0;">✓ مصورسازی پیشرفته</li>
                <li style="margin: 10px 0;">✓ تولید گزارش خودکار</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)