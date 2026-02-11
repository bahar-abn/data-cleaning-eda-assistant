"""
کامپوننت‌های قابل استفاده مجدد برای Streamlit
این ماژول شامل کامپوننت‌های UI قابل استفاده مجدد است
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Optional, Dict, Any, List

def metric_card(title: str, value: Any, delta: Optional[str] = None, help_text: Optional[str] = None):
    """
    نمایش کارت متریک با استایل زیبا
    
    پارامترها:
        title: عنوان
        value: مقدار
        delta: تغییر
        help_text: متن راهنما
    """
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    ">
        <h3 style="margin: 0; font-size: 16px; opacity: 0.9;">{title}</h3>
        <p style="margin: 10px 0; font-size: 28px; font-weight: bold;">{value}</p>
        {f'<p style="margin: 0; font-size: 14px;">{delta}</p>' if delta else ''}
        {f'<p style="margin: 5px 0 0 0; font-size: 12px; opacity: 0.8;">{help_text}</p>' if help_text else ''}
    </div>
    """, unsafe_allow_html=True)

def insight_box(message: str, icon: str = "💡", type: str = "info"):
    """
    نمایش باکس بینش
    
    پارامترها:
        message: پیام
        icon: آیکون
        type: نوع (info, warning, success, error)
    """
    colors = {
        "info": {"bg": "#ebf4ff", "border": "#4299e1"},
        "warning": {"bg": "#fffaf0", "border": "#ed8936"},
        "success": {"bg": "#f0fff4", "border": "#48bb78"},
        "error": {"bg": "#fff5f5", "border": "#f56565"}
    }
    
    color = colors.get(type, colors["info"])
    
    st.markdown(f"""
    <div style="
        background: {color['bg']};
        padding: 15px;
        border-radius: 8px;
        border-right: 4px solid {color['border']};
        margin: 10px 0;
        font-size: 14px;
    ">
        <span style="font-size: 18px; margin-left: 10px;">{icon}</span>
        {message}
    </div>
    """, unsafe_allow_html=True)

def data_profile(df: pd.DataFrame):
    """
    نمایش پروفایل کامل داده
    
    پارامترها:
        df: دیتافریم
    """
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        metric_card(
            "تعداد رکوردها",
            f"{df.shape[0]:,}",
            None,
            "تعداد سطرها"
        )
    
    with col2:
        metric_card(
            "تعداد ستون‌ها",
            df.shape[1],
            None,
            "تعداد ویژگی‌ها"
        )
    
    with col3:
        missing_total = df.isnull().sum().sum()
        missing_percent = (missing_total / (df.shape[0] * df.shape[1])) * 100
        metric_card(
            "مقادیر گمشده",
            f"{missing_total:,}",
            f"{missing_percent:.1f}%",
            "از کل داده‌ها"
        )
    
    with col4:
        memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        metric_card(
            "حافظه مصرفی",
            f"{memory_mb:.1f} MB",
            None,
            "میزان RAM مصرفی"
        )

def column_selector(df: pd.DataFrame, 
                   numeric_only: bool = False,
                   categorical_only: bool = False,
                   multi_select: bool = False,
                   key: str = "column_selector") -> List[str]:
    """
    انتخابگر ستون با قابلیت فیلتر
    
    پارامترها:
        df: دیتافریم
        numeric_only: فقط ستون‌های عددی
        categorical_only: فقط ستون‌های دسته‌بندی
        multi_select: انتخاب چندتایی
        key: کلید یکتا
    
    بازگشت:
        selected_columns: لیست ستون‌های انتخاب شده
    """
    columns = df.columns.tolist()
    
    if numeric_only:
        columns = df.select_dtypes(include=['number']).columns.tolist()
    elif categorical_only:
        columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if multi_select:
        selected = st.multiselect(
            "انتخاب ستون‌ها",
            options=columns,
            key=key
        )
    else:
        selected = st.selectbox(
            "انتخاب ستون",
            options=columns,
            key=key
        )
        selected = [selected] if selected else []
    
    return selected

def correlation_heatmap(corr_matrix: pd.DataFrame):
    """
    نمایش هیتمپ همبستگی با Plotly
    
    پارامترها:
        corr_matrix: ماتریس همبستگی
    """
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmin=-1,
        zmax=1,
        text=corr_matrix.round(2).values,
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="ماتریس همبستگی",
        xaxis_title="ویژگی‌ها",
        yaxis_title="ویژگی‌ها",
        width=600,
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)

def progress_steps(steps: List[str], current_step: int):
    """
    نمایش مراحل پیشرفت
    
    پارامترها:
        steps: لیست مراحل
        current_step: مرحله فعلی
    """
    cols = st.columns(len(steps))
    
    for idx, (col, step) in enumerate(zip(cols, steps)):
        with col:
            if idx < current_step:
                status = "✅"
                color = "#48bb78"
            elif idx == current_step:
                status = "🔄"
                color = "#4299e1"
            else:
                status = "⏳"
                color = "#a0aec0"
            
            st.markdown(f"""
            <div style="text-align: center; padding: 10px;">
                <div style="
                    background: {color};
                    color: white;
                    width: 40px;
                    height: 40px;
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    margin: 0 auto;
                    font-size: 20px;
                ">
                    {status}
                </div>
                <p style="margin-top: 10px; font-size: 14px; color: {color};">
                    {step}
                </p>
            </div>
            """, unsafe_allow_html=True)