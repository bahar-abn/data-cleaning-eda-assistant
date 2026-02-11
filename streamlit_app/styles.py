"""
ماژول استایل‌های CSS برای برنامه Streamlit
"""

def load_css():
    """
    بارگذاری استایل‌های CSS
    """
    return """
    <style>
        /* فونت فارسی */
        @import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@300;400;700&display=swap');
        
        * {
            font-family: 'Vazirmatn', sans-serif;
        }
        
        /* هدر اصلی */
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 1rem;
            margin-bottom: 2rem;
            text-align: center;
            box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1);
        }
        
        /* کارت آمار */
        .stat-card {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 0.75rem;
            text-align: center;
            box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }
        
        .stat-card:hover {
            transform: translateY(-5px);
        }
        
        /* باکس بینش */
        .insight-box {
            background: #ebf4ff;
            padding: 1rem;
            border-radius: 0.5rem;
            border-right: 4px solid #4299e1;
            margin: 0.75rem 0;
            transition: all 0.3s ease;
        }
        
        .insight-box:hover {
            background: #bee3f8;
            border-right-color: #2b6cb0;
        }
        
        /* دکمه‌ها */
        .stButton button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 0.5rem;
            font-weight: bold;
            transition: all 0.3s ease;
            width: 100%;
        }
        
        .stButton button:hover {
            transform: scale(1.02);
            box-shadow: 0 10px 15px -3px rgba(102, 126, 234, 0.4);
        }
        
        /* نوار پیشرفت */
        .stProgress > div > div {
            background: linear-gradient(90deg, #667eea, #764ba2);
        }
        
        /* تب‌ها */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2rem;
            background: white;
            padding: 0.5rem;
            border-radius: 0.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 0.375rem;
            padding: 0.5rem 1rem;
            font-weight: 600;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        /* سایدبار */
        .css-1v3fvcr {
            background: linear-gradient(180deg, #f7fafc 0%, #edf2f7 100%);
        }
        
        /* هشدارها */
        .stAlert {
            border-radius: 0.5rem;
            border-right-width: 4px;
        }
        
        /* فوتر */
        .footer {
            text-align: center;
            padding: 2rem;
            color: #718096;
            font-size: 0.875rem;
            margin-top: 3rem;
            border-top: 1px solid #e2e8f0;
        }
        
        /* جدول‌ها */
        .dataframe {
            border-collapse: collapse;
            width: 100%;
            margin: 1rem 0;
        }
        
        .dataframe th {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 0.75rem;
            text-align: right;
        }
        
        .dataframe td {
            padding: 0.75rem;
            border-bottom: 1px solid #e2e8f0;
        }
        
        .dataframe tr:hover {
            background: #f7fafc;
        }
        
        /* ریسپانسیو */
        @media (max-width: 768px) {
            .main-header h1 {
                font-size: 1.5rem;
            }
            
            .stat-card {
                padding: 1rem;
            }
        }
        
        /* انیمیشن‌ها */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .fade-in {
            animation: fadeIn 0.5s ease-in;
        }
        
        /* اسکرول بار سفارشی */
        ::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }
        
        ::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        }
        
        /* ابزارک‌ها */
        .tooltip {
            position: relative;
            display: inline-block;
        }
        
        .tooltip .tooltiptext {
            visibility: hidden;
            width: 120px;
            background: #2d3748;
            color: white;
            text-align: center;
            padding: 5px;
            border-radius: 6px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -60px;
            opacity: 0;
            transition: opacity 0.3s;
        }
        
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
        
        /* حالت تاریک */
        @media (prefers-color-scheme: dark) {
            body {
                background: #1a202c;
                color: #e2e8f0;
            }
            
            .stTabs [data-baseweb="tab-list"] {
                background: #2d3748;
            }
            
            .insight-box {
                background: #2d3748;
                color: #e2e8f0;
            }
        }
    </style>
    """