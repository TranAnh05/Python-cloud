import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine, text
import time
# Import thư viện cho Machine Learning và Biểu đồ tương tác
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go

# --- Cấu hình trang ---
st.set_page_config(
    page_title="Dashboard Phân Tích Bán Hàng",
    page_icon="🛒",
    layout="wide"
)

# --- Hàm kết nối và nạp dữ liệu (Phiên bản Lưỡng cư: Chạy cả Local & Cloud) ---
@st.cache_data(show_spinner=False)
def load_and_process_data():
    # Cấu hình kết nối PostgreSQL
    db_connection_str = 'postgresql://admin:adminpassword@db:5432/sales_db'
    
    # Tạo engine kết nối (Thêm try/except để tránh lỗi ngay từ bước tạo engine)
    try:
        db_connection = create_engine(db_connection_str)
    except Exception:
        db_connection = None

    df = pd.DataFrame() # Khởi tạo rỗng

    # 1. Cố gắng đọc từ Database trước
    if db_connection:
        try:
            # Thử kết nối nhanh xem DB có sống không
            with db_connection.connect() as connection:
                connection.execute(text("SELECT 1"))
            
            # Nếu sống thì đọc dữ liệu
            df = pd.read_sql("SELECT * FROM sales_table", db_connection)
        except Exception:
            pass # Nếu lỗi kết nối, bỏ qua để xuống bước đọc CSV

    # 2. Nếu không đọc được từ DB (hoặc DB trống), đọc từ CSV
    if df.empty:
        try:
            df = pd.read_csv('supermarket_sales.csv')
            
            # Xử lý dữ liệu
            df['Date'] = pd.to_datetime(df['Date'])
            df['Time'] = df['Time'].astype(str)
            
            # --- ĐOẠN SỬA QUAN TRỌNG NHẤT Ở ĐÂY ---
            # Chỉ cố lưu vào DB nếu kết nối tồn tại. Nếu không (đang ở Cloud), thì bỏ qua.
            if db_connection:
                try:
                    df.to_sql('sales_table', db_connection, if_exists='replace', index=False)
                except Exception:
                    pass # Không lưu được thì thôi, không báo lỗi
                    
        except FileNotFoundError:
            st.error("Không tìm thấy file 'supermarket_sales.csv'.")
            return None
    
    # --- Tạo các cột phụ trợ ---
    if not df.empty:
        df['Date'] = pd.to_datetime(df['Date'])
        df['Month'] = df['Date'].dt.month
        try:
            df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M').dt.hour
        except:
            df['Hour'] = df['Time'].astype(str).str.split(':').str[0].astype(int)

    return df

# --- Gọi hàm load dữ liệu ---
with st.spinner('Đang kết nối Database và tải dữ liệu...'):
    df_full = load_and_process_data()

# --- Xây dựng Giao diện Dashboard ---
st.title('📊 Dashboard Phân Tích Dữ Liệu Bán Hàng Siêu Thị')
st.write('Tương tác với bộ lọc bên dưới để khám phá dữ liệu.')

if df_full is not None:

    # --- Bộ lọc (Sidebar) ---
    st.sidebar.header('Bộ lọc Tương tác ⚙️')
    
    cities = ['Tất cả'] + list(df_full['Branch'].unique())
    selected_city = st.sidebar.selectbox('Chọn Thành Phố (Branch):', cities)

    months = sorted(df_full['Month'].unique())
    selected_month = st.sidebar.multiselect(
        'Chọn Tháng:',
        options=months,
        default=months
    )

    # Áp dụng bộ lọc
    df_filtered = df_full.copy()
    if selected_city != 'Tất cả':
        df_filtered = df_filtered[df_filtered['Branch'] == selected_city]
    
    if selected_month:
        df_filtered = df_filtered[df_filtered['Month'].isin(selected_month)]
    else:
        st.warning("Vui lòng chọn ít nhất một tháng.")
        df_filtered = pd.DataFrame()

    st.header(f'Kết quả cho: {selected_city} - Tháng: {", ".join(map(str, selected_month))}')

    if not df_filtered.empty:
        # --- KPI ---
        st.subheader('📈 Chỉ số Hiệu suất Chính (KPIs)')
        col1, col2, col3 = st.columns(3)
        total_revenue = df_filtered['Sales'].sum()
        avg_rating = df_filtered['Rating'].mean()
        avg_sale_value = df_filtered['Sales'].mean()

        col1.metric("Tổng Doanh Thu", f"${total_revenue:,.0f}")
        col2.metric("Đánh Giá TB", f"{avg_rating:.1f}/10 ⭐")
        col3.metric("Hóa Đơn TB", f"${avg_sale_value:,.2f}")

        st.markdown("---")

        # --- Biểu đồ chính (Matplotlib/Seaborn) ---
        fig_col1, fig_col2 = st.columns(2)

        with fig_col1:
            st.subheader('💰 Doanh thu theo Mặt hàng')
            sales_by_product = df_filtered.groupby('Product line')['Sales'].sum().sort_values(ascending=False)
            st.bar_chart(sales_by_product)

            st.subheader('💳 Cơ cấu Hình thức Thanh toán')
            payment_counts = df_filtered['Payment'].value_counts()
            fig_pie, ax_pie = plt.subplots(figsize=(5, 5))
            ax_pie.pie(payment_counts, labels=payment_counts.index, autopct='%1.1f%%', 
                       startangle=90, colors=sns.color_palette('pastel'))
            ax_pie.axis('equal')
            st.pyplot(fig_pie)

        with fig_col2:
            st.subheader('⏰ Doanh thu theo Giờ trong Ngày')
            sales_by_hour = df_filtered.groupby('Hour')['Sales'].sum()
            st.line_chart(sales_by_hour)

            st.subheader('⭐ Tương quan Đánh giá & Doanh thu')
            fig_scatter, ax_scatter = plt.subplots(figsize=(6, 5))
            sns.scatterplot(data=df_filtered, x='Rating', y='Sales', hue='Gender', alpha=0.6, ax=ax_scatter)
            st.pyplot(fig_scatter)

        # --- PHẦN MACHINE LEARNING & PLOTLY ---
        st.markdown("---")
        st.header('🤖 Dự Báo Doanh Thu (Machine Learning)')
        
        # 1. Chuẩn bị dữ liệu
        daily_sales = df_full.groupby('Date')['Sales'].sum().reset_index()
        daily_sales['Date_Ordinal'] = daily_sales['Date'].map(pd.Timestamp.toordinal)

        # 2. Huấn luyện mô hình
        X = daily_sales[['Date_Ordinal']]
        y = daily_sales['Sales']
        model = LinearRegression()
        model.fit(X, y)

        # 3. Dự báo
        trend_y = model.predict(X)
        last_date = daily_sales['Date'].max()
        future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 31)]
        future_ordinals = [[pd.Timestamp(d).toordinal()] for d in future_dates]
        future_sales = model.predict(future_ordinals)

        # 4. Vẽ biểu đồ Plotly
        fig = go.Figure()
        
        # Dữ liệu thực tế
        fig.add_trace(go.Scatter(x=daily_sales['Date'], y=daily_sales['Sales'], 
                                mode='lines', name='Thực tế', line=dict(color='#636EFA')))
        # Đường xu hướng
        fig.add_trace(go.Scatter(x=daily_sales['Date'], y=trend_y, 
                                mode='lines', name='Xu hướng', line=dict(color='#EF553B', dash='dash')))
        # Dự báo tương lai
        fig.add_trace(go.Scatter(x=future_dates, y=future_sales, 
                                mode='lines+markers', name='Dự báo 30 ngày', line=dict(color='#00CC96')))

        fig.update_layout(title='Dự báo Doanh thu Tương lai', xaxis_title='Thời gian', yaxis_title='Doanh thu')
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Không có dữ liệu phù hợp với bộ lọc.")