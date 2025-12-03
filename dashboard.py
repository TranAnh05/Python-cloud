# --- File: dashboard.py ---

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Cấu hình trang ---
st.set_page_config(
    page_title="Dashboard Phân Tích Bán Hàng",
    page_icon="🛒",
    layout="wide" # Sử dụng layout rộng hơn
)

# --- Tải và Xử lý Dữ liệu ---
# Sử dụng @st.cache_data để tăng tốc độ tải lại khi không có thay đổi
@st.cache_data
def load_and_process_data():
    """Tải, làm sạch và tích hợp dữ liệu."""
    try:
        # Đọc file CSV
        df_sales = pd.read_csv('supermarket_sales.csv')

        # Đọc file JSON (nếu có)
        try:
            df_products = pd.read_json('product_info.json')
        except ValueError: # Xử lý trường hợp file JSON rỗng hoặc lỗi
            df_products = None
            st.warning("Không tìm thấy hoặc file 'product_info.json' bị lỗi. Sẽ bỏ qua tích hợp danh mục.")

        # --- Làm sạch và Chế biến ---
        # Chuẩn hóa Date và Time
        df_sales['Date'] = pd.to_datetime(df_sales['Date'])
        # Cố gắng chuyển đổi Time, bỏ qua lỗi nếu có định dạng không nhất quán
        try:
            df_sales['Time'] = pd.to_datetime(df_sales['Time'], format='%H:%M').dt.time
        except ValueError:
             # Nếu format '%H:%M' lỗi, thử format khác hoặc giữ nguyên object
             try:
                 df_sales['Time'] = pd.to_datetime(df_sales['Time']).dt.time
             except Exception:
                 st.warning("Không thể chuẩn hóa cột 'Time' do định dạng không nhất quán.")


        # Tạo cột Month và Hour
        df_sales['Month'] = df_sales['Date'].dt.month
        # Cập nhật cách lấy Hour an toàn hơn
        def get_hour(time_obj):
            try:
                return time_obj.hour
            except AttributeError:
                return None # Trả về None nếu không phải đối tượng time hợp lệ
        df_sales['Hour'] = df_sales['Time'].apply(get_hour)
        # Loại bỏ các dòng có Hour bị lỗi (nếu có)
        df_sales.dropna(subset=['Hour'], inplace=True)
        df_sales['Hour'] = df_sales['Hour'].astype(int)


        # --- Tích hợp Dữ liệu (Merge) ---
        if df_products is not None:
            df_full = pd.merge(df_sales, df_products, on='Product line', how='left')
        else:
            df_full = df_sales
            df_full['Category'] = df_full['Product line'] # Tạm dùng Product line nếu không có Category

        return df_full

    except FileNotFoundError:
        st.error("Lỗi: Không tìm thấy file 'supermarket_sales.csv'. Vui lòng đặt file vào cùng thư mục với dashboard.py.")
        return None

# Gọi hàm tải dữ liệu
df_full = load_and_process_data()

# --- Xây dựng Giao diện Dashboard ---
st.title('📊 Dashboard Phân Tích Dữ Liệu Bán Hàng Siêu Thị')
st.write('Tương tác với bộ lọc bên dưới để khám phá dữ liệu.')

# Chỉ hiển thị dashboard nếu tải dữ liệu thành công
if df_full is not None:

    # --- Bộ lọc (Sidebar) ---
    st.sidebar.header('Bộ lọc Tương tác ⚙️')
    # 1. Lọc theo Thành phố (Branch)
    selected_city = st.sidebar.selectbox(
        'Chọn Thành Phố (Branch):',
        ['Tất cả'] + list(df_full['Branch'].unique())
    )

    # 2. Lọc theo Tháng (Month)
    selected_month = st.sidebar.multiselect(
        'Chọn Tháng:',
        options=sorted(df_full['Month'].unique()),
        default=sorted(df_full['Month'].unique()) # Mặc định chọn tất cả
    )

    # Áp dụng bộ lọc
    df_filtered = df_full.copy() # Tạo bản sao để không ảnh hưởng dữ liệu gốc
    if selected_city != 'Tất cả':
        df_filtered = df_filtered[df_filtered['Branch'] == selected_city]
    if selected_month:
        df_filtered = df_filtered[df_filtered['Month'].isin(selected_month)]
    else:
        # Nếu không chọn tháng nào, hiển thị thông báo
        st.warning("Vui lòng chọn ít nhất một tháng.")
        df_filtered = pd.DataFrame() # Trả về DataFrame rỗng

    # --- Hiển thị Kết quả Phân tích ---
    st.header(f'Kết quả cho: {selected_city} - Tháng: {", ".join(map(str, selected_month))}')

    if not df_filtered.empty:
        # 1. Các chỉ số KPI chính
        st.subheader('📈 Chỉ số Hiệu suất Chính (KPIs)')
        col1, col2, col3 = st.columns(3)
        total_revenue = int(df_filtered['Sales'].sum())
        avg_rating = round(df_filtered['Rating'].mean(), 1) if not df_filtered['Rating'].empty else 0
        avg_sale_value = round(df_filtered['Sales'].mean(), 2) if not df_filtered['Sales'].empty else 0

        col1.metric("Tổng Doanh Thu", f"${total_revenue:,.0f}")
        col2.metric("Đánh Giá TB", f"{avg_rating}/10 ⭐")
        col3.metric("Hóa Đơn TB", f"${avg_sale_value:,.2f}")

        st.markdown("---") # Đường kẻ ngang

        # 2. Layout 2 cột cho các biểu đồ chính
        fig_col1, fig_col2 = st.columns(2)

        with fig_col1:
            # Biểu đồ 1: Doanh thu theo Mặt hàng (Bar Chart)
            st.subheader('💰 Doanh thu theo Mặt hàng')
            sales_by_product = df_filtered.groupby('Product line')['Sales'].sum().sort_values(ascending=False)
            st.bar_chart(sales_by_product)

            # Biểu đồ 2: Cơ cấu Thanh toán (Pie Chart - Dùng Matplotlib)
            st.subheader('💳 Cơ cấu Hình thức Thanh toán')
            payment_counts = df_filtered['Payment'].value_counts()
            fig_pie, ax_pie = plt.subplots(figsize=(5, 5)) # Giảm kích thước
            ax_pie.pie(payment_counts, labels=payment_counts.index, autopct='%1.1f%%',
                       startangle=90, colors=sns.color_palette('pastel'), textprops={'fontsize': 8}) # Giảm cỡ chữ
            ax_pie.axis('equal')
            st.pyplot(fig_pie)

        with fig_col2:
            # Biểu đồ 3: Xu hướng Doanh thu theo Giờ (Line Chart)
            st.subheader('⏰ Doanh thu theo Giờ trong Ngày')
            sales_by_hour = df_filtered.groupby('Hour')['Sales'].sum()
            st.line_chart(sales_by_hour)

            # Biểu đồ 4: Tương quan Đánh giá & Doanh thu (Scatter Plot - Dùng Matplotlib/Seaborn)
            st.subheader('⭐ Tương quan Đánh giá & Doanh thu')
            fig_scatter, ax_scatter = plt.subplots(figsize=(6, 5)) # Giảm kích thước
            sns.scatterplot(data=df_filtered, x='Rating', y='Sales', hue='Gender', alpha=0.6, ax=ax_scatter)
            ax_scatter.tick_params(axis='both', which='major', labelsize=8) # Giảm cỡ chữ trục
            ax_scatter.xaxis.label.set_size(10) # Giảm cỡ chữ nhãn trục X
            ax_scatter.yaxis.label.set_size(10) # Giảm cỡ chữ nhãn trục Y
            plt.legend(fontsize='small') # Giảm cỡ chữ chú thích
            st.pyplot(fig_scatter)

        # --- PHẦN MỚI (CẬP NHẬT): TÍCH HỢP MACHINE LEARNING VỚI BIỂU ĐỒ TƯƠNG TÁC (PLOTLY) ---
        from sklearn.linear_model import LinearRegression
        import plotly.graph_objects as go # Thư viện vẽ biểu đồ tương tác

        st.markdown("---")
        st.header('🤖 Dự Báo Doanh Thu (Machine Learning)')
        st.write("Sử dụng thuật toán **Hồi quy Tuyến tính (Linear Regression)** để dự báo xu hướng doanh thu trong 30 ngày tới.")

        # 1. Chuẩn bị dữ liệu: Gom doanh thu theo ngày
        daily_sales = df_full.groupby('Date')['Sales'].sum().reset_index()

        # 2. Chuyển đổi ngày tháng sang dạng số (Ordinal) để mô hình hiểu được
        daily_sales['Date_Ordinal'] = daily_sales['Date'].map(pd.Timestamp.toordinal)

        # 3. Khởi tạo và Huấn luyện mô hình
        X = daily_sales[['Date_Ordinal']] # Dữ liệu đầu vào
        y = daily_sales['Sales']          # Dữ liệu mục tiêu

        model = LinearRegression()
        model.fit(X, y)

        # 4. Dự báo
        # Dự báo cho quá khứ (để vẽ đường xu hướng)
        trend_y = model.predict(X)

        # Dự báo cho 30 ngày tương lai
        last_date = daily_sales['Date'].max()
        future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 31)]
        future_ordinals = [[pd.Timestamp(d).toordinal()] for d in future_dates]
        future_sales = model.predict(future_ordinals)

        # 5. TRỰC QUAN HÓA TƯƠNG TÁC BẰNG PLOTLY
        fig = go.Figure()

        # Vẽ đường doanh thu thực tế (Quá khứ)
        fig.add_trace(go.Scatter(
            x=daily_sales['Date'], 
            y=daily_sales['Sales'],
            mode='lines',
            name='Doanh thu thực tế',
            line=dict(color='#636EFA'), # Màu xanh tím
            hovertemplate='Ngày: %{x|%d/%m/%Y}<br>Doanh thu: $%{y:,.0f}'
        ))

        # Vẽ đường xu hướng (Regression Line)
        fig.add_trace(go.Scatter(
            x=daily_sales['Date'], 
            y=trend_y,
            mode='lines',
            name='Đường xu hướng (Trendline)',
            line=dict(color='#EF553B', dash='dash'), # Màu đỏ, nét đứt
            hovertemplate='Xu hướng: $%{y:,.0f}'
        ))

        # Vẽ đường dự báo (Tương lai)
        fig.add_trace(go.Scatter(
            x=future_dates, 
            y=future_sales,
            mode='lines+markers',
            name='Dự báo 30 ngày tới',
            line=dict(color='#00CC96'), # Màu xanh lá
            marker=dict(size=6),
            hovertemplate='Ngày dự báo: %{x|%d/%m/%Y}<br>Doanh thu: $%{y:,.0f}'
        ))

        # Cấu hình giao diện biểu đồ
        fig.update_layout(
            title='Biểu đồ Dự báo Doanh thu Tương tác',
            xaxis_title='Thời gian',
            yaxis_title='Doanh thu (USD)',
            hovermode="x unified", # Hiển thị thông tin tất cả các đường khi di chuột
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )

        # Hiển thị lên Streamlit
        st.plotly_chart(fig, use_container_width=True)

        # Hiển thị độ chính xác
        r2_score = model.score(X, y)
        st.info(f"Độ phù hợp của mô hình (R-squared): {r2_score:.2%}")   

        st.markdown("---")

        # 3. Hiển thị dữ liệu chi tiết (có lọc)
        st.subheader('📄 Xem Dữ liệu Chi tiết (Đã lọc)')
        st.dataframe(df_filtered)

       

        

    else:
        st.info("Không có dữ liệu phù hợp với bộ lọc đã chọn.")

else:
    st.error("Không thể tải dữ liệu. Vui lòng kiểm tra lại file CSV.")