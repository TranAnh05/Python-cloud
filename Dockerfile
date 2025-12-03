# 1. Sử dụng image Python 3.10 slim
FROM python:3.10-slim

# 2. Thiết lập thư mục làm việc
WORKDIR /app

# 3. Copy file requirements trước
COPY requirements.txt ./

# 4. Cài đặt thư viện (ĐÃ SỬA ĐỔI ĐỂ KHẮC PHỤC LỖI)
# - Nâng cấp pip lên bản mới nhất trước (giúp tải ổn định hơn)
RUN pip install --upgrade pip

# - Cài đặt các thư viện với thời gian chờ (timeout) tăng lên 1000 giây
# - --default-timeout=1000: Giúp không bị lỗi khi mạng chậm
# - --no-cache-dir: Giữ image nhẹ
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# 5. Copy mã nguồn
COPY . .

# 6. Mở cổng
EXPOSE 8501

# 7. Chạy ứng dụng
CMD ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
