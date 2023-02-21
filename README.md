# Lab 01: Các mô hình máy học cho bài toán Regression

Sử dụng các dataset đã được cung cấp để áp dụng các mô hình: Linear Regression, SVR với Polynomial kernel, Random Forest. Tiến hành thí nghiệm và so sánh các phương pháp.

# Lab 02: Xử lý dữ liệu GIS

Bước 1: Cài đặt thư viện geopandas

Bước 2: git clone https://github.com/CityScope/CSL_HCMC

Bước 3: dùng geopandas để đọc shapefile trong /Data/GIS/Population/population_HCMC/population_shapefile/Population_Ward_Level.shp

Bước 4: hãy thực hiện các tác vụ truy vấn sau

- Phường nào có diện tích lớn nhất

- Phường nào có dân số 2019 (Pop_2019) cao nhất

- Phường nào có diện tích nhỏ nhất

- Phường nào có dân số thấp nhất (2019)
- Phường nào có tốc độ tăng trưởng dân số nhanh nhất (dựa trên Pop_2009 và Pop_2019)

- Phường nào có tốc độ tăng trưởng dân số thấp nhất

- Phường nào có biến động dân số nhanh nhất

- Phường nào có biến động dân số chậm nhất

- Phường nào có mật độ dân số cao nhất (2019)

- Phường nào có mật độ dân số thấp nhất (2019)

# Lab 03: Trực quan hoá dữ liệu bản đồ

Bước 1: Cài đặt geopandas và folium

Bước 2: git clone https://github.com/CityScope/CSL_HCMC

Bước 3: dùng geopandas để đọc shapefile trong /Data/GIS/Population/population_HCMC/population_shapefile/Population_District_Level.shp

Bước 4: hãy thực hiện vẽ ranh giới các quận lên bản đồ dựa theo hướng dẫn sau:
https://geopandas.readthedocs.io/en/latest/gallery/polygon_plotting_with_folium.html

# Lab 04: Gom cụm dữ liệu click của người dùng

Bước 1: Cài đặt các thư viện cần thiết: matplotlib==3.1.3, osmnet, folium, rtree, pygeos, geojson, geopandas

Bước 2: clone data từ https://github.com/CityScope/CSL_HCMC

Bước 3: Load ranh giới quận huyện và dân số quận huyện từ: Data\GIS\Population\population_HCMC\population_shapefile\Population_District_Level.shp

Bước 4: Load dữ liệu click của người dùng

Bước 5: Lọc ra 5 quận huyện có tốc độ tăng MẬT ĐỘ dân số nhanh nhất (Dùng dữ liệu 2019  và 2017)

Bước 6: Dùng spatial join (from geopandas.tools import sjoin) để lọc ra các điểm click của người dùng trong 5 quận/huyện hot nhất

Bước 7: chạy KMean cho top 5 quận huyện này. Lấy K = 20

Bước 8: Lưu 01 cụm điểm nhiều nhất trong các quận huyện ở Bước 5.

Bước 9: show lên bản đồ các cụm đông nhất theo từng quận huyện theo dạng HEATMAP

Bước 10: Lưu heatmap xuống file png

# Lab 05: Linear Regression + Evaluation + Streamlit

Các bạn nộp bài tập hồi quy tuyến tính sử dụng 2 phương pháp Train/Test split và K-fold cross validation để đánh giá mô hình.

Sử dụng Streamlit để làm giao diện ứng dụng theo gợi ý trên lớp lý thuyết.

# Lab 06: Phân lớp với Logistic Regression và đánh giá mô hình

# Lab 07: Classification với PCA để giảm số chiều

# Lab 08: Phân loại văn bản với Naive Bayes

# Lab 09: CNN và các biến thể

# Lab 10:  XGBoost

# Lab 11: Chọn lựa mô hình với Grid Search


