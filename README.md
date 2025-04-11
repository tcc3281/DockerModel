# DockerModel - Traffic Sign Detection and Classification

DockerModel là một ứng dụng sử dụng YOLOv8 để phát hiện biển báo giao thông và ResNet50 để phân loại hình ảnh. Dự án được triển khai bằng FastAPI và có thể chạy trong môi trường Docker.

## Mục lục
- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Hướng dẫn sử dụng](#hướng-dẫn-sử-dụng)
  - [Chạy bằng Docker](#chạy-bằng-docker)
  - [Chạy cục bộ](#chạy-cục-bộ)
- [API Endpoints](#api-endpoints)
- [Giao diện Web](#giao-diện-web)
- [Tác giả](#tác-giả)
- [License](#license)

---

## Yêu cầu hệ thống
- **Docker**: Phiên bản mới nhất.
- **Python**: Phiên bản 3.10 (nếu chạy cục bộ).
- **Thư viện**: Được liệt kê trong `requirements.txt`.

---

## Cấu trúc dự án
```
DockerModel/
│
├── app/
│   ├── app.py               # Mã nguồn chính của ứng dụng
│   ├── requirements.txt     # Danh sách các thư viện cần thiết
│   ├── index.html           # Giao diện web
│   ├── class_mapping.txt    # File ánh xạ nhãn
│   └── model/               # Thư mục chứa các file model
│       ├── best.pt          # Model YOLOv8
│       └── resnet50_best_model_v05.pth # Model ResNet50
│
├── Dockerfile               # File cấu hình Docker
├── docker-compose.yml       # File cấu hình Docker Compose
├── .dockerignore            # File loại trừ khi build Docker
└── README.md                # Hướng dẫn sử dụng
```

---

## Hướng dẫn sử dụng

### Chạy bằng Docker
1. **Build Docker Image**:
   ```bash
   docker-compose build
   ```

2. **Chạy Container**:
   ```bash
   docker-compose up
   ```

3. **Truy cập API và giao diện web**:
   - API: [http://localhost:8000/test](http://localhost:8000/test)
   - Giao diện web: [http://localhost:8000](http://localhost:8000)

---

### Chạy cục bộ
1. **Cài đặt môi trường ảo**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Trên Linux/MacOS
   .venv\Scripts\activate     # Trên Windows
   ```

2. **Cài đặt thư viện**:
   ```bash
   pip install -r app/requirements.txt
   ```

3. **Chạy ứng dụng**:
   ```bash
   python app/app.py
   ```

4. **Truy cập API và giao diện web**:
   - API: [http://localhost:8000/test](http://localhost:8000/test)
   - Giao diện web: [http://localhost:8000](http://localhost:8000)

---

## API Endpoints
| Endpoint                | Phương thức | Mô tả                                      |
|-------------------------|-------------|--------------------------------------------|
| `/test`                 | GET         | Kiểm tra trạng thái API                   |
| `/detect`               | POST        | Phát hiện đối tượng trong ảnh             |
| `/classify`             | POST        | Phân loại hình ảnh                        |
| `/detect_and_classify`  | POST        | Phát hiện và phân loại đối tượng trong ảnh|

---

## Giao diện Web
- Giao diện web cho phép người dùng tải lên hình ảnh và chọn chế độ:
  - **Object Detection**: Phát hiện đối tượng.
  - **Image Classification**: Phân loại hình ảnh.
  - **Detect & Classify**: Kết hợp phát hiện và phân loại.

---

## Tác giả
- **👤 Tên**: Chiến Trần  
- **📧 Email**: [tcc3281@gmail.com](mailto:tcc3281@gmail.com)  
- **🌐 GitHub**: [GitHub Profile](https://github.com/tcc3281)

---

## License
This project is licensed under the [MIT License](LICENSE).