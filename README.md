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
- [Hướng dẫn triển khai lên Ubuntu server](#hướng-dẫn-triển-khai-lên-ubuntu-server)
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
   - **Web demo**: [http://3.107.235.182:8000/](http://3.107.235.182:8000/)

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
   - **Web demo**: [http://3.107.235.182:8000/](http://3.107.235.182:8000/)

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

- **Demo giao diện web**:
  ![Web Demo](assets/demo_interface.png)

---

## Hướng dẫn triển khai lên Ubuntu server (IP: `3.107.209.218`) sử dụng Termius

### **1. Chuẩn bị trên máy local**
1. **Nén dự án**:
   ```bash
   cd d:\tai_lieu\ki_cuoi\DockerModel
   tar -czvf DockerModel.tar.gz .
   ```

2. **Mở Termius**:
   - Tạo kết nối mới đến server.
   - Nhập thông tin:
     ```
     Hostname: 3.107.209.218
     Username: ubuntu
     Authentication: Key (chọn file .pem hoặc .ppk)
     ```

---

### **2. Kết nối và thiết lập server qua Termius**
1. **Upload file dự án**:
   - Vào tab **SFTP** trong Termius.
   - Kéo thả file `DockerModel.tar.gz` vào thư mục `/home/ubuntu`.

2. **SSH vào server**:
   - Mở terminal trong Termius.
   - Giải nén dự án:
     ```bash
     cd /home/ubuntu
     tar -xzvf DockerModel.tar.gz
     cd DockerModel
     ```

---

### **3. Cài đặt Docker trên server**
Chạy lần lượt các lệnh sau trong Termius:
```bash
sudo apt update
sudo apt install -y docker.io
sudo systemctl enable --now docker
```

**Kiểm tra cài đặt**:
```bash
docker --version
# Output mong đợi: Docker version 24.0.5 hoặc tương tự
```

---

### **4. Cài đặt Docker Compose**
```bash
sudo apt install -y docker-compose-plugin
docker compose version
# Output mong đợi: Docker Compose version v2.20.2
```

---

### **5. Triển khai dự án**
1. **Build và chạy container**:
   ```bash
   docker compose up -d --build
   ```

2. **Kiểm tra container**:
   ```bash
   docker ps
   # Phải thấy container đang chạy với STATUS = Up
   ```

---

### **6. Mở cổng firewall**
```bash
sudo ufw allow 8000/tcp
sudo ufw enable
sudo ufw status
# Phải thấy 8000 ALLOW
```

---

### **7. Kiểm tra kết quả**
1. **Truy cập API**:
   Mở trình duyệt truy cập:
   ```
   http://3.107.209.218:8000/test
   ```

2. **Truy cập giao diện web**:
   ```
   http://3.107.209.218:8000
   ```

3. **Xem log real-time** (nếu cần):
   ```bash
   docker compose logs -f
   ```

---

### **8. Xử lý lỗi thường gặp**
| Lỗi | Giải pháp |
|------|----------|
| **Permission denied** | Chạy `sudo chmod 666 /var/run/docker.sock` (tạm thời) |
| **Port 8000 không mở** | Kiểm tra Security Group trên AWS Console |
| **Container không build** | Kiểm tra file `Dockerfile` và logs bằng `docker compose logs` |
| **Không kết nối được** | Kiểm tra IP public và key SSH |

---

### **9. Cập nhật sau này**
Khi có thay đổi code:
1. Upload file mới qua SFTP trong Termius.
2. Rebuild container:
   ```bash
   docker compose up -d --build
   ```

---

### **Lưu ý quan trọng**
1. Đảm bảo file model (`best.pt` và `resnet50_best_model_v05.pth`) đã có trong thư mục `app/model/`.
2. Nếu dùng GPU, cần cài thêm NVIDIA Container Toolkit:
   ```bash
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt update
   sudo apt install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```
3. Để dừng dịch vụ:
   ```bash
   docker compose down
   ```

4. Bạn có thể dùng tính năng **Port Forwarding** trong Termius để debug cổng 8000 về máy local nếu cần.

---

## Tác giả
- **👤 Tên**: Chiến Trần  
- **📧 Email**: [tcc3281@gmail.com](mailto:tcc3281@gmail.com)  
- **🌐 GitHub**: [GitHub Profile](https://github.com/tcc3281)

---

## License
This project is licensed under the [MIT License](LICENSE).