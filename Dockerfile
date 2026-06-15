# Menggunakan base image Python yang stabil dan ringan
FROM python:3.11-slim

# Mengatur variabel lingkungan agar Python tidak menulis file .pyc ke disk 
# dan tidak menyangga (buffer) stdout/stderr untuk log yang lebih real-time
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Mengatur direktori kerja di dalam kontainer
WORKDIR /app

# Menginstal dependensi sistem jika proyek membutuhkan library tambahan (misalnya untuk OpenCV/image processing jika ada)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Menyalin file requirements.txt terlebih dahulu agar bisa memanfaatkan caching Docker
COPY requirements.txt .

# Menginstal semua dependensi Python
RUN pip install --no-cache-dir -r requirements.txt

# Menyalin seluruh kode proyek ke dalam direktori kerja kontainer
COPY . .

RUN apt-get update && apt-get install -y curl && \
    curl -L -o MobileNetV2_RiceLeaf.onnx https://huggingface.co/roosyid66/rice-leaf-model/resolve/main/MobileNetV2_RiceLeaf.onnx

# Menentukan port yang akan digunakan di dalam kontainer (ganti jika Anda menggunakan port lain selain 8000)
EXPOSE 8000

# Perintah untuk menjalankan API menggunakan Uvicorn (Sesuaikan 'main:app' dengan file entrypoint Anda)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
