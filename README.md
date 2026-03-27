# NumboRecordnation

Clone về máy
```
git clone https://github.com/TgX-2/NumboRecordnation.git
```

# Cách sử dụng dự đoán thuần

## 1. Đảm bảo đã có các thư viện cần thiết
```
pip install tensorflow pillow numpy matplotlib
```
**Chú ý:** Dùng python 3.10.x (vì tensorflow méo có trên các phiên bản cao hơn)

## 2. Ghi số vào trong file test.png
## 3. Chạy code thôi
```
python digit_recognizer.py
```

# Cách sừ dụng dự đoán đóng gói thành app

## 1. Tải thêm thư viện
```
pip install scipy
```
## 2. Train
Chạy code `train_tf.py`. Sau đó sẽ thấy một file đuôi `.h5`.

## 3. Chạy GUI
Chạy code `app_tf.py` thôi. Sau đó viết rồi dự đón thôi.
