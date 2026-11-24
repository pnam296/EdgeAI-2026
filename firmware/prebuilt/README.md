# Firmware build sẵn (ESP32-S3)

Flash thẳng, không cần cài ESP-IDF.

| File | Địa chỉ |
|---|---|
| `bootloader.bin` | `0x0` |
| `partition-table.bin` | `0x8000` |
| `esp32s3_infer.bin` | `0x10000` |

**Board:** ESP32-S3 **N16R8** (flash 16MB, PSRAM Octal 8MB). Board khác (flash 4MB / PSRAM Quad) phải build lại
từ `firmware/` sau khi sửa `sdkconfig.defaults`.

## Cách flash (Windows)

1. Cài Python, rồi `pip install esptool`.
2. Cắm board, xem cổng COM trong Device Manager.
3. Bấm đúp `flash.bat`, nhập cổng COM.

Lệnh tay (mọi hệ điều hành):

```
python -m esptool --chip esp32s3 -p COM5 -b 460800 write-flash --flash-mode dio --flash-size 16MB --flash-freq 80m \
  0x0 bootloader.bin 0x8000 partition-table.bin 0x10000 esp32s3_infer.bin
```

Sau khi flash: board thử nối WiFi SSID `12345678` (cấu hình lúc build, mật khẩu trống) 5 lần; không được thì tự
phát WiFi `ESP32S3-INFER` (mật khẩu `12345678`) -> nối vào rồi mở http://192.168.4.1/.

Build ngày 2026-11-24 từ `firmware/` (model trong `firmware/main/models/`). Đây là bản build nên chưa được chạy thử
trên board thật (board của người build lỗi phần cứng khi nạp).
