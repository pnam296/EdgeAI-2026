/*
 * Camera OV7670 + MJPEG stream (lay tu ov7670_webcam). Chan:
 *   SDA=16, SCL=17, XCLK=12, PCLK=13, HREF=15, VSYNC=14, D0..D7=GPIO4..11, RST->3.3V, PWDN->GND
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_http_server.h"
#include "esp_camera.h"
#include "img_converters.h"
#include "sdkconfig.h"
#include "cam.h"

static const char *TAG = "cam";

static SemaphoreHandle_t s_mtx;          /* stream va cam_capture_rgb888 dung chung driver */
static volatile bool s_pause, s_ready;

#define CAM_PIN_PWDN   -1   /* PWDN noi GND */
#define CAM_PIN_RESET  -1   /* RST noi 3.3V */
#define CAM_PIN_XCLK   12
#define CAM_PIN_SIOD   16   /* SDA */
#define CAM_PIN_SIOC   17   /* SCL */
#define CAM_PIN_D0      4
#define CAM_PIN_D1      5
#define CAM_PIN_D2      6
#define CAM_PIN_D3      7
#define CAM_PIN_D4      8
#define CAM_PIN_D5      9
#define CAM_PIN_D6     10
#define CAM_PIN_D7     11
#define CAM_PIN_VSYNC  14
#define CAM_PIN_HREF   15
#define CAM_PIN_PCLK   13

#define JPEG_QUALITY_PERCENT  (100 - (CONFIG_CAM_JPEG_QUALITY * 100 / 63))

/* Thanh ghi chat luong anh OV7670 (gamma, AEC/AGC, AWB, ma tran mau) lay tu driver Linux
 * drivers/media/i2c/ov7670.c. Driver esp32-camera chi ghi bo toi gian -> anh chay sang, am tim.
 * Bo qua clock/cua so/scaling de khong doi FPS va khung hinh. */
static const uint8_t OV7670_QUALITY_REGS[][2] = {
    /* Gamma */
    {0x7a,0x20},{0x7b,0x10},{0x7c,0x1e},{0x7d,0x35},{0x7e,0x5a},{0x7f,0x69},{0x80,0x76},{0x81,0x80},
    {0x82,0x88},{0x83,0x8f},{0x84,0x96},{0x85,0xa3},{0x86,0xaf},{0x87,0xc4},{0x88,0xd7},{0x89,0xe8},
    /* AGC/AEC: tat truoc khi chinh. AEC kieu trung binh (0xaa=0x14, Linux dung histogram 0x94 bo qua AEW/AEB), muc sang AEW/AEB/VPT */
    {0x13,0xe0},{0x00,0x00},{0x10,0x00},{0x0d,0x40},{0x14,0x18},{0xa5,0x05},{0xab,0x07},
    {0x24,0xa0},{0x25,0x90},{0x26,0xe5},{0x9f,0x78},{0xa0,0x68},{0xa1,0x03},{0xa6,0xd8},
    {0xa7,0xd8},{0xa8,0xf0},{0xa9,0x90},{0xaa,0x14},{0x13,0xe5},
    /* Reserved "magic" */
    {0x0e,0x61},{0x0f,0x4b},{0x16,0x02},{0x1e,0x07},{0x21,0x02},{0x22,0x91},{0x29,0x07},{0x33,0x0b},
    {0x35,0x0b},{0x37,0x1d},{0x38,0x71},{0x39,0x2a},{0x3c,0x78},{0x4d,0x40},{0x4e,0x20},{0x69,0x00},
    {0x74,0x10},{0x8d,0x4f},{0x8e,0x00},{0x8f,0x00},{0x90,0x00},{0x91,0x00},{0x96,0x00},{0x9a,0x00},
    {0xb0,0x84},{0xb1,0x0c},{0xb2,0x0e},{0xb3,0x82},{0xb8,0x0a},
    /* Can bang trang (AWB) */
    {0x43,0x0a},{0x44,0xf0},{0x45,0x34},{0x46,0x58},{0x47,0x28},{0x48,0x3a},{0x59,0x88},{0x5a,0x88},
    {0x5b,0x44},{0x5c,0x67},{0x5d,0x49},{0x5e,0x0e},{0x6c,0x0a},{0x6d,0x55},{0x6e,0x11},{0x6f,0x9f},
    {0x6a,0x40},{0x01,0x40},{0x02,0x60},{0x13,0xe7},
    /* Ma tran, edge, denoise, banding */
    {0x58,0x9e},{0x41,0x08},{0x3f,0x00},{0x75,0x05},{0x76,0xe1},{0x4c,0x00},{0x77,0x01},{0x4b,0x09},
    {0xc9,0x60},{0x41,0x38},{0x56,0x40},{0x34,0x11},{0x3b,0x12},{0xa4,0x88},{0x96,0x00},{0x97,0x30},
    {0x98,0x20},{0x99,0x30},{0x9a,0x84},{0x9b,0x29},{0x9c,0x03},{0x9d,0x4c},{0x9e,0x3f},{0x78,0x04},
    {0x79,0x01},{0xc8,0xf0},{0x79,0x0f},{0xc8,0x00},{0x79,0x10},{0xc8,0x7e},{0x79,0x0a},{0xc8,0x80},
    {0x79,0x0b},{0xc8,0x01},{0x79,0x0c},{0xc8,0x0f},{0x79,0x0d},{0xc8,0x20},{0x79,0x09},{0xc8,0x80},
    {0x79,0x02},{0xc8,0xc0},{0x79,0x03},{0xc8,0x40},{0x79,0x05},{0xc8,0x30},{0x79,0x26},
    {0x14,0x38},{0x4f,0xb3},{0x50,0xb3},{0x51,0x00},{0x52,0x3d},{0x53,0xa7},{0x54,0xe4},{0x3d,0xc0},
};

static esp_err_t camera_init(void)
{
    camera_config_t cfg = {0};
    cfg.ledc_channel = LEDC_CHANNEL_0;
    cfg.ledc_timer   = LEDC_TIMER_0;
    cfg.pin_d0 = CAM_PIN_D0;  cfg.pin_d1 = CAM_PIN_D1;
    cfg.pin_d2 = CAM_PIN_D2;  cfg.pin_d3 = CAM_PIN_D3;
    cfg.pin_d4 = CAM_PIN_D4;  cfg.pin_d5 = CAM_PIN_D5;
    cfg.pin_d6 = CAM_PIN_D6;  cfg.pin_d7 = CAM_PIN_D7;
    cfg.pin_xclk  = CAM_PIN_XCLK;
    cfg.pin_pclk  = CAM_PIN_PCLK;
    cfg.pin_vsync = CAM_PIN_VSYNC;
    cfg.pin_href  = CAM_PIN_HREF;
    cfg.pin_sccb_sda = CAM_PIN_SIOD;
    cfg.pin_sccb_scl = CAM_PIN_SIOC;
    cfg.sccb_i2c_port = 0;
    cfg.pin_pwdn  = CAM_PIN_PWDN;
    cfg.pin_reset = CAM_PIN_RESET;
    cfg.xclk_freq_hz = CONFIG_CAM_XCLK_MHZ * 1000000;
    cfg.pixel_format = PIXFORMAT_RGB565;
    cfg.frame_size   = FRAMESIZE_QVGA;      /* 320x240 = input model detection */
    cfg.jpeg_quality = CONFIG_CAM_JPEG_QUALITY;
    cfg.fb_count     = 2;                   /* firmware nay bat buoc PSRAM (arena detection) */
    cfg.fb_location  = CAMERA_FB_IN_PSRAM;
    cfg.grab_mode    = CAMERA_GRAB_LATEST;

    esp_err_t err = esp_camera_init(&cfg);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "esp_camera_init loi 0x%x (%s)", err, esp_err_to_name(err));
        return err;
    }

    sensor_t *s = esp_camera_sensor_get();
    if (s) {
        ESP_LOGI(TAG, "Sensor PID=0x%02x VER=0x%02x", s->id.PID, s->id.VER);
        /* Driver ov7670 doc CLKRC qua SCCB bi loi (tra ve 0xa1) roi ghi lai -> clock chia 34, ~1 FPS.
         * shortcut: ghi de CLKRC sau init, bo khi driver esp32-camera sua loi doc SCCB. */
        extern int SCCB_Write(uint8_t, uint8_t, uint8_t);
        SCCB_Write(s->slv_addr, 0x11, CONFIG_CAM_CLKRC);
        for (size_t i = 0; i < sizeof(OV7670_QUALITY_REGS) / sizeof(OV7670_QUALITY_REGS[0]); i++) {
            if (SCCB_Write(s->slv_addr, OV7670_QUALITY_REGS[i][0], OV7670_QUALITY_REGS[i][1]) != 0) {
                ESP_LOGW(TAG, "Ghi thanh ghi OV7670 0x%02x loi", OV7670_QUALITY_REGS[i][0]);
            }
        }
        /* Cua so anh QVGA theo driver Linux (driver esp32-camera de ~8 cot rac o mep trai) */
        int hs = 172, he = 28, vs = 12, ve = 492;
        SCCB_Write(s->slv_addr, 0x17, hs >> 3);                                  /* HSTART */
        SCCB_Write(s->slv_addr, 0x18, he >> 3);                                  /* HSTOP */
        SCCB_Write(s->slv_addr, 0x32, 0x80 | ((he & 7) << 3) | (hs & 7));        /* HREF */
        SCCB_Write(s->slv_addr, 0x19, vs >> 2);                                  /* VSTRT */
        SCCB_Write(s->slv_addr, 0x1a, ve >> 2);                                  /* VSTOP */
        SCCB_Write(s->slv_addr, 0x03, ((ve & 3) << 2) | (vs & 3));               /* VREF */
        /* Lat/guong anh theo cach gan camera tren kit (menuconfig). MVFP 0x1E: bit5 mirror, bit4 vflip */
        SCCB_Write(s->slv_addr, 0x1e, (CONFIG_CAM_HMIRROR ? 0x20 : 0) | (CONFIG_CAM_VFLIP ? 0x10 : 0));
    }
    return ESP_OK;
}

#define STREAM_BOUNDARY "frameboundary1234567890"
static const char *STREAM_CT   = "multipart/x-mixed-replace;boundary=" STREAM_BOUNDARY;
static const char *STREAM_SEP  = "\r\n--" STREAM_BOUNDARY "\r\n";
static const char *STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

static esp_err_t stream_handler(httpd_req_t *req)
{
    esp_err_t res = httpd_resp_set_type(req, STREAM_CT);
    if (res != ESP_OK) return res;
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");

    /* Thong ke FPS: cong don thoi gian tung buoc, in log moi giay */
    int frames = 0;
    int64_t t_cap = 0, t_enc = 0, t_send = 0, bytes = 0;
    int64_t t_win = esp_timer_get_time();

    while (true) {
        if (s_pause) {   /* model dang chay: khong tranh CPU/PSRAM */
            vTaskDelay(pdMS_TO_TICKS(20));
            t_win = esp_timer_get_time();
            frames = 0; t_cap = t_enc = t_send = bytes = 0;
            continue;
        }
        int64_t t0 = esp_timer_get_time();
        xSemaphoreTake(s_mtx, portMAX_DELAY);
        camera_fb_t *fb = esp_camera_fb_get();
        if (!fb) {
            xSemaphoreGive(s_mtx);
            ESP_LOGE(TAG, "Camera capture failed");
            res = ESP_FAIL;
            break;
        }
        int64_t t1 = esp_timer_get_time();
        uint8_t *jpg = NULL;
        size_t len = 0;
        bool ok = frame2jpg(fb, JPEG_QUALITY_PERCENT, &jpg, &len);
        esp_camera_fb_return(fb);
        xSemaphoreGive(s_mtx);
        if (!ok) {
            ESP_LOGE(TAG, "JPEG compression failed");
            res = ESP_FAIL;
            break;
        }
        int64_t t2 = esp_timer_get_time();

        char hdr[80];
        size_t hlen = snprintf(hdr, sizeof(hdr), STREAM_PART, (unsigned)len);

        res = httpd_resp_send_chunk(req, STREAM_SEP, strlen(STREAM_SEP));
        if (res == ESP_OK) res = httpd_resp_send_chunk(req, hdr, hlen);
        if (res == ESP_OK) res = httpd_resp_send_chunk(req, (const char *)jpg, len);
        free(jpg);
        if (res != ESP_OK) break;
        int64_t t3 = esp_timer_get_time();

        frames++;
        t_cap += t1 - t0; t_enc += t2 - t1; t_send += t3 - t2; bytes += len;
        if (t3 - t_win >= 1000000) {
            ESP_LOGI(TAG, "FPS %.1f | cap %lldms enc %lldms send %lldms | %lldKB/frame",
                     frames * 1e6 / (t3 - t_win), t_cap / frames / 1000, t_enc / frames / 1000,
                     t_send / frames / 1000, bytes / frames / 1024);
            frames = 0; t_cap = t_enc = t_send = bytes = 0; t_win = t3;
        }
    }
    return res;
}

bool cam_ready(void) { return s_ready; }

void cam_pause_stream(bool pause) { s_pause = pause; }

int cam_capture_rgb888(uint8_t *dst)
{
    if (!s_ready) return -1;
    xSemaphoreTake(s_mtx, portMAX_DELAY);
    /* GRAB_LATEST + 2 buffer: khung dau co the la khung cu (chup truoc khi bam Run) -> bo, lay khung tiep */
    for (int i = 0; i < CONFIG_CAM_DROP_FRAMES; i++) {
        camera_fb_t *old = esp_camera_fb_get();
        if (old) esp_camera_fb_return(old);
    }
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) { xSemaphoreGive(s_mtx); ESP_LOGE(TAG, "capture loi"); return -1; }
    if (fb->format != PIXFORMAT_RGB565 || fb->width != 320 || fb->height != 240) {
        ESP_LOGE(TAG, "khung %ux%u format %d, can 320x240 RGB565", fb->width, fb->height, fb->format);
        esp_camera_fb_return(fb);
        xSemaphoreGive(s_mtx);
        return -1;
    }
    const uint8_t *s = fb->buf;
    for (int i = 0; i < 320 * 240; i++, s += 2, dst += 3) {
        /* RGB565 big-endian (byte cao truoc): RRRRRGGG GGGBBBBB. Sai mau (do <-> xanh) -> bat CAM_SWAP_BYTES */
        uint16_t v = CONFIG_CAM_SWAP_BYTES ? (uint16_t)(s[1] << 8 | s[0]) : (uint16_t)(s[0] << 8 | s[1]);
        uint8_t r = (v >> 11) & 0x1f, g = (v >> 5) & 0x3f, b = v & 0x1f;
        dst[0] = (r << 3) | (r >> 2);   /* mo rong 5/6 bit -> 8 bit, 31 -> 255 */
        dst[1] = (g << 2) | (g >> 4);
        dst[2] = (b << 3) | (b >> 2);
    }
    esp_camera_fb_return(fb);
    xSemaphoreGive(s_mtx);
    return 0;
}

esp_err_t camera_web_start(void)
{
    if (!s_mtx) s_mtx = xSemaphoreCreateMutex();
    esp_err_t err = camera_init();
    if (err != ESP_OK) return err;
    s_ready = true;

    /* Server rieng cong 81: stream chay vo han, khong chan trang chinh / /run o cong 80 */
    httpd_config_t cfg = HTTPD_DEFAULT_CONFIG();
    cfg.server_port = 81;
    cfg.ctrl_port = 32769;
    cfg.stack_size = 8192;
    cfg.max_open_sockets = 3;
    cfg.lru_purge_enable = true;
    httpd_handle_t h;
    err = httpd_start(&h, &cfg);
    if (err != ESP_OK) return err;
    httpd_register_uri_handler(h, &(httpd_uri_t){.uri = "/stream", .method = HTTP_GET, .handler = stream_handler});
    ESP_LOGI(TAG, "Camera stream: http://<ip>:81/stream");
    return ESP_OK;
}
