/*
 * Web server inference tren ESP32-S3:
 *   GET  /           trang web
 *   POST /run        body = 320x240 RGB888 (230400 byte) + cau hoi (ASCII khong dau) -> JSON
 *   POST /run_cam    body = cau hoi; chup OV7670 -> detect + intent -> JSON
 *   GET  /frame.rgb  khung RGB888 vua dua vao model (web ve box len dung khung nay)
 *   GET  /snap.rgb   chup khung moi, khong chay model (thu thap anh: tools/capture.py)
 *   GET  /info       cau hinh dang chay
 *   GET  :81/stream  MJPEG camera live
 * /run va /run_cam nhan them ?conf=&iou=&agn= de doi nguong khong can nap lai firmware.
 * JSON: {src, ms_cap, ms_det, ms_intent, ms_total(=det+intent), ms_invoke, dets:[{c,s,b:[x1,y1,x2,y2]}],
 *        q, intent, answer}
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_netif.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "esp_http_server.h"
#include "nvs_flash.h"
#include "infer.h"
#include "cam.h"

#define MAX_Q 512
#define MAX_DETS 64

static const char *TAG = "main";
static EventGroupHandle_t s_ev;
#define BIT_GOT_IP BIT0
#define BIT_FAIL   BIT1
static int s_retry;
static bool s_ap_mode, s_connected_once;

static void on_wifi(void *arg, esp_event_base_t base, int32_t id, void *data)
{
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        if (s_ap_mode) return;
        if (s_connected_once || s_retry < CONFIG_WIFI_MAX_RETRY) {
            s_retry++;
            ESP_LOGW(TAG, "Mat ket noi WiFi, thu lai (%d)", s_retry);
            esp_wifi_connect();
        } else {
            xEventGroupSetBits(s_ev, BIT_FAIL);
        }
    } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *e = data;
        s_retry = 0;
        s_connected_once = true;
        ESP_LOGI(TAG, "==> Mo trinh duyet: http://" IPSTR "/", IP2STR(&e->ip_info.ip));
        xEventGroupSetBits(s_ev, BIT_GOT_IP);
    }
}

static void start_softap(void)
{
    s_ap_mode = true;
    esp_wifi_stop();
    wifi_config_t ap = {0};
    strlcpy((char *)ap.ap.ssid, CONFIG_AP_SSID, sizeof(ap.ap.ssid));
    strlcpy((char *)ap.ap.password, CONFIG_AP_PASSWORD, sizeof(ap.ap.password));
    ap.ap.ssid_len = strlen(CONFIG_AP_SSID);
    ap.ap.channel = 6;
    ap.ap.max_connection = 4;
    ap.ap.authmode = strlen(CONFIG_AP_PASSWORD) >= 8 ? WIFI_AUTH_WPA2_PSK : WIFI_AUTH_OPEN;
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_AP));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_AP, &ap));
    ESP_ERROR_CHECK(esp_wifi_start());
    ESP_LOGW(TAG, "==> Che do AP: ket noi WiFi \"%s\" (pass \"%s\") roi mo http://192.168.4.1/",
             CONFIG_AP_SSID, CONFIG_AP_PASSWORD);
}

static void wifi_start(void)
{
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        nvs_flash_erase();
        err = nvs_flash_init();
    }
    ESP_ERROR_CHECK(err);
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();
    esp_netif_create_default_wifi_ap();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));
    s_ev = xEventGroupCreate();
    ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, on_wifi, NULL));
    ESP_ERROR_CHECK(esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP, on_wifi, NULL));

    if (strlen(CONFIG_WIFI_SSID) == 0) {
        start_softap();
    } else {
        wifi_config_t sta = {0};
        strlcpy((char *)sta.sta.ssid, CONFIG_WIFI_SSID, sizeof(sta.sta.ssid));
        strlcpy((char *)sta.sta.password, CONFIG_WIFI_PASSWORD, sizeof(sta.sta.password));
        ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
        ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &sta));
        ESP_ERROR_CHECK(esp_wifi_start());
        ESP_LOGI(TAG, "Dang ket noi WiFi \"%s\"...", CONFIG_WIFI_SSID);
        EventBits_t b = xEventGroupWaitBits(s_ev, BIT_GOT_IP | BIT_FAIL, pdFALSE, pdFALSE,
                                            pdMS_TO_TICKS(20000));
        if (!(b & BIT_GOT_IP)) {
            ESP_LOGW(TAG, "Khong ket noi duoc WiFi -> chuyen sang AP du phong");
            start_softap();
        }
    }
    esp_wifi_set_ps(WIFI_PS_NONE);
}

extern const char index_html[] asm("_binary_index_html_start");
extern const char index_html_end[] asm("_binary_index_html_end");
static uint8_t *s_body;   /* PSRAM: anh RGB888 (khung vua dung cho model) + cau hoi */
static bool s_have_frame;
static char s_q[MAX_Q + 1];

static esp_err_t index_get(httpd_req_t *req)
{
    httpd_resp_set_type(req, "text/html; charset=utf-8");
    return httpd_resp_send(req, index_html, index_html_end - index_html);
}

/* ?conf=0.5&iou=0.6&agn=1 -> ghi de nguong mac dinh cho lan chay nay */
static void parse_params(httpd_req_t *req, det_params_t *p)
{
    infer_default_params(p);
    char qs[96], v[16];
    if (httpd_req_get_url_query_str(req, qs, sizeof qs) != ESP_OK) return;
    if (httpd_query_key_value(qs, "conf", v, sizeof v) == ESP_OK && v[0]) p->conf = strtof(v, NULL);
    if (httpd_query_key_value(qs, "iou", v, sizeof v) == ESP_OK && v[0]) p->iou = strtof(v, NULL);
    if (httpd_query_key_value(qs, "agn", v, sizeof v) == ESP_OK && v[0]) p->agnostic = atoi(v);
    if (!(p->conf > 0.01f && p->conf < 0.99f)) p->conf = 0.5f;   /* chan gia tri vo ly */
    if (!(p->iou > 0.05f && p->iou < 0.99f)) p->iou = 0.6f;
}

static int recv_all(httpd_req_t *req, uint8_t *dst, size_t len)
{
    for (size_t got = 0; got < len;) {
        int r = httpd_req_recv(req, (char *)dst + got, len - got);
        if (r == HTTPD_SOCK_ERR_TIMEOUT) continue;
        if (r <= 0) return -1;
        got += r;
    }
    return 0;
}

/* Chay detect + intent tren s_body (anh) va q (cau hoi da normalize), tra JSON */
static esp_err_t infer_and_reply(httpd_req_t *req, char *q, const det_params_t *p, const char *src, int ms_cap)
{
    static det_t d[MAX_DETS];
    det_timing_t tm = {0};
    cam_pause_stream(true);
    int64_t t0 = esp_timer_get_time();
    int n = infer_detect(s_body, p, d, MAX_DETS, &tm);
    int64_t t1 = esp_timer_get_time();
    int intent = -1;
    char answer[384] = "";
    if (n >= 0 && strspn(q, " ") < strlen(q)) {
        intent = infer_intent(q);
        if (intent >= 0) make_answer(intent, q, d, n, answer, sizeof answer);
    }
    int64_t t2 = esp_timer_get_time();
    cam_pause_stream(false);
    if (n < 0) return httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "detection loi");

    int ms_det = (int)((t1 - t0) / 1000), ms_int = (int)((t2 - t1) / 1000);
    static char js[MAX_DETS * 90 + 1024];
    size_t o = snprintf(js, sizeof js,
                        "{\"src\":\"%s\",\"ms_cap\":%d,\"ms_det\":%d,\"ms_intent\":%d,\"ms_total\":%d,"
                        "\"ms_fill\":%d,\"ms_invoke\":%d,\"ms_decode\":%d,"
                        "\"conf\":%.3f,\"iou\":%.3f,\"agn\":%d,\"dets\":[",
                        src, ms_cap, ms_det, ms_int, ms_det + ms_int, tm.ms_fill, tm.ms_invoke, tm.ms_decode,
                        p->conf, p->iou, p->agnostic);
    for (int i = 0; i < n && o < sizeof js; i++)
        o += snprintf(js + o, sizeof js - o, "%s{\"c\":\"%s\",\"s\":%.3f,\"b\":[%.4f,%.4f,%.4f,%.4f]}",
                      i ? "," : "", CLASS_NAMES[d[i].cls], d[i].score, d[i].x1, d[i].y1, d[i].x2, d[i].y2);
    if (o < sizeof js)
        snprintf(js + o, sizeof js - o, "],\"q\":\"%s\",\"intent\":\"%s\",\"answer\":\"%s\"}", q,
                 intent >= 0 ? INTENT_NAMES[intent] : "", answer);   /* q/answer khong chua dau " hay \ */
    ESP_LOGI(TAG, "[%s] %d box | cap %d det %d (invoke %d) intent %d ms | %s | %s", src, n, ms_cap, ms_det,
             tm.ms_invoke, ms_int, intent >= 0 ? INTENT_NAMES[intent] : "-", answer);
    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_sendstr(req, js);
}

/* POST /run : body = 230400 byte RGB888 + cau hoi */
static esp_err_t run_post(httpd_req_t *req)
{
    size_t len = req->content_len;
    if (len < IMG_BYTES || len > IMG_BYTES + MAX_Q)
        return httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "body = 230400 byte RGB + cau hoi");
    if (recv_all(req, s_body, len)) return ESP_FAIL;
    s_have_frame = true;
    memcpy(s_q, s_body + IMG_BYTES, len - IMG_BYTES);
    s_q[len - IMG_BYTES] = 0;
    normalize_text(s_q);
    det_params_t p;
    parse_params(req, &p);
    return infer_and_reply(req, s_q, &p, "upload", 0);
}

/* POST /run_cam : body = cau hoi (co the rong). Chup camera -> detect + intent */
static esp_err_t run_cam_post(httpd_req_t *req)
{
    size_t len = req->content_len;
    if (len > MAX_Q) return httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "cau hoi qua dai");
    if (!cam_ready()) return httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "Khong co camera");
    if (recv_all(req, (uint8_t *)s_q, len)) return ESP_FAIL;
    s_q[len] = 0;
    normalize_text(s_q);
    det_params_t p;
    parse_params(req, &p);
    int64_t t0 = esp_timer_get_time();
    if (cam_capture_rgb888(s_body)) return httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "chup anh loi");
    int ms_cap = (int)((esp_timer_get_time() - t0) / 1000);
    s_have_frame = true;
    return infer_and_reply(req, s_q, &p, "camera", ms_cap);
}

static esp_err_t send_rgb(httpd_req_t *req)
{
    httpd_resp_set_type(req, "application/octet-stream");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    httpd_resp_set_hdr(req, "Cache-Control", "no-store");
    return httpd_resp_send(req, (const char *)s_body, IMG_BYTES);
}

/* GET /frame.rgb : khung RGB888 320x240 ma model vua chay (de ve box / luu lai anh) */
static esp_err_t frame_get(httpd_req_t *req)
{
    if (!s_have_frame) return httpd_resp_send_err(req, HTTPD_404_NOT_FOUND, "chua chay lan nao");
    return send_rgb(req);
}

/* GET /snap.rgb : chup khung moi, KHONG chay model (thu thap anh camera de fine-tune: tools/capture.py) */
static esp_err_t snap_get(httpd_req_t *req)
{
    if (cam_capture_rgb888(s_body)) return httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "Khong co camera");
    s_have_frame = true;
    return send_rgb(req);
}

/* GET /info : cau hinh dang chay (web doc de hien nguong mac dinh) */
static esp_err_t info_get(httpd_req_t *req)
{
    det_params_t p;
    infer_default_params(&p);
    char js[256];
    snprintf(js, sizeof js, "{\"det\":\"%s\",\"conf\":%.3f,\"iou\":%.3f,\"agn\":%d,\"cam\":%s,\"psram_free_kb\":%u}",
             infer_det_kind(), p.conf, p.iou, p.agnostic, cam_ready() ? "true" : "false",
             (unsigned)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024));
    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_sendstr(req, js);
}

void app_main(void)
{
    s_body = heap_caps_malloc(IMG_BYTES + MAX_Q + 1, MALLOC_CAP_SPIRAM);
    ESP_ERROR_CHECK(s_body ? ESP_OK : ESP_ERR_NO_MEM);
    if (infer_init() != 0) ESP_LOGE(TAG, "Nap model loi -> /run se khong dung duoc");
    wifi_start();

    httpd_handle_t h;
    httpd_config_t cfg = HTTPD_DEFAULT_CONFIG();
    cfg.stack_size = 16 * 1024;   /* Invoke() chay trong task httpd */
    cfg.max_uri_handlers = 12;
    ESP_ERROR_CHECK(httpd_start(&h, &cfg));
    httpd_register_uri_handler(h, &(httpd_uri_t){.uri = "/", .method = HTTP_GET, .handler = index_get});
    httpd_register_uri_handler(h, &(httpd_uri_t){.uri = "/run", .method = HTTP_POST, .handler = run_post});
    httpd_register_uri_handler(h, &(httpd_uri_t){.uri = "/run_cam", .method = HTTP_POST, .handler = run_cam_post});
    httpd_register_uri_handler(h, &(httpd_uri_t){.uri = "/frame.rgb", .method = HTTP_GET, .handler = frame_get});
    httpd_register_uri_handler(h, &(httpd_uri_t){.uri = "/snap.rgb", .method = HTTP_GET, .handler = snap_get});
    httpd_register_uri_handler(h, &(httpd_uri_t){.uri = "/info", .method = HTTP_GET, .handler = info_get});

    if (camera_web_start() != ESP_OK) ESP_LOGW(TAG, "Khong co camera -> web chi dung anh tai len");
}
