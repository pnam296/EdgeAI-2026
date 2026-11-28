/*
 * Web server inference tren ESP32-S3:
 *   GET  /           trang web
 *   POST /run        body = 320x240 RGB888 (230400 byte) + cau hoi (ASCII khong dau) -> JSON
 *   POST /run_cam    body = cau hoi; chup OV7670 -> detect + intent -> JSON
 *   GET  /frame.rgb  khung RGB888 vua dua vao model (web ve box len dung khung nay)
 *   GET  /snap.rgb   chup khung moi, khong chay model (thu thap anh: tools/capture.py)
 *   GET  /info       cau hinh dang chay
 *   GET  /cam?ae=96  doi muc sang tu dong phoi sang camera (48..208) luc dang chay
 *   GET  /wifi       trang doi WiFi;  POST /wifi (ssid=&pass=) luu vao NVS roi khoi dong lai
 *   GET  :81/stream  MJPEG camera live
 * WiFi: uu tien SSID/pass luu trong NVS (dat qua /wifi), khong co thi dung CONFIG_WIFI_*; ket noi loi
 * thi phat AP du phong CONFIG_AP_SSID. Ten mDNS: http://esp32s3-infer.local/
 * /run va /run_cam nhan them ?conf=&loc=&iou=&agn= de doi nguong khong can nap lai firmware
 * (conf ghi de nguong moi lop; bo trong = nguong rieng tung lop CLASS_CONF_INIT; loc = du phong LOCATION).
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
#include "nvs.h"
#include "esp_system.h"
#include "mdns.h"
#include "infer.h"
#include "cam.h"
#include "img_converters.h"   /* fmt2jpg (esp32-camera) */

#define MAX_Q 512
#define MAX_DETS 64

static const char *TAG = "main";
static EventGroupHandle_t s_ev;
#define BIT_GOT_IP BIT0
#define BIT_FAIL   BIT1
static int s_retry;
static bool s_ap_mode, s_connected_once;
static char s_ssid[33], s_pass[65];   /* WiFi dang dung (NVS hoac CONFIG_WIFI_*) */
static char s_ip[16] = "192.168.4.1";
#define MDNS_HOST "esp32s3-infer"

/* Doc SSID/pass tu NVS namespace "wifi"; chua luu thi dung gia tri luc build */
static void wifi_load_creds(void)
{
    strlcpy(s_ssid, CONFIG_WIFI_SSID, sizeof s_ssid);
    strlcpy(s_pass, CONFIG_WIFI_PASSWORD, sizeof s_pass);
    nvs_handle_t nh;
    if (nvs_open("wifi", NVS_READONLY, &nh) != ESP_OK) return;
    size_t n1 = sizeof s_ssid, n2 = sizeof s_pass;
    char ssid[33], pass[65];
    if (nvs_get_str(nh, "ssid", ssid, &n1) == ESP_OK && nvs_get_str(nh, "pass", pass, &n2) == ESP_OK) {
        strlcpy(s_ssid, ssid, sizeof s_ssid);
        strlcpy(s_pass, pass, sizeof s_pass);
        ESP_LOGI(TAG, "WiFi lay tu NVS: \"%s\"", s_ssid);
    }
    nvs_close(nh);
}

static esp_err_t wifi_save_creds(const char *ssid, const char *pass)
{
    nvs_handle_t nh;
    esp_err_t err = nvs_open("wifi", NVS_READWRITE, &nh);
    if (err != ESP_OK) return err;
    err = nvs_set_str(nh, "ssid", ssid);
    if (err == ESP_OK) err = nvs_set_str(nh, "pass", pass);
    if (err == ESP_OK) err = nvs_commit(nh);
    nvs_close(nh);
    return err;
}

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
        snprintf(s_ip, sizeof s_ip, IPSTR, IP2STR(&e->ip_info.ip));
        ESP_LOGI(TAG, "==> Mo trinh duyet: http://%s/  hoac  http://" MDNS_HOST ".local/", s_ip);
        xEventGroupSetBits(s_ev, BIT_GOT_IP);
    }
}

static void start_softap(void)
{
    s_ap_mode = true;
    strlcpy(s_ip, "192.168.4.1", sizeof s_ip);
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

    wifi_load_creds();
    if (strlen(s_ssid) == 0) {
        start_softap();
    } else {
        wifi_config_t sta = {0};
        strlcpy((char *)sta.sta.ssid, s_ssid, sizeof(sta.sta.ssid));
        strlcpy((char *)sta.sta.password, s_pass, sizeof(sta.sta.password));
        ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
        ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &sta));
        ESP_ERROR_CHECK(esp_wifi_start());
        ESP_LOGI(TAG, "Dang ket noi WiFi \"%s\"...", s_ssid);
        EventBits_t b = xEventGroupWaitBits(s_ev, BIT_GOT_IP | BIT_FAIL, pdFALSE, pdFALSE,
                                            pdMS_TO_TICKS(20000));
        if (!(b & BIT_GOT_IP)) {
            ESP_LOGW(TAG, "Khong ket noi duoc WiFi -> chuyen sang AP du phong");
            start_softap();
        }
    }
    esp_wifi_set_ps(WIFI_PS_NONE);

    /* http://esp32s3-infer.local/ : khoi phai tim IP khi ESP noi vao WiFi khac */
    if (mdns_init() == ESP_OK) {
        mdns_hostname_set(MDNS_HOST);
        mdns_instance_name_set("ESP32-S3 Infer");
        mdns_service_add(NULL, "_http", "_tcp", 80, NULL, 0);
    } else {
        ESP_LOGW(TAG, "mDNS khong khoi dong duoc (van vao bang IP)");
    }
}

/* ------------------------------------------------------------------ trang doi WiFi */
static int recv_all(httpd_req_t *req, uint8_t *dst, size_t len);

static int hexval(char c)
{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

/* giai ma application/x-www-form-urlencoded tai cho (%XX, '+' = space) */
static void url_decode(char *s)
{
    char *o = s;
    for (; *s; s++) {
        if (*s == '+') *o++ = ' ';
        else if (*s == '%' && hexval(s[1]) >= 0 && hexval(s[2]) >= 0) {
            *o++ = (char)(hexval(s[1]) * 16 + hexval(s[2]));
            s += 2;
        } else *o++ = *s;
    }
    *o = 0;
}

/* chep chuoi vao HTML, doi & < > " de khong vo trang */
static size_t html_esc(char *dst, size_t cap, const char *src)
{
    size_t o = 0;
    for (; *src && o + 7 < cap; src++) {
        const char *r = NULL;
        switch (*src) {
            case '&': r = "&amp;"; break;
            case '<': r = "&lt;"; break;
            case '>': r = "&gt;"; break;
            case '"': r = "&quot;"; break;
        }
        if (r) o += strlcpy(dst + o, r, cap - o);
        else dst[o++] = *src;
    }
    dst[o] = 0;
    return o;
}

static const char WIFI_PAGE_HEAD[] =
    "<!doctype html><html lang=vi><head><meta charset=utf-8>"
    "<meta name=viewport content='width=device-width,initial-scale=1'><title>WiFi ESP32-S3</title>"
    "<style>body{font:15px system-ui,sans-serif;max-width:420px;margin:24px auto;padding:0 16px;color:#1d2330}"
    "label{display:block;margin:12px 0 4px}input{width:100%;padding:8px;box-sizing:border-box;font:inherit}"
    "button{margin-top:16px;padding:9px 16px;font:inherit;border:0;border-radius:6px;background:#1f6feb;color:#fff}"
    ".b2{background:#667085}.m{color:#667085;font-size:13px}code{background:#eef0f3;padding:1px 4px}</style>"
    "</head><body><h2>WiFi cho ESP32-S3</h2>";

static esp_err_t wifi_get(httpd_req_t *req)
{
    char ssid_esc[200];
    html_esc(ssid_esc, sizeof ssid_esc, s_ssid);
    static char page[2048];
    snprintf(page, sizeof page,
             "%s<p>Dang: <b>%s</b><br>IP: <code>%s</code></p>"
             "<form method=post action=/wifi>"
             "<label>Ten WiFi (SSID)</label><input name=ssid maxlength=32 value=\"%s\" required>"
             "<label>Mat khau (de trong neu WiFi mo)</label><input name=pass type=password maxlength=64>"
             "<button>Luu va khoi dong lai</button></form>"
             "<form method=post action=/wifi><input type=hidden name=ssid value=''>"
             "<button class=b2>Xoa WiFi da luu (ve che do tu phat AP)</button></form>"
             "<p class=m>ESP chi dung WiFi 2.4 GHz. Noi khong duoc trong 20 giay thi ESP tu phat WiFi "
             "<b>" CONFIG_AP_SSID "</b> de ban vao lai trang nay o <code>http://192.168.4.1/wifi</code>.</p>"
             "<p><a href=/>&larr; Ve trang chinh</a></p></body></html>",
             WIFI_PAGE_HEAD,
             s_ap_mode ? "tu phat AP " CONFIG_AP_SSID : (s_connected_once ? "da noi WiFi" : "dang noi WiFi"),
             s_ip, s_ap_mode ? "" : ssid_esc);
    httpd_resp_set_type(req, "text/html; charset=utf-8");
    return httpd_resp_sendstr(req, page);
}

static void reboot_task(void *arg)
{
    vTaskDelay(pdMS_TO_TICKS(1500));
    esp_restart();
}

static esp_err_t wifi_post(httpd_req_t *req)
{
    char body[256], ssid[100] = "", pass[200] = "";
    if (req->content_len >= sizeof body)
        return httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "qua dai");
    if (recv_all(req, (uint8_t *)body, req->content_len)) return ESP_FAIL;
    body[req->content_len] = 0;
    httpd_query_key_value(body, "ssid", ssid, sizeof ssid);
    httpd_query_key_value(body, "pass", pass, sizeof pass);
    url_decode(ssid);
    url_decode(pass);
    if (strlen(ssid) > 32 || strlen(pass) > 64)
        return httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "SSID toi da 32, mat khau toi da 64 ky tu");
    if (pass[0] && strlen(pass) < 8)
        return httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "Mat khau WiFi phai tu 8 ky tu (hoac de trong)");
    if (wifi_save_creds(ssid, pass) != ESP_OK)
        return httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "Luu NVS loi");
    ESP_LOGW(TAG, "Da luu WiFi \"%s\" -> khoi dong lai", ssid);

    char ssid_esc[200];
    html_esc(ssid_esc, sizeof ssid_esc, ssid);
    static char page[1536];
    if (ssid[0])
        snprintf(page, sizeof page,
                 "%s<p>Da luu WiFi <b>%s</b>. ESP dang khoi dong lai.</p>"
                 "<p>Chuyen laptop sang cung WiFi do, doi khoang 15 giay roi mo "
                 "<code>http://" MDNS_HOST ".local/</code></p>"
                 "<p class=m>Khong mo duoc thi xem IP cua ESP trong danh sach thiet bi cua hotspot/router. "
                 "Sai mat khau thi sau 20 giay ESP tu phat lai WiFi <b>" CONFIG_AP_SSID "</b>.</p></body></html>",
                 WIFI_PAGE_HEAD, ssid_esc);
    else
        snprintf(page, sizeof page,
                 "%s<p>Da xoa WiFi. ESP khoi dong lai va tu phat WiFi <b>" CONFIG_AP_SSID "</b> "
                 "(mat khau <code>" CONFIG_AP_PASSWORD "</code>), vao <code>http://192.168.4.1/</code></p></body></html>",
                 WIFI_PAGE_HEAD);
    httpd_resp_set_type(req, "text/html; charset=utf-8");
    httpd_resp_sendstr(req, page);
    xTaskCreate(reboot_task, "reboot", 2048, NULL, 5, NULL);
    return ESP_OK;
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

/* ?conf=0.5&loc=0.1&iou=0.6&agn=1 -> ghi de nguong mac dinh cho lan chay nay.
 * conf ghi de nguong cua MOI lop (khong co conf = nguong rieng tung lop trong model_config.h). */
static void parse_params(httpd_req_t *req, det_params_t *p, answer_params_t *ap)
{
    infer_default_params(p);
    answer_default_params(ap);
    char qs[128], v[16];
    if (httpd_req_get_url_query_str(req, qs, sizeof qs) == ESP_OK) {
        if (httpd_query_key_value(qs, "conf", v, sizeof v) == ESP_OK && v[0]) {
            float c = strtof(v, NULL);
            if (c > 0.01f && c < 0.99f)   /* chan gia tri vo ly */
                for (int k = 0; k < NUM_CLASSES; k++) ap->cls_conf[k] = c;
        }
        if (httpd_query_key_value(qs, "loc", v, sizeof v) == ESP_OK && v[0]) {
            float l = strtof(v, NULL);
            if (l > 0.01f && l <= 1.0f) ap->loc_floor = l;
        }
        if (httpd_query_key_value(qs, "iou", v, sizeof v) == ESP_OK && v[0]) p->iou = strtof(v, NULL);
        if (httpd_query_key_value(qs, "agn", v, sizeof v) == ESP_OK && v[0]) p->agnostic = atoi(v);
    }
    if (!(p->iou > 0.05f && p->iou < 0.99f)) p->iou = 0.6f;
    p->conf = answer_floor(ap);   /* decode o nguong thap nhat, make_answer loc lai theo tung lop */
}

/* [0.500,0.450,...] nguong tung lop cho JSON */
static int conf_json(char *o, size_t sz, const answer_params_t *ap)
{
    int k = snprintf(o, sz, "[");
    for (int c = 0; c < NUM_CLASSES && k < (int)sz; c++)
        k += snprintf(o + k, sz - k, "%s%.3f", c ? "," : "", ap->cls_conf[c]);
    if (k < (int)sz) k += snprintf(o + k, sz - k, "]");
    return k;
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
static esp_err_t infer_and_reply(httpd_req_t *req, char *q, const det_params_t *p, const answer_params_t *ap,
                                 const char *src, int ms_cap)
{
    static det_t d[MAX_DETS];
    det_timing_t tm = {0};
    cam_pause_stream(true);
    int64_t t0 = esp_timer_get_time();
    int n = infer_detect(s_body, p, d, MAX_DETS, &tm);   /* o nguong san answer_floor(ap) */
    int64_t t1 = esp_timer_get_time();
    int intent = -1;
    char answer[384] = "";
    if (n >= 0 && strspn(q, " ") < strlen(q)) {
        intent = infer_intent(q);
        if (intent >= 0) intent = make_answer(intent, q, d, n, ap, answer, sizeof answer);
    }
    int64_t t2 = esp_timer_get_time();
    cam_pause_stream(false);
    if (n < 0) return httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "detection loi");
    n = filter_by_class_conf(d, n, ap, d);   /* web chi ve box du nguong lop (box du phong LOCATION khong ve) */

    int ms_det = (int)((t1 - t0) / 1000), ms_int = (int)((t2 - t1) / 1000);
    static char js[MAX_DETS * 90 + 1024];
    char cj[96];
    conf_json(cj, sizeof cj, ap);
    size_t o = snprintf(js, sizeof js,
                        "{\"src\":\"%s\",\"ms_cap\":%d,\"ms_det\":%d,\"ms_intent\":%d,\"ms_total\":%d,"
                        "\"ms_fill\":%d,\"ms_invoke\":%d,\"ms_decode\":%d,"
                        "\"conf\":%s,\"loc\":%.3f,\"iou\":%.3f,\"agn\":%d,\"dets\":[",
                        src, ms_cap, ms_det, ms_int, ms_det + ms_int, tm.ms_fill, tm.ms_invoke, tm.ms_decode,
                        cj, ap->loc_floor, p->iou, p->agnostic);
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
    answer_params_t ap;
    parse_params(req, &p, &ap);
    return infer_and_reply(req, s_q, &p, &ap, "upload", 0);
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
    answer_params_t ap;
    parse_params(req, &p, &ap);
    int64_t t0 = esp_timer_get_time();
    if (cam_capture_rgb888(s_body)) return httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "chup anh loi");
    int ms_cap = (int)((esp_timer_get_time() - t0) / 1000);
    s_have_frame = true;
    return infer_and_reply(req, s_q, &p, &ap, "camera", ms_cap);
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

/* GET /frame.jpg : nhu /frame.rgb nhung nen JPEG (~15 KB thay vi 230 KB) -> web hien anh nhanh qua WiFi yeu.
 * fmt2jpg coi PIXFORMAT_RGB888 la thu tu B,G,R (to_jpg.cpp dao kenh) -> dao s_body (R,G,B) sang BGR truoc. */
static esp_err_t frame_jpg_get(httpd_req_t *req)
{
    if (!s_have_frame) return httpd_resp_send_err(req, HTTPD_404_NOT_FOUND, "chua chay lan nao");
    static uint8_t *bgr;
    if (!bgr) bgr = heap_caps_malloc(IMG_BYTES, MALLOC_CAP_SPIRAM);
    if (!bgr) return httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "het bo nho");
    for (size_t i = 0; i < IMG_BYTES; i += 3) {
        bgr[i] = s_body[i + 2];
        bgr[i + 1] = s_body[i + 1];
        bgr[i + 2] = s_body[i];
    }
    uint8_t *jpg = NULL;
    size_t len = 0;
    if (!fmt2jpg(bgr, IMG_BYTES, IMG_W, IMG_H, PIXFORMAT_RGB888, 85, &jpg, &len))
        return httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "nen JPEG loi");
    httpd_resp_set_type(req, "image/jpeg");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    httpd_resp_set_hdr(req, "Cache-Control", "no-store");
    esp_err_t r = httpd_resp_send(req, (const char *)jpg, len);
    free(jpg);
    return r;
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
    answer_params_t ap;
    infer_default_params(&p);
    answer_default_params(&ap);
    char js[384], cj[96];
    conf_json(cj, sizeof cj, &ap);
    snprintf(js, sizeof js, "{\"det\":\"%s\",\"conf\":%s,\"loc\":%.3f,\"iou\":%.3f,\"agn\":%d,\"cam\":%s,\"ae\":%d,"
             "\"psram_free_kb\":%u}",
             infer_det_kind(), cj, ap.loc_floor, p.iou, p.agnostic, cam_ready() ? "true" : "false",
             cam_get_ae_target(), (unsigned)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024));
    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_sendstr(req, js);
}

/* GET /cam?ae=96 : doi muc sang tu dong phoi sang luc dang chay (mat khi khoi dong lai;
 * muon giu lau dai thi dat CONFIG_CAM_AE_TARGET trong menuconfig) */
static esp_err_t cam_get(httpd_req_t *req)
{
    char qs[32], v[8];
    if (!cam_ready()) return httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "Khong co camera");
    if (httpd_req_get_url_query_str(req, qs, sizeof qs) == ESP_OK &&
        httpd_query_key_value(qs, "ae", v, sizeof v) == ESP_OK && v[0]) {
        if (cam_set_ae_target(atoi(v)) != 0)
            return httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "ghi thanh ghi camera loi");
    }
    char js[64];
    snprintf(js, sizeof js, "{\"ae\":%d}", cam_get_ae_target());
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
    httpd_register_uri_handler(h, &(httpd_uri_t){.uri = "/frame.jpg", .method = HTTP_GET, .handler = frame_jpg_get});
    httpd_register_uri_handler(h, &(httpd_uri_t){.uri = "/snap.rgb", .method = HTTP_GET, .handler = snap_get});
    httpd_register_uri_handler(h, &(httpd_uri_t){.uri = "/info", .method = HTTP_GET, .handler = info_get});
    httpd_register_uri_handler(h, &(httpd_uri_t){.uri = "/wifi", .method = HTTP_GET, .handler = wifi_get});
    httpd_register_uri_handler(h, &(httpd_uri_t){.uri = "/wifi", .method = HTTP_POST, .handler = wifi_post});
    httpd_register_uri_handler(h, &(httpd_uri_t){.uri = "/cam", .method = HTTP_GET, .handler = cam_get});

    if (camera_web_start() != ESP_OK) ESP_LOGW(TAG, "Khong co camera -> web chi dung anh tai len");
}
