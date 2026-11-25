#pragma once
#include <stdbool.h>
#include <stdint.h>
#include "esp_err.h"

/* Khoi tao OV7670 (QVGA 320x240 RGB565 = dung kich thuoc input detection) va mo server
 * MJPEG http://<ip>:81/stream. Loi -> tra ve ma loi, phan con lai cua firmware van chay. */
esp_err_t camera_web_start(void);

bool cam_ready(void);

/* Chup 1 khung moi -> RGB888 320x240 (R,G,B theo hang) vao dst (IMG_BYTES).
 * An toan khi stream dang chay (dung chung mutex). 0 = ok. */
int cam_capture_rgb888(uint8_t *dst);

/* true: stream MJPEG tam nghi (khong ma hoa JPEG) de nhuong CPU cho model trong luc infer */
void cam_pause_stream(bool pause);

/* Muc sang muc tieu tu dong phoi sang (48..208, mac dinh CONFIG_CAM_AE_TARGET). 0 = ok. Mat khi khoi dong lai. */
int cam_set_ae_target(int target);
int cam_get_ae_target(void);
