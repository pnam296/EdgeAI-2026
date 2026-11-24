set(MODEL_SRCS models/det_model.cc models/textcnn_model.cc models/vocab.cc)
foreach(f ${MODEL_SRCS})
    if(NOT EXISTS ${CMAKE_CURRENT_LIST_DIR}/${f})
        message(FATAL_ERROR "Thieu main/${f} - xem main/models/README.md")
    endif()
endforeach()

idf_component_register(
    SRCS "main.c" "logic.c" "infer.cc" "cam.c" ${MODEL_SRCS}
    INCLUDE_DIRS "." "models"
    EMBED_FILES "index.html"
    REQUIRES esp_wifi esp_netif esp_event nvs_flash esp_http_server esp_timer
)
