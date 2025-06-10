#include "gpu_pipeline.h"
#include "cuda_edge_detector.h"
#include <iostream>
#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>
#include <cuda_runtime.h>

GPUPipeline::GPUPipeline() 
    : pipeline(nullptr), source(nullptr), decoder(nullptr), 
      videoconvert(nullptr), appsink(nullptr), appsrc(nullptr),
      encoder(nullptr), filesink(nullptr), bus(nullptr), loop(nullptr),
      edge_detector(nullptr), is_initialized(false), processing_active(false) {
}

GPUPipeline::~GPUPipeline() {
    cleanup();
}

bool GPUPipeline::initialize() {
    std::cout << "Initializing GPU Pipeline..." << std::endl;
    
    // GStreamer elementlerini oluştur
    pipeline = gst_pipeline_new("gpu-pipeline");
    if (!pipeline) {
        std::cerr << "Failed to create pipeline" << std::endl;
        return false;
    }
    
    // Jetson için özelleştirilmiş decoder
    if (!setup_jetson_decoder()) {
        std::cerr << "Failed to setup Jetson decoder" << std::endl;
        return false;
    }
    
    // Jetson için özelleştirilmiş encoder
    if (!setup_jetson_encoder()) {
        std::cerr << "Failed to setup Jetson encoder" << std::endl;
        return false;
    }
    
    // NVMM memory konfigürasyonu
    if (!configure_nvmm_memory()) {
        std::cerr << "Failed to configure NVMM memory" << std::endl;
        return false;
    }
    
    // Pipeline'ı bağla
    if (!link_elements()) {
        std::cerr << "Failed to link pipeline elements" << std::endl;
        return false;
    }
    
    // Bus callback'i ayarla
    bus = gst_element_get_bus(pipeline);
    gst_bus_add_watch(bus, bus_callback, this);
    
    // Main loop oluştur
    loop = g_main_loop_new(nullptr, FALSE);
    
    is_initialized = true;
    std::cout << "GPU Pipeline initialized successfully" << std::endl;
    return true;
}

bool GPUPipeline::setup_jetson_decoder() {
    // Jetson hardware decoder elementleri
    source = gst_element_factory_make("filesrc", "file-source");
    decoder = gst_element_factory_make("nvv4l2decoder", "nvv4l2-decoder");
    videoconvert = gst_element_factory_make("nvvidconv", "nvvidconv");
    appsink = gst_element_factory_make("appsink", "app-sink");
    
    if (!source || !decoder || !videoconvert || !appsink) {
        std::cerr << "Failed to create decoder elements" << std::endl;
        
        // Fallback to software decoder if Jetson elements not available
        std::cout << "Fallback to software decoder..." << std::endl;
        if (decoder) gst_object_unref(decoder);
        if (videoconvert) gst_object_unref(videoconvert);
        
        decoder = gst_element_factory_make("avdec_h264", "sw-decoder");
        videoconvert = gst_element_factory_make("videoconvert", "sw-convert");
        
        if (!decoder || !videoconvert) {
            std::cerr << "Failed to create fallback decoder elements" << std::endl;
            return false;
        }
    }
    
    // Properties ayarla
    if (!input_file.empty()) {
        g_object_set(source, "location", input_file.c_str(), nullptr);
    }
    
    // AppSink properties
    g_object_set(appsink, 
                 "emit-signals", TRUE,
                 "sync", FALSE,
                 "max-buffers", 1,
                 "drop", TRUE,
                 nullptr);
    
    // AppSink callback
    g_signal_connect(appsink, "new-sample", G_CALLBACK(new_sample_callback), this);
    
    // Pipeline'a ekle
    gst_bin_add_many(GST_BIN(pipeline), source, decoder, videoconvert, appsink, nullptr);
    
    return true;
}

bool GPUPipeline::setup_jetson_encoder() {
    // Jetson hardware encoder elementleri
    appsrc = gst_element_factory_make("appsrc", "app-source");
    encoder = gst_element_factory_make("nvv4l2h264enc", "nvv4l2-h264enc");
    filesink = gst_element_factory_make("filesink", "file-sink");
    
    if (!appsrc || !encoder || !filesink) {
        std::cerr << "Failed to create encoder elements" << std::endl;
        
        // Fallback to software encoder
        std::cout << "Fallback to software encoder..." << std::endl;
        if (encoder) gst_object_unref(encoder);
        
        encoder = gst_element_factory_make("x264enc", "sw-encoder");
        if (!encoder) {
            std::cerr << "Failed to create fallback encoder" << std::endl;
            return false;
        }
        
        // x264enc properties
        g_object_set(encoder,
                     "bitrate", 8000,
                     "speed-preset", 1, // ultrafast
                     nullptr);
    } else {
        // nvv4l2h264enc properties
        g_object_set(encoder,
                     "bitrate", 8000000,
                     "preset-level", 1, // UltraFastPreset
                     "insert-sps-pps", TRUE,
                     nullptr);
    }
    
    // AppSrc properties
    g_object_set(appsrc,
                 "caps", gst_caps_from_string("video/x-raw,format=I420,width=1920,height=1080,framerate=60/1"),
                 "format", GST_FORMAT_TIME,
                 "is-live", TRUE,
                 nullptr);
    
    // FileSink properties
    if (!output_file.empty()) {
        g_object_set(filesink, "location", output_file.c_str(), nullptr);
    }
    
    // Pipeline'a ekle
    gst_bin_add_many(GST_BIN(pipeline), appsrc, encoder, filesink, nullptr);
    
    return true;
}

bool GPUPipeline::configure_nvmm_memory() {
    // Jetson'da NVMM memory için caps ayarla
    GstCaps *nvmm_caps = gst_caps_from_string("video/x-raw(memory:NVMM),format=I420,width=1920,height=1080,framerate=60/1");
    
    if (nvmm_caps) {
        // videoconvert ile appsink arasına caps filter ekle
        GstElement *capsfilter = gst_element_factory_make("capsfilter", "nvmm-caps");
        if (capsfilter) {
            g_object_set(capsfilter, "caps", nvmm_caps, nullptr);
            gst_bin_add(GST_BIN(pipeline), capsfilter);
        }
        gst_caps_unref(nvmm_caps);
    }
    
    return true;
}

bool GPUPipeline::link_elements() {
    // Decode pipeline'ını bağla
    if (!gst_element_link_many(source, decoder, videoconvert, appsink, nullptr)) {
        std::cerr << "Failed to link decode pipeline" << std::endl;
        return false;
    }
    
    // Encode pipeline'ını bağla
    if (!gst_element_link_many(appsrc, encoder, filesink, nullptr)) {
        std::cerr << "Failed to link encode pipeline" << std::endl;
        return false;
    }
    
    return true;
}

void GPUPipeline::set_input_file(const std::string& file) {
    input_file = file;
    if (source) {
        g_object_set(source, "location", file.c_str(), nullptr);
    }
}

void GPUPipeline::set_output_file(const std::string& file) {
    output_file = file;
    if (filesink) {
        g_object_set(filesink, "location", file.c_str(), nullptr);
    }
}

void GPUPipeline::set_edge_detector(CudaEdgeDetector* detector) {
    edge_detector = detector;
}

bool GPUPipeline::process() {
    if (!is_initialized) {
        std::cerr << "Pipeline not initialized" << std::endl;
        return false;
    }
    
    std::cout << "Starting GPU pipeline processing..." << std::endl;
    processing_active = true;
    
    // Pipeline'ı başlat
    GstStateChangeReturn ret = gst_element_set_state(pipeline, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        std::cerr << "Failed to start pipeline" << std::endl;
        processing_active = false;
        return false;
    }
    
    // Main loop'u çalıştır (başka thread'de)
    std::thread loop_thread([this]() {
        g_main_loop_run(loop);
    });
    
    // EOS bekle veya timeout
    GstMessage *msg = gst_bus_timed_pop_filtered(bus, 30 * GST_SECOND, 
                                                 static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));
    
    bool success = false;
    if (msg) {
        if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_EOS) {
            std::cout << "GPU pipeline completed successfully" << std::endl;
            success = true;
        } else {
            GError *error;
            gchar *debug_info;
            gst_message_parse_error(msg, &error, &debug_info);
            std::cerr << "GPU Pipeline error: " << error->message << std::endl;
            if (debug_info) {
                std::cerr << "Debug info: " << debug_info << std::endl;
                g_free(debug_info);
            }
            g_error_free(error);
        }
        gst_message_unref(msg);
    } else {
        std::cerr << "GPU Pipeline timeout" << std::endl;
    }
    
    // Pipeline'ı durdur
    gst_element_set_state(pipeline, GST_STATE_NULL);
    g_main_loop_quit(loop);
    
    if (loop_thread.joinable()) {
        loop_thread.join();
    }
    
    processing_active = false;
    return success;
}

GstFlowReturn GPUPipeline::new_sample_callback(GstAppSink *appsink, gpointer user_data) {
    GPUPipeline *pipeline = static_cast<GPUPipeline*>(user_data);
    
    GstSample *sample = gst_app_sink_pull_sample(appsink);
    if (!sample) {
        return GST_FLOW_ERROR;
    }
    
    GstBuffer *buffer = gst_sample_get_buffer(sample);
    if (buffer) {
        pipeline->process_frame_buffer(buffer);
    }
    
    gst_sample_unref(sample);
    return GST_FLOW_OK;
}

void GPUPipeline::process_frame_buffer(GstBuffer *buffer) {
    if (!edge_detector) {
        // Edge detection olmadan direkt gönder
        gst_app_src_push_buffer(GST_APP_SRC(appsrc), gst_buffer_ref(buffer));
        return;
    }
    
    GstMapInfo map_info;
    if (!gst_buffer_map(buffer, &map_info, GST_MAP_READ)) {
        std::cerr << "Failed to map buffer" << std::endl;
        return;
    }
    
    // CUDA edge detection uygula
    uint8_t *processed_data = nullptr;
    size_t processed_size = 0;
    
    if (edge_detector->process_frame(map_info.data, map_info.size, 
                                    1920, 1080, &processed_data, &processed_size)) {
        
        // Yeni buffer oluştur
        GstBuffer *new_buffer = gst_buffer_new_allocate(nullptr, processed_size, nullptr);
        if (new_buffer) {
            GstMapInfo new_map;
            if (gst_buffer_map(new_buffer, &new_map, GST_MAP_WRITE)) {
                memcpy(new_map.data, processed_data, processed_size);
                gst_buffer_unmap(new_buffer, &new_map);
                
                // Timestamp'i kopyala
                GST_BUFFER_PTS(new_buffer) = GST_BUFFER_PTS(buffer);
                GST_BUFFER_DTS(new_buffer) = GST_BUFFER_DTS(buffer);
                
                // AppSrc'ye gönder
                gst_app_src_push_buffer(GST_APP_SRC(appsrc), new_buffer);
            } else {
                gst_buffer_unref(new_buffer);
            }
        }
        
        // CUDA memory'yi temizle
        if (processed_data) {
            cudaFree(processed_data);
        }
    } else {
        // Edge detection başarısız, orijinal buffer'ı gönder
        gst_app_src_push_buffer(GST_APP_SRC(appsrc), gst_buffer_ref(buffer));
    }
    
    gst_buffer_unmap(buffer, &map_info);
}

gboolean GPUPipeline::bus_callback(GstBus *bus, GstMessage *message, gpointer user_data) {
    GPUPipeline *pipeline = static_cast<GPUPipeline*>(user_data);
    
    switch (GST_MESSAGE_TYPE(message)) {
        case GST_MESSAGE_ERROR: {
            GError *error;
            gchar *debug_info;
            gst_message_parse_error(message, &error, &debug_info);
            std::cerr << "GPU Pipeline Bus Error: " << error->message << std::endl;
            if (debug_info) {
                std::cerr << "Debug info: " << debug_info << std::endl;
                g_free(debug_info);
            }
            g_error_free(error);
            g_main_loop_quit(pipeline->loop);
            break;
        }
        case GST_MESSAGE_EOS:
            std::cout << "GPU Pipeline reached EOS" << std::endl;
            g_main_loop_quit(pipeline->loop);
            break;
        default:
            break;
    }
    
    return TRUE;
}

void GPUPipeline::cleanup() {
    if (pipeline) {
        gst_element_set_state(pipeline, GST_STATE_NULL);
        gst_object_unref(pipeline);
        pipeline = nullptr;
    }
    
    if (bus) {
        gst_object_unref(bus);
        bus = nullptr;
    }
    
    if (loop) {
        g_main_loop_unref(loop);
        loop = nullptr;
    }
    
    is_initialized = false;
}