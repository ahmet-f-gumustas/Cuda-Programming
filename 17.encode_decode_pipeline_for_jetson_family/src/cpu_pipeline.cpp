#include "cpu_pipeline.h"
#include <iostream>
#include <thread>
#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>

CPUPipeline::CPUPipeline() 
    : pipeline(nullptr), source(nullptr), decoder(nullptr), 
      videoconvert(nullptr), appsink(nullptr), appsrc(nullptr),
      encoder(nullptr), filesink(nullptr), bus(nullptr), loop(nullptr),
      is_initialized(false), processing_active(false) {
}

CPUPipeline::~CPUPipeline() {
    cleanup();
}

bool CPUPipeline::initialize() {
    std::cout << "Initializing CPU Pipeline..." << std::endl;
    
    // Pipeline oluştur
    pipeline = gst_pipeline_new("cpu-pipeline");
    if (!pipeline) {
        std::cerr << "Failed to create CPU pipeline" << std::endl;
        return false;
    }
    
    // Decode pipeline setup
    if (!setup_decode_pipeline()) {
        std::cerr << "Failed to setup CPU decode pipeline" << std::endl;
        return false;
    }
    
    // Encode pipeline setup
    if (!setup_encode_pipeline()) {
        std::cerr << "Failed to setup CPU encode pipeline" << std::endl;
        return false;
    }
    
    // Elements'leri bağla
    if (!link_elements()) {
        std::cerr << "Failed to link CPU pipeline elements" << std::endl;
        return false;
    }
    
    // Bus setup
    bus = gst_element_get_bus(pipeline);
    gst_bus_add_watch(bus, bus_callback, this);
    
    // Main loop
    loop = g_main_loop_new(nullptr, FALSE);
    
    is_initialized = true;
    std::cout << "CPU Pipeline initialized successfully" << std::endl;
    return true;
}

bool CPUPipeline::setup_decode_pipeline() {
    // Software decoder elements
    source = gst_element_factory_make("filesrc", "cpu-file-source");
    decoder = gst_element_factory_make("avdec_h264", "cpu-decoder");
    videoconvert = gst_element_factory_make("videoconvert", "cpu-convert");
    appsink = gst_element_factory_make("appsink", "cpu-app-sink");
    
    if (!source || !decoder || !videoconvert || !appsink) {
        std::cerr << "Failed to create CPU decoder elements" << std::endl;
        return false;
    }
    
    // Source properties
    if (!input_file.empty()) {
        g_object_set(source, "location", input_file.c_str(), nullptr);
    }
    
    // AppSink properties
    g_object_set(appsink,
                 "emit-signals", TRUE,
                 "sync", FALSE,
                 "max-buffers", 1,
                 "drop", TRUE,
                 "caps", gst_caps_from_string("video/x-raw,format=BGR,width=1920,height=1080"),
                 nullptr);
    
    // Callback bağla
    g_signal_connect(appsink, "new-sample", G_CALLBACK(new_sample_callback), this);
    
    // Pipeline'a ekle
    gst_bin_add_many(GST_BIN(pipeline), source, decoder, videoconvert, appsink, nullptr);
    
    return true;
}

bool CPUPipeline::setup_encode_pipeline() {
    // Software encoder elements
    appsrc = gst_element_factory_make("appsrc", "cpu-app-source");
    encoder = gst_element_factory_make("x264enc", "cpu-encoder");
    filesink = gst_element_factory_make("filesink", "cpu-file-sink");
    
    if (!appsrc || !encoder || !filesink) {
        std::cerr << "Failed to create CPU encoder elements" << std::endl;
        return false;
    }
    
    // AppSrc properties
    g_object_set(appsrc,
                 "caps", gst_caps_from_string("video/x-raw,format=BGR,width=1920,height=1080,framerate=60/1"),
                 "format", GST_FORMAT_TIME,
                 "is-live", TRUE,
                 nullptr);
    
    // x264enc properties - optimize for speed
    g_object_set(encoder,
                 "bitrate", 8000,
                 "speed-preset", 1, // ultrafast
                 "tune", 0x00000004, // zerolatency
                 "key-int-max", 60,
                 nullptr);
    
    // FileSink properties
    if (!output_file.empty()) {
        g_object_set(filesink, "location", output_file.c_str(), nullptr);
    }
    
    // Pipeline'a ekle
    gst_bin_add_many(GST_BIN(pipeline), appsrc, encoder, filesink, nullptr);
    
    return true;
}

bool CPUPipeline::link_elements() {
    // Decode chain
    if (!gst_element_link_many(source, decoder, videoconvert, appsink, nullptr)) {
        std::cerr << "Failed to link CPU decode chain" << std::endl;
        return false;
    }
    
    // Encode chain
    if (!gst_element_link_many(appsrc, encoder, filesink, nullptr)) {
        std::cerr << "Failed to link CPU encode chain" << std::endl;
        return false;
    }
    
    return true;
}

void CPUPipeline::set_input_file(const std::string& file) {
    input_file = file;
    if (source) {
        g_object_set(source, "location", file.c_str(), nullptr);
    }
}

void CPUPipeline::set_output_file(const std::string& file) {
    output_file = file;
    if (filesink) {
        g_object_set(filesink, "location", file.c_str(), nullptr);
    }
}

bool CPUPipeline::process() {
    if (!is_initialized) {
        std::cerr << "CPU Pipeline not initialized" << std::endl;
        return false;
    }
    
    std::cout << "Starting CPU pipeline processing..." << std::endl;
    processing_active = true;
    
    // Pipeline'ı başlat
    GstStateChangeReturn ret = gst_element_set_state(pipeline, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        std::cerr << "Failed to start CPU pipeline" << std::endl;
        processing_active = false;
        return false;
    }
    
    // Main loop thread
    std::thread loop_thread([this]() {
        g_main_loop_run(loop);
    });
    
    // EOS bekle
    GstMessage *msg = gst_bus_timed_pop_filtered(bus, 30 * GST_SECOND,
                                                 static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));
    
    bool success = false;
    if (msg) {
        if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_EOS) {
            std::cout << "CPU pipeline completed successfully" << std::endl;
            success = true;
        } else {
            GError *error;
            gchar *debug_info;
            gst_message_parse_error(msg, &error, &debug_info);
            std::cerr << "CPU Pipeline error: " << error->message << std::endl;
            if (debug_info) {
                std::cerr << "Debug info: " << debug_info << std::endl;
                g_free(debug_info);
            }
            g_error_free(error);
        }
        gst_message_unref(msg);
    } else {
        std::cerr << "CPU Pipeline timeout" << std::endl;
    }
    
    // Cleanup
    gst_element_set_state(pipeline, GST_STATE_NULL);
    g_main_loop_quit(loop);
    
    if (loop_thread.joinable()) {
        loop_thread.join();
    }
    
    processing_active = false;
    return success;
}

GstFlowReturn CPUPipeline::new_sample_callback(GstAppSink *appsink, gpointer user_data) {
    CPUPipeline *pipeline = static_cast<CPUPipeline*>(user_data);
    
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

void CPUPipeline::process_frame_buffer(GstBuffer *buffer) {
    GstMapInfo map_info;
    if (!gst_buffer_map(buffer, &map_info, GST_MAP_READ)) {
        std::cerr << "Failed to map CPU buffer" << std::endl;
        return;
    }
    
    // OpenCV Mat oluştur (BGR format)
    cv::Mat frame(1080, 1920, CV_8UC3, map_info.data);
    
    // Edge detection uygula
    if (apply_cpu_edge_detection(frame, processed_frame)) {
        // Yeni buffer oluştur
        size_t data_size = processed_frame.total() * processed_frame.elemSize();
        GstBuffer *new_buffer = gst_buffer_new_allocate(nullptr, data_size, nullptr);
        
        if (new_buffer) {
            GstMapInfo new_map;
            if (gst_buffer_map(new_buffer, &new_map, GST_MAP_WRITE)) {
                memcpy(new_map.data, processed_frame.data, data_size);
                gst_buffer_unmap(new_buffer, &new_map);
                
                // Timestamp kopyala
                GST_BUFFER_PTS(new_buffer) = GST_BUFFER_PTS(buffer);
                GST_BUFFER_DTS(new_buffer) = GST_BUFFER_DTS(buffer);
                
                // AppSrc'ye gönder
                gst_app_src_push_buffer(GST_APP_SRC(appsrc), new_buffer);
            } else {
                gst_buffer_unref(new_buffer);
            }
        }
    } else {
        // Edge detection başarısız, orijinal frame'i gönder
        gst_app_src_push_buffer(GST_APP_SRC(appsrc), gst_buffer_ref(buffer));
    }
    
    gst_buffer_unmap(buffer, &map_info);
}

bool CPUPipeline::apply_cpu_edge_detection(const cv::Mat& input, cv::Mat& output) {
    try {
        cv::Mat gray;
        cv::Mat grad_x, grad_y;
        cv::Mat abs_grad_x, abs_grad_y;
        
        // BGR'dan Grayscale'e çevir
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
        
        // Sobel operatörü uygula
        cv::Sobel(gray, grad_x, CV_16S, 1, 0, 3);
        cv::Sobel(gray, grad_y, CV_16S, 0, 1, 3);
        
        // Mutlak değerleri al
        cv::convertScaleAbs(grad_x, abs_grad_x);
        cv::convertScaleAbs(grad_y, abs_grad_y);
        
        // Gradientleri birleştir
        cv::Mat edge_gray;
        cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, edge_gray);
        
        // Grayscale'i BGR'a çevir (encoder için)
        cv::cvtColor(edge_gray, output, cv::COLOR_GRAY2BGR);
        
        return true;
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV error in edge detection: " << e.what() << std::endl;
        return false;
    }
}

gboolean CPUPipeline::bus_callback(GstBus *bus, GstMessage *message, gpointer user_data) {
    CPUPipeline *pipeline = static_cast<CPUPipeline*>(user_data);
    
    switch (GST_MESSAGE_TYPE(message)) {
        case GST_MESSAGE_ERROR: {
            GError *error;
            gchar *debug_info;
            gst_message_parse_error(message, &error, &debug_info);
            std::cerr << "CPU Pipeline Bus Error: " << error->message << std::endl;
            if (debug_info) {
                std::cerr << "Debug info: " << debug_info << std::endl;
                g_free(debug_info);
            }
            g_error_free(error);
            g_main_loop_quit(pipeline->loop);
            break;
        }
        case GST_MESSAGE_EOS:
            std::cout << "CPU Pipeline reached EOS" << std::endl;
            g_main_loop_quit(pipeline->loop);
            break;
        default:
            break;
    }
    
    return TRUE;
}

void CPUPipeline::cleanup() {
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