#ifndef GPU_PIPELINE_H
#define GPU_PIPELINE_H

#include <string>
#include <memory>
#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>

class CudaEdgeDetector;

class GPUPipeline {
private:
    GstElement *pipeline;
    GstElement *source;
    GstElement *decoder;
    GstElement *videoconvert;
    GstElement *appsink;
    GstElement *appsrc;
    GstElement *encoder;
    GstElement *filesink;
    
    GstBus *bus;
    GMainLoop *loop;
    
    std::string input_file;
    std::string output_file;
    CudaEdgeDetector *edge_detector;
    
    bool is_initialized;
    bool processing_active;
    
    // Callback fonksiyonları
    static GstFlowReturn new_sample_callback(GstAppSink *appsink, gpointer user_data);
    static gboolean bus_callback(GstBus *bus, GstMessage *message, gpointer user_data);
    
    // Pipeline kurulum fonksiyonları
    bool setup_decode_pipeline();
    bool setup_encode_pipeline();
    bool link_elements();
    
    // Buffer işleme
    void process_frame_buffer(GstBuffer *buffer);
    
public:
    GPUPipeline();
    ~GPUPipeline();
    
    bool initialize();
    void cleanup();
    
    void set_input_file(const std::string& file);
    void set_output_file(const std::string& file);
    void set_edge_detector(CudaEdgeDetector* detector);
    
    bool process();
    bool is_processing() const { return processing_active; }
    
    // Jetson özellikli fonksiyonlar
    bool setup_jetson_decoder();
    bool setup_jetson_encoder();
    bool configure_nvmm_memory();
};

#endif // GPU_PIPELINE_H