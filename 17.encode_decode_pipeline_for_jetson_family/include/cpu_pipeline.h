#ifndef CPU_PIPELINE_H
#define CPU_PIPELINE_H

#include <string>
#include <gst/gst.h>
#include <opencv2/opencv.hpp>

class CPUPipeline {
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
    
    bool is_initialized;
    bool processing_active;
    
    // OpenCV edge detection
    cv::Mat current_frame;
    cv::Mat processed_frame;
    
    // Callback functions
    static GstFlowReturn new_sample_callback(GstAppSink *appsink, gpointer user_data);
    static gboolean bus_callback(GstBus *bus, GstMessage *message, gpointer user_data);
    
    // Pipeline setup
    bool setup_decode_pipeline();
    bool setup_encode_pipeline();
    bool link_elements();
    
    // Frame processing
    void process_frame_buffer(GstBuffer *buffer);
    bool apply_cpu_edge_detection(const cv::Mat& input, cv::Mat& output);
    
public:
    CPUPipeline();
    ~CPUPipeline();
    
    bool initialize();
    void cleanup();
    
    void set_input_file(const std::string& file);
    void set_output_file(const std::string& file);
    
    bool process();
    bool is_processing() const { return processing_active; }
};

#endif // CPU_PIPELINE_H