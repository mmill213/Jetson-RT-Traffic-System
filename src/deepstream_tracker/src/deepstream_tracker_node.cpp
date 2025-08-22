#include <rclcpp/rclcpp.hpp>

#include "ament_index_cpp/get_package_share_directory.hpp"

#include <vision_msgs/msg/detection2_d_array.hpp>
#include <vision_msgs/msg/object_hypothesis_with_pose.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>


#include <gst/gst.h>
#include <gst/app/gstappsrc.h>
#include "nvdsmeta.h"
#include "nvbufsurface.h"
#include "nvdsinfer.h"
#include "gstnvdsmeta.h"

using vision_msgs::msg::Detection2D;
using vision_msgs::msg::Detection2DArray;
using vision_msgs::msg::ObjectHypothesisWithPose;

class DeepStreamTrackerNode : public rclcpp::Node {
public:
  DeepStreamTrackerNode() : Node("deepstream_tracker_node") {
    declare_parameter<std::string>("source_topic", "");


    std::string pkg_share = ament_index_cpp::get_package_share_directory("deepstream_tracker");

    declare_parameter<std::string>("pgie_config", pkg_share + "/cfg/pgie_trafficcamnet_config.txt");
    declare_parameter<std::string>("tracker_config", pkg_share + "/cfg/tracker_iou_config.txt");


    pub_ = create_publisher<Detection2DArray>("detections", 10);

    std::string source_topic = get_parameter("source_topic").as_string();
    if (source_topic.empty()) {
        RCLCPP_FATAL(get_logger(), "Set source_topic to a valid input stream");
        rclcpp::shutdown();
        return;
    }

    sub_ = create_subscription<sensor_msgs::msg::Image>(
        source_topic, 
        10, 
        std::bind(&DeepStreamTrackerNode::image_callback, this, std::placeholders::_1)
    ); 

    RCLCPP_INFO(get_logger(), "DeepStream tracker node started");
  }

  ~DeepStreamTrackerNode() override {
    if (pipeline_) {
        gst_element_set_state(pipeline_, GST_STATE_NULL);
        gst_object_unref(pipeline_);
        pipeline_ = nullptr;
    }
  }

private:

  GstElement* build_pipeline(int width, int height) {
    
      std::string pgie = get_parameter("pgie_config").as_string();
      std::string tracker_cfg = get_parameter("tracker_config").as_string();

      gst_init(nullptr, nullptr);

      pipeline_ = gst_pipeline_new("ds-pipeline");
      appsrc_    = gst_element_factory_make("appsrc", "source");
      auto streammux = gst_element_factory_make("nvstreammux", "streammux");
      auto pgie_elt  = gst_element_factory_make("nvinfer", "primary-nvinfer");
      auto tracker   = gst_element_factory_make("nvtracker", "tracker");
      auto nvvconv   = gst_element_factory_make("nvvideoconvert", "conv");
      auto sink      = gst_element_factory_make("fakesink", "fakesink");


      if (!pipeline_ || !appsrc_ || !streammux || !pgie_elt || !tracker || !nvvconv || !sink) {
          RCLCPP_FATAL(get_logger(), "Failed to create GStreamer elements");
          return nullptr;
      }
    
      // Appsrc caps
      GstCaps* caps = gst_caps_new_simple(
          "video/x-raw",
          "format", G_TYPE_STRING, "BGR",
          "width", G_TYPE_INT, width,
          "height", G_TYPE_INT, height,
          "framerate", GST_TYPE_FRACTION, 30, 1,
          NULL
      );

      // settings 
      g_object_set(appsrc_, "caps", caps, "format", GST_FORMAT_TIME, "is-live", TRUE, NULL);
      gst_caps_unref(caps);
      g_object_set(G_OBJECT(streammux),
          "batch-size", 1,
          "width", width,
          "height", height,
          "batched-push-timeout", 40000,
          NULL);

      g_object_set(G_OBJECT(pgie_elt), "config-file-path", pgie.c_str(), NULL);
      g_object_set(G_OBJECT(tracker), "ll-config-file", tracker_cfg.c_str(), NULL);
      g_object_set(G_OBJECT(tracker), "ll-lib-file", "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so", NULL);

    
      gst_bin_add_many(GST_BIN(pipeline_), appsrc_, streammux, pgie_elt, tracker, nvvconv, sink, NULL);
    
      // Link appsrc → streammux
      GstPad* sinkpad = gst_element_get_request_pad(streammux, "sink_0");
      GstPad* srcpad  = gst_element_get_static_pad(appsrc_, "src");
      if (gst_pad_link(srcpad, sinkpad) != GST_PAD_LINK_OK) {
          RCLCPP_FATAL(get_logger(), "Failed to link appsrc to streammux");
          return nullptr;
      }
      gst_object_unref(srcpad);
      gst_object_unref(sinkpad);
    
      // Rest of pipeline
      if (!gst_element_link_many(streammux, pgie_elt, tracker, nvvconv, sink, NULL)) {
          RCLCPP_FATAL(get_logger(), "Failed to link downstream pipeline");
          return nullptr;
      }
    
      gst_element_set_state(pipeline_, GST_STATE_PLAYING);
    
      RCLCPP_INFO(this->get_logger(), "Pipeline built!");
      return pipeline_;
    }


    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg){
      cv::Mat frame = cv_bridge::toCvShare(msg, "bgr8")->image;



      if (!has_set_up && !frame.empty()){
        RCLCPP_INFO(this->get_logger(), "Recieved first frame, generating pipeline with size: %dx%d", frame.cols, frame.rows);
        build_pipeline(frame.cols, frame.rows);
        has_set_up = true;
      }

      
      // Push frame
      GstBuffer* buffer = gst_buffer_new_allocate(NULL, frame.total() * frame.elemSize(), NULL);
      GstMapInfo map;
      if (gst_buffer_map(buffer, &map, GST_MAP_WRITE)) {
        memcpy(map.data, frame.data, frame.total() * frame.elemSize());
        gst_buffer_unmap(buffer, &map);
      }
      gst_app_src_push_buffer(GST_APP_SRC(appsrc_), buffer);

    }


    bool has_set_up{false};

    rclcpp::Publisher<Detection2DArray>::SharedPtr pub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;


    GstElement* pipeline_{nullptr};
    GstElement* appsrc_{nullptr};
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<DeepStreamTrackerNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
