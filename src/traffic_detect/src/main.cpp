#include <chrono>
#include <memory>
#include "string.h"

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"



#include "cv_bridge/cv_bridge.h"
#include "opencv2/opencv.hpp"

//#include "vision_msgs/msg/detection2_d_array.hpp"

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <fstream>




//#define MODEL_PATH "/home/nvidia/Jetson-RT-Traffic-System/camera-model/model.trt"
#define MODEL_PATH "/home/main-user/Jetson-RT-Traffic-System/camera-model/model.trt"


class Logger : public nvinfer1::ILogger {
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING)
      std::cout << "[TensorRT] " << msg << std::endl;
  }
} gLogger;





class TrafficDetector : public rclcpp::Node {
public:
  TrafficDetector() : Node("traffic_detect") {
    subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
      "/image_raw", 10,
      std::bind(&TrafficDetector::image_callback, this, std::placeholders::_1)
    );

    //detection_pub_ = this->create_publisher<vision_msgs::msg::Detection2DArray>("detections", 10);


    //load engine
    std::ifstream engine_file(MODEL_PATH, std::ios::binary);
    if (!engine_file) {
      RCLCPP_ERROR(this->get_logger(), "Failed to open TensorRT engine file.");
      throw std::runtime_error("Failed to open TensorRT engine file.");
    }
    engine_file.seekg(0, std::ifstream::end);
    size_t engine_size = engine_file.tellg();
    engine_file.seekg(0, std::ifstream::beg);
    std::vector<char> engine_data(engine_size);
    engine_file.read(engine_data.data(), engine_size);
    //end load engine
    runtime_ = nvinfer1::createInferRuntime(gLogger); // bind logger
    engine_ = runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size());
    context_ = engine_->createExecutionContext();

    for(int i = 0; i < engine_->getNbIOTensors(); i++){
      RCLCPP_INFO(this->get_logger(), "name: %s\n", engine_->getIOTensorName(i));
    }
    
  }


  ~TrafficDetector() override {
    if (context_) delete context_;
    if (engine_) delete engine_;
    if (runtime_) delete runtime_;
  }

private:


  void image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {


    RCLCPP_INFO(this->get_logger(), "Received image: %d x %d, Format: %s", msg->width, msg->height, msg->encoding.c_str()); // heartbeat msg

    
    // Check for empty image data
    if (msg->data.empty() || msg->width == 0 || msg->height == 0) {
      RCLCPP_WARN(this->get_logger(), "Received empty image data.");
      return;
    }
   
    //grab image from latest topic msg
    cv::Mat image;
    try {
      image = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::YUV422_YUY2)->image;
    } catch (cv_bridge::Exception& e) {
      RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
      return;
    }
    
    // Preprocess
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(960, 544));
    resized.convertTo(resized, CV_32FC3, 1.0 / 255);

    // NHWC to NCHW
    std::vector<float> input_tensor((3 * 544 * 960));
    int idx = 0;
    for (int c = 0; c < 3; ++c) {
      for (int i = 0; i < 544; ++i) {
        for (int j = 0; j < 960; ++j) {
          input_tensor[idx++] = resized.at<cv::Vec3f>(i, j)[c];
        }
      }
    }

    
    int input_idx = -1;
    int cov_idx = -1;
    int bbox_idx  = -1;
    
    for (int i = 0; i < engine_->getNbIOTensors(); i++){
      if (strcmp(engine_->getIOTensorName(i), "input_1:0") == 0){
        input_idx = i;
      } else if (strcmp(engine_->getIOTensorName(i), "output_cov/Sigmoid:0") == 0){
        cov_idx = i;
      } else if (strcmp(engine_->getIOTensorName(i), "output_bbox/BiasAdd:0") == 0) {
        bbox_idx = i;
      }
    } // load idxs

    void* buffers[3];

    size_t input_size = 1 * 3 * 544 * 960 * sizeof(float);
    size_t cov_size = 1 * 4 * 34 * 60 * sizeof(float);
    size_t bbox_size = 1 * 16 * 34 * 60 * sizeof(float);
    
    
    cudaMalloc(&buffers[input_idx], input_size);
    cudaMalloc(&buffers[cov_idx], cov_size);
    cudaMalloc(&buffers[bbox_idx], bbox_size);

    // Upload input
    cudaMemcpy(buffers[input_idx], input_tensor.data(), input_tensor.size(), cudaMemcpyHostToDevice);

    // Run inference
    context_->executeV2(buffers);


    std::vector<float> output_cov(cov_size);
    std::vector<float> output_bbox(bbox_size);
    // Download outputs
    cudaMemcpy(output_cov.data(), buffers[cov_idx], output_cov.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(output_bbox.data(), buffers[bbox_idx], output_bbox.size(), cudaMemcpyDeviceToHost);

    RCLCPP_INFO(this->get_logger(), "Output cov:");
    for (size_t i = 0; i < std::min<size_t>(10, output_cov.size()); ++i) {
      std::cout << output_cov[i] << " ";
    }
    std::cout << std::endl;

    RCLCPP_INFO(this->get_logger(), "Output bbox:");
    for (size_t i = 0; i < std::min<size_t>(10, output_bbox.size()); ++i) {
      std::cout << output_bbox[i] << " ";
    }
    std::cout << std::endl;

    // Free CUDA buffers
    cudaFree(buffers[input_idx]);
    cudaFree(buffers[cov_idx]);
    cudaFree(buffers[bbox_idx]);
    
  }

  nvinfer1::IRuntime* runtime_{nullptr};
  nvinfer1::ICudaEngine* engine_{nullptr};
  nvinfer1::IExecutionContext* context_{nullptr};

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
};







int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<TrafficDetector>();

  while (rclcpp::ok()) {
    // Your custom loop code here
    // For example: node->do_something();

    rclcpp::spin_some(node); // Process callbacks
    std::this_thread::sleep_for(std::chrono::milliseconds(1)); // Control loop rate
  }

  rclcpp::shutdown();
  return 0;
}
