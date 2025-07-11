#include <chrono>
#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"

#include "cv_bridge/cv_bridge.h"
#include "opencv2/opencv.hpp"

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <fstream>

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

    std::ifstream engine_file("/home/nvidia/Jetson-RT-Traffic-System/camera-model/model.trt", std::ios::binary);
    if (!engine_file) {
      RCLCPP_ERROR(this->get_logger(), "Failed to open TensorRT engine file.");
      throw std::runtime_error("Failed to open TensorRT engine file.");
    }
    engine_file.seekg(0, std::ifstream::end);
    size_t engine_size = engine_file.tellg();
    engine_file.seekg(0, std::ifstream::beg);
    std::vector<char> engine_data(engine_size);
    engine_file.read(engine_data.data(), engine_size);
    runtime_ = nvinfer1::createInferRuntime(gLogger);
    engine_ = runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size());
    context_ = engine_->createExecutionContext();
  }


  ~TrafficDetector() override {
    if (context_) delete context_;
    if (engine_) delete engine_;
    if (runtime_) delete runtime_;
  }

private:
  void image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
    RCLCPP_INFO(this->get_logger(), "Received image: %d x %d, Format: %s", msg->width, msg->height, msg->encoding.c_str());

    
    // Check for empty image data
    if (msg->data.empty() || msg->width == 0 || msg->height == 0) {
      RCLCPP_WARN(this->get_logger(), "Received empty image data.");
      return;
    }
   
    cv::Mat image;
    try {
      image = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::YUV422)->image;
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

    float* d_input = nullptr;
    size_t input_size = input_tensor.size() * sizeof(float);
    cudaError_t cudaStatus = cudaMalloc((void**)&d_input, input_size);
    if (cudaStatus != cudaSuccess) {
      RCLCPP_ERROR(this->get_logger(), "cudaMalloc failed: %s", cudaGetErrorString(cudaStatus));
      return;
    }
    cudaStatus = cudaMemcpy(d_input, input_tensor.data(), input_size, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
      RCLCPP_ERROR(this->get_logger(), "cudaMemcpy failed: %s", cudaGetErrorString(cudaStatus));
      cudaFree(d_input);
      return;
    }





    cudaFree(d_input);
    // Remember to cudaFree(d_input) after inference is done!
    
    
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
