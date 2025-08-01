#include <chrono>
#include <memory>
#include "string.h"
#include <algorithm>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "vision_msgs/msg/bounding_box2_d_array.hpp"



#include "cv_bridge/cv_bridge.h"
#include "opencv2/opencv.hpp"

//#include "vision_msgs/msg/detection2_d_array.hpp"

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <fstream>
#include "vision_msgs/msg/detection2_d_array.hpp"


#include <cstdlib>
#include <string>

std::string get_model_path() {
  const char* home = std::getenv("HOME");
  if (!home) home = "~";
  return std::string(home) + "/Jetson-RT-Traffic-System/camera-model/model.trt";
}
#define MODEL_PATH get_model_path().c_str()


class Logger : public nvinfer1::ILogger {
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING)
      std::cout << "[TensorRT] " << msg << std::endl;
  }
} gLogger;



// Compute Intersection over Union for two boxes
float IoU(const vision_msgs::msg::Detection2D& a, const vision_msgs::msg::Detection2D& b) {
    float ax1 = a.bbox.center.position.x - a.bbox.size_x / 2.0f;
    float ay1 = a.bbox.center.position.y - a.bbox.size_y / 2.0f;
    float ax2 = a.bbox.center.position.x + a.bbox.size_x / 2.0f;
    float ay2 = a.bbox.center.position.y + a.bbox.size_y / 2.0f;

    float bx1 = b.bbox.center.position.x - b.bbox.size_x / 2.0f;
    float by1 = b.bbox.center.position.y - b.bbox.size_y / 2.0f;
    float bx2 = b.bbox.center.position.x + b.bbox.size_x / 2.0f;
    float by2 = b.bbox.center.position.y + b.bbox.size_y / 2.0f;

    float inter_x1 = std::max(ax1, bx1);
    float inter_y1 = std::max(ay1, by1);
    float inter_x2 = std::min(ax2, bx2);
    float inter_y2 = std::min(ay2, by2);

    float inter_area = std::max(0.0f, inter_x2 - inter_x1) * std::max(0.0f, inter_y2 - inter_y1);
    float a_area = (ax2 - ax1) * (ay2 - ay1);
    float b_area = (bx2 - bx1) * (by2 - by1);

    return inter_area / (a_area + b_area - inter_area);
}

// NMS filter function
std::vector<vision_msgs::msg::Detection2D> non_max_suppression(
    const std::vector<vision_msgs::msg::Detection2D>& boxes,
    float iou_threshold = 0.5f) {

    std::vector<vision_msgs::msg::Detection2D> result;

    // Copy boxes and sort descending by confidence
    std::vector<vision_msgs::msg::Detection2D> sorted_boxes = boxes;
    std::sort(sorted_boxes.begin(), sorted_boxes.end(),
        [](const auto& a, const auto& b) {
            return a.results[0].hypothesis.score > b.results[0].hypothesis.score;
        });

    std::vector<bool> suppressed(sorted_boxes.size(), false);

    for (size_t i = 0; i < sorted_boxes.size(); ++i) {
        if (suppressed[i]) continue;

        result.push_back(sorted_boxes[i]);

        for (size_t j = i + 1; j < sorted_boxes.size(); ++j) {
            if (suppressed[j]) continue;

            if (IoU(sorted_boxes[i], sorted_boxes[j]) > iou_threshold) {
                suppressed[j] = true;
            }
        }
    }
    return result;
}

class TrafficDetector : public rclcpp::Node {
public:
  TrafficDetector() : Node("traffic_detect") {
    this->declare_parameter<float>("conf_thresh", 0.3f);
    this->declare_parameter<float>("intersection_max", 0.3f);
    
    this->get_parameter("conf_thresh", conf_thresh_);
    this->get_parameter("intersection_max", iou_t_);
    


    subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
      "/image_raw", 10,
      std::bind(&TrafficDetector::image_callback, this, std::placeholders::_1)
    );

    detection_pub_ = this->create_publisher<vision_msgs::msg::Detection2DArray>("traffic_detections", 10);


    //load engine
    std::ifstream engine_file(MODEL_PATH, std::ios::binary);
    if (!engine_file) {
      RCLCPP_ERROR(this->get_logger(), "Failed to open TensorRT engine file.");
      throw std::runtime_error("Failed to open TensorRT engine file.");
    }
    engine_file.seekg(0, std::ifstream::end);
    size_t engine_size = engine_file.tellg(); // grab engine size at EOF
    engine_file.seekg(0, std::ifstream::beg);
    std::vector<char> engine_data(engine_size);
    engine_file.read(engine_data.data(), engine_size);
    //end load engine
    runtime_ = nvinfer1::createInferRuntime(gLogger); // bind logger
    engine_ = runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size());
    context_ = engine_->createExecutionContext(); // create context for model

    for(int i = 0; i < engine_->getNbIOTensors(); i++){
      RCLCPP_INFO(this->get_logger(), "name: %s\n", engine_->getIOTensorName(i)); // list all tensor names in order
    }
    
    input_idx = -1;
    cov_idx = -1;
    bbox_idx  = -1;
    
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
    
    
    cudaMalloc(&this->buffers[input_idx], input_size);
    cudaMalloc(&this->buffers[cov_idx], cov_size);
    cudaMalloc(&this->buffers[bbox_idx], bbox_size);


  }


  ~TrafficDetector() override {
    // Free CUDA buffers
    
    cudaFree(buffers[input_idx]);
    cudaFree(buffers[cov_idx]);
    cudaFree(buffers[bbox_idx]);


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
   
    const int model_width = 960;
    const int model_height = 544;

    // model expects rgb8
    cv::Mat image;
    try {
      auto cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
      image = cv_ptr->image;
    } catch (cv_bridge::Exception& e) {
      RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
      return;
    }

    // Resize to model size
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(model_width, model_height));

    // Convert to float and scale to [0, 1]
    resized.convertTo(resized, CV_32FC3, 1.0 / 255.0); // CV_32FC3 -> f32 @ 3 channels

    // Convert height x width x color -> normalized-channel x height x width
    std::vector<float> input_tensor(3 * model_height * model_width);
    int idx = 0;
    for (int c = 0; c < 3; ++c) {
      for (int i = 0; i < model_height; ++i) {
        for (int j = 0; j < model_width; ++j) {
          input_tensor[idx++] = resized.at<cv::Vec3f>(i, j)[c];
        }
      }
    }

    
  
    // Upload input
    cudaMemcpy(buffers[input_idx], input_tensor.data(), input_tensor.size() * sizeof(float), cudaMemcpyHostToDevice);

    // Run inference thru model context
    context_->executeV2(buffers);


    std::vector<float> output_cov(4 * 34 * 60);
    std::vector<float> output_bbox(16 * 34 * 60);
    // Download outputs
    cudaMemcpy(output_cov.data(), buffers[cov_idx], output_cov.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(output_bbox.data(), buffers[bbox_idx], output_bbox.size() * sizeof(float), cudaMemcpyDeviceToHost);

    
    // publish the found data after processing into a ROS msg
    send_data(&output_cov, &output_bbox, static_cast<float>(msg->width), static_cast<float>(msg->height), msg->header.frame_id);
  
    
  }

  

  void send_data(std::vector<float>* cov, std::vector<float>* bbox, float img_w, float img_h, std::string& frid) {
    const int grid_h = 34;
    const int grid_w = 60;
    const int grid_d = 4;

    const float cell_w = 960.0f / static_cast<float>(grid_w);  // cell size on model input scale
    const float cell_h = 544.0f / static_cast<float>(grid_h);


    const char* ids[] = {
      "car",
      "bicycle",
      "person",
      "road_sign"
    };


    std::vector<vision_msgs::msg::Detection2D> raw_detections;
    

    float scale_x = img_w / 960.0f;
    float scale_y = img_h / 544.0f; 

    for (int i = 0; i < grid_h; ++i) {
      for (int j = 0; j < grid_w; ++j) {
        for (int k = 0; k < grid_d; ++k) {
          
          // if the covaraince vector is less than the thresh at the index of our object, skip this elem
          float confidence = (*cov)[k * grid_h * grid_w + i * grid_w + j];
          if (confidence < conf_thresh_) continue;

          // otherwise, we need to proc the elem in the bbox vector
          // bboxes have size of 4x ints, so base == 'l' == k * 4 
          int base = k * 4;
          

          // (x, y, w, h)
          // collect all values at index == base fro mthe model bbox vector
          // base + n is essentially k once the vector is flattened
          float x_raw = (*bbox)[(base + 0) * grid_h * grid_w + i * grid_w + j];
          float y_raw = (*bbox)[(base + 1) * grid_h * grid_w + i * grid_w + j];
          float w_raw = (*bbox)[(base + 2) * grid_h * grid_w + i * grid_w + j];
          float h_raw = (*bbox)[(base + 3) * grid_h * grid_w + i * grid_w + j];

          //RCLCPP_INFO(this->get_logger(), "raw vals: {x = %.2f, y = %.2f, w = %.2f, h = %.2f}", x_raw, y_raw, w_raw, h_raw);
          
          float w_model = w_raw * scale_x; // TODO try permuations with 960, cell_w, and 16
          float h_model = h_raw * scale_y;

          //grab the actual x-y from the current cell
          float x_model = (j + x_raw) * cell_w;
          float y_model = (i + y_raw) * cell_h;

          // Now scale to original image size:
          float x = x_model * scale_x;
          float y = y_model * scale_y;
          float w = w_model * 60.0f;
          float h = h_model * 34.0f;
                  

          vision_msgs::msg::Detection2D det;
          

          //log to the bbox data structure
          det.bbox.center.position.x = x;
          det.bbox.center.position.y = y;
          det.bbox.size_x = w;
          det.bbox.size_y = h;

          det.results.resize(1);
          det.results[0].hypothesis.class_id = ids[k];  // or std::to_string(k)
          det.results[0].hypothesis.score = confidence;
          //push bbox
          raw_detections.push_back(det);
        }
      }
    }

    // Now group by class and run NMS
    std::map<std::string, std::vector<vision_msgs::msg::Detection2D>> detections_by_class;
    for (auto& det : raw_detections) {
        detections_by_class[det.results[0].hypothesis.class_id].push_back(det);
    }

    vision_msgs::msg::Detection2DArray filtered_detections;
    filtered_detections.header.stamp = this->get_clock()->now();
    filtered_detections.header.frame_id = frid.c_str();

    for (auto& [class_id, dets] : detections_by_class) {
        auto nmsed = non_max_suppression(dets, iou_t_);
        for (auto& d : nmsed) {
            
            filtered_detections.detections.push_back(d);
        }
    }

    this->detection_pub_->publish(filtered_detections);
  }



  rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr detection_pub_;


  nvinfer1::IRuntime* runtime_{nullptr};
  nvinfer1::ICudaEngine* engine_{nullptr};
  nvinfer1::IExecutionContext* context_{nullptr};

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;

  
  float conf_thresh_{0.05f};  // default value
  float iou_t_{0.5f};

  void* buffers[3];
  int input_idx;
  int cov_idx;
  int bbox_idx;
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
