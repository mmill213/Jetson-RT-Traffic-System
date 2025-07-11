#include <chrono>
#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"

#include "cv_bridge/cv_bridge.h"
#include "opencv2/opencv.hpp"

class TrafficDetector : public rclcpp::Node {
public:
  TrafficDetector() : Node("traffic_detect") {
    subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
      "/image_raw", 10,
      std::bind(&TrafficDetector::image_callback, this, std::placeholders::_1)
    );
  }

private:
  void image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
    RCLCPP_INFO(this->get_logger(), "Received image: %d x %d", msg->width, msg->height);
    RCLCPP_INFO(this->get_logger(), "Format: %s", msg->encoding.c_str());

    
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
    


    
  }

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
