#include <chrono>
#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"

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
    std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Control loop rate
  }

  rclcpp::shutdown();
  return 0;
}
