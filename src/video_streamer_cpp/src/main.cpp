#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <image_transport/image_transport.hpp>

class VideoStreamer : public rclcpp::Node
{
public:
    VideoStreamer()
    : Node("video_streamer")
    {

        declare_parameter<std::string>("video_filepath", "");
        declare_parameter<int>("fps", 30);

        declare_parameter("width", 960);
        declare_parameter("height", 544);
        
        width_ = get_parameter("width").as_int();
        height_ = get_parameter("height").as_int();

        int fps = get_parameter("fps").as_int();

        std::string video_path = get_parameter("video_filepath").as_string();
        cap_.open(video_path);
        if (!cap_.isOpened()) {
            RCLCPP_ERROR(this->get_logger(), "Cannot open video file: %s", video_path.c_str());
            throw std::runtime_error("Failed to open video file");
        }

        image_pub_ = image_transport::create_publisher(this, "image_raw");

        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(static_cast<int>(1000/fps)),  
            std::bind(&VideoStreamer::publish_frame, this)
        );
    }

private:
    void publish_frame()
    {
        cv::Mat frame;
        if (!cap_.read(frame)) {
            RCLCPP_INFO(this->get_logger(), "Video finished. Looping...");
            cap_.set(cv::CAP_PROP_POS_FRAMES, 0);
            return;
        }
        cv::Mat output_frame;
        cv::resize(frame, output_frame, cv::Size(width_, height_));
        
        cv::Mat rgb_frame;
        cv::cvtColor(output_frame, rgb_frame, cv::COLOR_BGR2RGB);

        sensor_msgs::msg::Image::SharedPtr msg = cv_bridge::CvImage(
           std_msgs::msg::Header(), "rgb8", rgb_frame
        ).toImageMsg();

        // sensor_msgs::msg::Image::SharedPtr msg = cv_bridge::CvImage(
        //     std_msgs::msg::Header(), "bgr8", frame
        // ).toImageMsg();

        image_pub_.publish(msg);
    }

    cv::VideoCapture cap_;
    image_transport::Publisher image_pub_;
    rclcpp::TimerBase::SharedPtr timer_;

    int width_;
    int height_;
};

int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);

    auto node = std::make_shared<VideoStreamer>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
