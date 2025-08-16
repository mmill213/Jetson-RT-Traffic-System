#!/usr/bin/env python3
import sys
sys.path.append('../')
import os
sys.path.append(os.path.expanduser('~/Jetson-RT-Traffic-System/src/deepstream_detect/deepstream_detect/common'))
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst
from common.bus_call import bus_call

import configparser
import pyds

import rclpy
from rclpy import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray
from vision_msgs.msg import ObjectHypothesisWithPose

import numpy as np
from cv_bridge import CvBridge
from threading import Thread

PGIE_CLASS_ID_VEHICLE = 0
MUXER_BATCH_TIMEOUT_USEC = 33000

class DeepStreamNode(Node):
    def __init__(self):
        super().__init__("deepstream_detector")
        self.declare_parameter("topic_name", "/image_raw")
        self.declare_parameter("fps", 30.0)

        self.fps = self.get_parameter('fps').get_parameter_value().double_value
        topic = self.get_parameter('topic_name').get_parameter_value().string_value
        if not topic:
            self.get_logger().error('Parameter "topic_name" not set! Exiting...')
            raise RuntimeError('topic_name not set')

        self.sub = self.create_subscription(Image, topic, self.image_callback, 10)
        self.pub = self.create_publisher(Detection2DArray, "traffic_detections", 10)

        self.prev_stamp = None
        self.caps_set = False
        self.bridge = CvBridge()

        # Initialize GStreamer
        Gst.init(None)
        self.pipeline = Gst.Pipeline()
        if not self.pipeline:
            raise RuntimeError("Failed to create GStreamer pipeline")

        # Streammux
        self.streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
        self.pipeline.add(self.streammux)

        # Source bin
        self.source_bin = self.create_source_bin(topic)
        self.pipeline.add(self.source_bin)
        sinkpad = self.streammux.request_pad_simple(f"sink{topic.replace('/', '-')}")
        srcpad = self.source_bin.get_static_pad("src")
        srcpad.link(sinkpad)

        # Primary inference
        self.pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
        self.pgie.set_property('config-file-path', "dstest2_pgie_config.txt")
        self.pipeline.add(self.pgie)

        # Tracker
        self.tracker = Gst.ElementFactory.make("nvtracker", "tracker")
        self.load_tracker_config('dstest2_tracker_config.txt')
        self.pipeline.add(self.tracker)

        # Convert + OSD
        self.nv_vid_converter = Gst.ElementFactory.make("nvvideoconvert", "converter")
        self.nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
        self.pipeline.add(self.nv_vid_converter)
        self.pipeline.add(self.nvosd)

        # Link the elements
        self.streammux.link(self.pgie)
        self.pgie.link(self.tracker)
        self.tracker.link(self.nv_vid_converter)
        self.nv_vid_converter.link(self.nvosd)

        # Probe for inference
        osdsinkpad = self.nvosd.get_static_pad("sink")
        osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, self.on_inference, 0)

        # GLib MainLoop in a separate thread
        self.glib_loop = GLib.MainLoop()
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", bus_call, self.glib_loop)
        self.glib_thread = Thread(target=self.glib_loop.run, daemon=True)
        self.glib_thread.start()

        # Start pipeline
        self.pipeline.set_state(Gst.State.PLAYING)

    def load_tracker_config(self, filename):
        config = configparser.ConfigParser()
        config.read(filename)
        for key in config['tracker']:
            val = config['tracker'][key]
            if key in ['tracker-width', 'tracker-height', 'gpu-id']:
                self.tracker.set_property(key.replace('-', '_'), int(val))
            else:
                self.tracker.set_property(key.replace('-', '_'), val)

    def create_source_bin(self, topic_name):
        bin_name = f"source-ros-bin-{topic_name.replace('/', '-')}"
        new_bin = Gst.Bin.new(bin_name)
        self.appsrc = Gst.ElementFactory.make("appsrc", f"appsrc{topic_name.replace('/', '-')}")
        placeholder_caps = Gst.Caps.from_string("video/x-raw,format=BGR,width=640,height=480,framerate=30/1")
        self.appsrc.set_property("caps", placeholder_caps)
        self.appsrc.set_property("is-live", True)
        self.appsrc.set_property("block", True)
        new_bin.add(self.appsrc)
        ghost_pad = Gst.GhostPad.new("src", self.appsrc.get_static_pad("src"))
        new_bin.add_pad(ghost_pad)
        return new_bin

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        if not self.caps_set:
            self.width, self.height = msg.width, msg.height
            gst_format = {'rgb8':'RGB','bgr8':'BGR','mono8':'GRAY8'}.get(msg.encoding, 'BGR')
            caps = Gst.Caps.from_string(
                f"video/x-raw,format={gst_format},width={self.width},height={self.height},framerate={int(self.fps)}/1"
            )
            self.appsrc.set_property("caps", caps)
            self.caps_set = True

        data = frame.tobytes()
        buf = Gst.Buffer.new_allocate(None, len(data), None)
        buf.fill(0, data)
        buf.duration = Gst.util_uint64_scale_int(1, Gst.SECOND, int(self.fps))
        self.appsrc.emit("push-buffer", buf)

    def on_inference(self, pad, info, u_data):
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(info.get_buffer()))
        for l_frame in batch_meta.frame_meta_list:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            detections = []
            for l_obj in frame_meta.obj_meta_list:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                detection = Detection2D()
                detection.bbox.center.x = (obj_meta.left + obj_meta.right)/2
                detection.bbox.center.y = (obj_meta.top + obj_meta.bottom)/2
                detection.bbox.size_x = obj_meta.right - obj_meta.left
                detection.bbox.size_y = obj_meta.bottom - obj_meta.top

                hypo = ObjectHypothesisWithPose()
                hypo.id = obj_meta.class_id
                hypo.score = obj_meta.confidence
                detection.results.append(hypo)
                detections.append(detection)

            self.publish_detections(detections)
        return Gst.PadProbeReturn.OK

    def publish_detections(self, detections):
        msg = Detection2DArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera_frame"
        msg.detections = detections
        self.pub.publish(msg)

def main():
    rclpy.init()
    node = DeepStreamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.pipeline.set_state(Gst.State.NULL)
    node.glib_loop.quit()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
