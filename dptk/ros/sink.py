from typing import Callable, Iterable
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from ..context import FrameContext


def RosPublisherSink(
    node: Node, topic_name: str
) -> Callable[[Iterable[FrameContext]], None]:
    """
    Creates a sink that publishes frames to a ROS 2 topic.

    Args:
        node: A running rclpy Node to use for publishing.
        topic_name: The topic name to publish images to.

    Returns:
        A callable sink consumer.
    """
    publisher = node.create_publisher(Image, topic_name, 10)
    bridge = CvBridge()

    def consume(stream: Iterable[FrameContext]) -> None:
        for ctx in stream:
            try:
                # Convert OpenCV image (bgr8) to ROS Image message
                msg = bridge.cv2_to_imgmsg(ctx.frame, encoding="bgr8")

                # Update header from context
                msg.header.stamp.sec = int(ctx.timestamp)
                msg.header.stamp.nanosec = int(
                    (ctx.timestamp - int(ctx.timestamp)) * 1e9
                )
                msg.header.frame_id = ctx.metadata.get("ros_topic", "dptk_frame")

                publisher.publish(msg)
            except Exception as e:
                node.get_logger().error(f"Error publishing frame: {e}")

    return consume
