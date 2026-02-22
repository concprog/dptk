import threading
import multiprocess
from typing import Iterator
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from ..context import FrameContext
from ..stream import Stream
from ..decorators import _SENTINEL

class _RosListenerNode(Node):
    """
    Internal helper node. 
    It runs in a background thread and dumps data into a multiprocessing.Queue.
    """
    def __init__(self, topic_name: str, queue: multiprocess.Queue):
        super().__init__('pipeline_listener_node')
        self.queue = queue
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            topic_name,
            self.listener_callback,
            10
        )
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            ctx = FrameContext(
                frame=cv_image,
                index=msg.header.stamp.nanosec, 
                timestamp=msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9,
                metadata={'ros_topic': msg.header.frame_id}
            )
            
            self.queue.put(ctx)
        except Exception as e:
            self.get_logger().error(f'Error in callback: {e}')

def RosSource(topic_name: str) -> Stream:
    """
    Creates a Stream from a ROS 2 topic.
    """
    q = multiprocess.Queue(maxsize=64)
    
    def ros_spin_loop():
        # Only init if not already initialized
        if not rclpy.ok():
            rclpy.init()
        node = _RosListenerNode(topic_name, q)
        try:
            rclpy.spin(node)
        finally:
            node.destroy_node()
            # Only shutdown if we initialized it, or rely on user shutdown.
            # To be safe and keep it running for sinks, we don't strictly rclpy.shutdown() here
            # if we expect other nodes, but if standalone, we should.
            if rclpy.ok():
                rclpy.shutdown()
            q.put(_SENTINEL)

    t = threading.Thread(target=ros_spin_loop, daemon=True)
    t.start()

    def generator() -> Iterator[FrameContext]:
        for ctx in iter(q.get, _SENTINEL):
            yield ctx

    return Stream(source_generator=generator)
