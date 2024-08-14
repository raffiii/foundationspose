import queue
import threading
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime

import numpy as np


@dataclass
class DepthFrame:
    color: np.array
    depth: np.array
    time: datetime
    id: str


class CameraStream:
    def __init__(self, cam_id, stopper):
        self.cam_id = cam_id
        self.queue = queue.Queue()
        self.K = None
        self.running = True
        self.stopper = stopper
    def __exit__(self,*_):
        self.running = False
    def stop(self):
        self.stopper()




def multistream(f, cameras = {}):
    streams = []
    threads = []

    with ExitStack() as stack:
        for cam_id, cam_f in cameras.items():
            stream = CameraStream(cam_id, stack.close)
            stack.push(stream)
            thread = threading.Thread(target=cam_f, args=(stream,))
            threads.append(thread)
            streams.append(stream)

        t = threading.Thread(target=f, args=(streams,))
        threads.append(t)

        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
