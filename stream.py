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
    def __init__(self, cam_id, stopper, queued=True):
        self.cam_id = cam_id
        self.queue = queue.Queue()
        self.K = None
        self.running = True
        self.queued = queued
        self.stopper = stopper

    def __exit__(self,*_):
        self.running = False

    def stop(self):
        self.stopper()

    def put(self, msg):
        self.queue.put(msg)

    def get(self):
        # TODO: check if this works
        if not self.queued:
            try:
                msg = None
                while True:
                    msg = self.queue.get(False)
            except:
                if msg is not None:
                    return msg
        return self.queue.get()



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



def multistream_sync(f, cameras = {}):
    streams = []
    threads = []
    q = queue.Queue()

    def sync(streams, q):
        while all([s.running for s in streams]):
            if not any([s.queue.empty for s in streams]):
                q.put({s.cam_id: s.get() for s in streams})

    with ExitStack() as stack:
        for cam_id, cam_f in cameras.items():
            stream = CameraStream(cam_id, stack.close, queued=False)
            stack.push(stream)
            thread = threading.Thread(target=cam_f, args=(stream,))
            threads.append(thread)
            streams.append(stream)

        t = threading.Thread(target=f, args=(streams,q))
        threads.append(t)
        s = threading.Thread(target=sync, args = (streams,q))
        threads.append(s)
        print(f"Starting {len(streams)} cameras and {len(threads)} threads in total")

        for t in threads:
            t.start()
        
        for t in threads:
            t.join()