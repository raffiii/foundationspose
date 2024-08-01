import cv2
import numpy as np
#import hololabeling.video2label
from hololabeling.drifter import LabelingServer
import queue

class LabelQueue:
    def __init__(self):
        self.queue = queue.Queue(maxsize=10)

    def create_dir(self, _):
        pass

    def save_file(self, filename, content):
        self.queue.put((filename, content))

def run_with_labeling_server(f):
    labelqueue = LabelQueue()
    server = LabelingServer(
        file_saver=labelqueue,
        host="127.0.0.1", 
        port=8053
    )
    try:
        server.start_server_thread(block=False)
        f(labelqueue.queue)
    finally:
        server.close_server()

if __name__ == "__main__":
    def print_all(q):
        while True:
            if q.not_empty:
                print(q.get())
    run_with_labeling_server(print_all)