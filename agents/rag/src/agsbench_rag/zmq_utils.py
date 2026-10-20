import threading
import multiprocessing
import zmq


class ZMQClient:
    def __init__(self, func, addrs: list, executor: str = "process"):
        self.addrs = addrs
        self.next_id = 0
        self.lock = threading.Lock()
        self.executor = executor
        self.workers = []
        self.launch_servers(func, executor=self.executor)

    def launch_servers(self, func, executor: str = "process"):
        addrs = self.addrs
        if executor == "process":
            for addr in addrs:
                p = multiprocessing.Process(target=func, args=(addr,))
                p.start()
                self.workers.append(p)
        elif executor == "thread":
            for addr in addrs:
                t = threading.Thread(target=func, args=(addr,))
                t.start()
                self.workers.append(t)
        else:
            raise ValueError(f"Unsupported executor: {executor}")

    def next_addr(self):
        with self.lock:
            addr = self.addrs[self.next_id]
            self.next_id = (self.next_id + 1) % len(self.addrs)
        return addr

    def get_socket(self):
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.connect(self.next_addr())
        return socket

    def send_recv_json(self, obj: dict):
        socket = self.get_socket()
        socket.send_json(obj)
        response = socket.recv_json()
        return response
