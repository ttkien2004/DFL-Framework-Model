import zmq

class NodeCommunicator:
    def __init__(self, node_id, peers):
        self.node_id = node_id
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{8000 + node_id}")
        self.peers = peers

    def send_update(self, update):
        # Send to all peers
        for peer in self.peers:
            socket = self.context.socket(zmq.REQ)
            socket.connect(f"tcp://{peer}:{8000 + peer}")
            socket.send_pyobj(update)
            socket.recv()  # Ack

    def receive_updates(self):
        updates = []
        while True:
            try:
                update = self.socket.recv_pyobj(flags=zmq.NOBLOCK)
                updates.append(update)
                self.socket.send(b"ACK")
            except zmq.Again:
                break
        return updates