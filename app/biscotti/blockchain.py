import hashlib
import time

class Block:
    def __init__(self, data, prev_hash, stake_map):
        self.timestamp = time.time()
        self.data = data  # Dict: {'iteration': int, 'global_weights': dict (state_dict), 'accepted_updates': list}
        self.prev_hash = prev_hash
        self.stake_map = stake_map  # Dict: {node_id: stake}
        self.hash = self.compute_hash()

    def compute_hash(self):
        block_string = f"{self.timestamp}{self.data}{self.prev_hash}{self.stake_map}"
        return hashlib.sha256(block_string.encode()).hexdigest()

class Blockchain:
    def __init__(self):
        self.chain = [self.create_genesis_block()]

    def create_genesis_block(self):
        return Block({'iteration': 0, 'global_weights': {}, 'accepted_updates': []}, "0", {})

    def add_block(self, data, stake_map):
        prev_block = self.chain[-1]
        new_block = Block(data, prev_block.hash, stake_map)
        self.chain.append(new_block)
        return new_block

    def get_latest_global_weights(self):
        return self.chain[-1].data['global_weights']