import os
import hashlib
import hmac

class VRF:
    def __init__(self):
        self.sk = os.urandom(32)  # Secret key
        self.pk = hashlib.sha256(self.sk).digest()  # Public key (simplified)

    def prove(self, input_data):
        proof = hmac.new(self.sk, input_data, hashlib.sha256).digest()
        output = hashlib.sha256(proof).digest()
        return output, proof

    def verify(self, input_data, pk, output, proof):
        expected_proof = hmac.new(pk, input_data, hashlib.sha256).digest()  # Simplified
        return hmac.compare_digest(proof, expected_proof)

    def select_miners(self, stake_map, input_data, num_requested, total_nodes, exclude_id):
        output, _ = self.prove(input_data)
        lottery = []
        for node_id, stake in stake_map.items():
            lottery.extend([node_id] * stake)
        selected = set()
        i = 0
        while len(selected) < num_requested and i < len(output):
            idx = output[i] % len(lottery)
            winner = lottery[idx]
            if winner not in selected and winner != exclude_id:
                selected.add(winner)
            i += 1
        return list(selected)