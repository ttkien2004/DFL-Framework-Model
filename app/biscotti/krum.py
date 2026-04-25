import numpy as np
import torch

class KRUM:
    def __init__(self, num_adversaries=0.5):
        self.num_adversaries = num_adversaries

    def state_dict_to_vector(self, state_dict):
        return torch.cat([param.flatten() for param in state_dict.values()])

    def validate(self, updates):
        vectors = [self.state_dict_to_vector(update) for update in updates]
        num_updates = len(vectors)
        scores = []
        for i, u in enumerate(vectors):
            distances = [torch.norm(u - v) for j, v in enumerate(vectors) if j != i]
            scores.append(sum(distances).item())
        
        # Lấy index của các cập nhật được chấp nhận
        num_accepted = num_updates - int(self.num_adversaries * num_updates)
        accepted_indices = np.argsort(scores)[:num_accepted].tolist()
        
        # Trả về cả weights và indices
        accepted_updates = [updates[i] for i in accepted_indices]
        return accepted_updates, accepted_indices