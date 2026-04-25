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
        if num_updates == 0:
            return [], []

        if 0 < self.num_adversaries < 1:
            num_adversaries = max(0, int(self.num_adversaries * num_updates))
        else:
            num_adversaries = int(self.num_adversaries)

        num_adversaries = min(max(num_adversaries, 0), max(0, num_updates - 2))
        nearest_count = max(1, num_updates - num_adversaries - 2)
        scores = []

        for i, u in enumerate(vectors):
            distances = [torch.norm(u - v).item() for j, v in enumerate(vectors) if j != i]
            distances.sort()
            scores.append(sum(distances[:nearest_count]))

        num_accepted = max(1, num_updates - num_adversaries)
        accepted_indices = np.argsort(scores)[:num_accepted].tolist()

        accepted_updates = [updates[i] for i in accepted_indices]
        return accepted_updates, accepted_indices