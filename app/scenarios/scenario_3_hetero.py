# app/scenarios/scenario_3_hetero.py
from app.scenarios.base_scenario import BaseScenario
class ScenarioExperiment3(BaseScenario):
    def setup_data(self, workers, dataset_name):
        # Chia IID bình thường (hoặc Non-IID tùy ý)
        pass 

    def setup_network(self, workers):
        print("   -> [Scenario 3] Setting up Dynamic Bandwidth...")
        import random
        # Gán 30% node mạng yếu, 70% node mạng mạnh
        for w in workers:
            if random.random() < 0.3:
                w.bandwidth = random.uniform(1, 5) # 3G
            else:
                w.bandwidth = random.uniform(50, 100) # 5G