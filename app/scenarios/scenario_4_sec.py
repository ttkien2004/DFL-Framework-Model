# app/scenarios/scenario_4_security.py
from app.scenarios.base_scenario import BaseScenario
class ScenarioExperiment4(BaseScenario):
    def setup_security(self, workers):
        attack_type = self.config.get('attack_type', 'LABEL_FLIPPING')
        ratio = self.config.get('malicious_ratio', 0.3)
        num_malicious = int(len(workers) * ratio)
        
        print(f"   -> [Scenario 4] Injecting {attack_type} to {num_malicious} nodes...")
        for i in range(num_malicious):
            workers[i].set_attack_type(attack_type)