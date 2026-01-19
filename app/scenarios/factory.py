# app/scenarios/factory.py
from app.scenarios.scenario_1_baseline import ScenarioExperiment1
from app.scenarios.scenario_3_hetero import ScenarioExperiment3
from app.scenarios.scenario_4_sec import ScenarioExperiment4

class ScenarioFactory:
    @staticmethod
    def get_runner(scenario_id, config):
        if scenario_id == 1:
            return ScenarioExperiment1(config)
        elif scenario_id == 3:
            return ScenarioExperiment3(config)
        elif scenario_id == 4:
            return ScenarioExperiment4(config)
        else:
            return ScenarioExperiment1(config)