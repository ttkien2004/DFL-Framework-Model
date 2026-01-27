# How to run scenarios for experiment

## Run experiemnt 1 simulation (Temporarily for testing, I will fix later)

```bash
{
    "scenario": "1",
    "system_mode": "PROPOSED",
    "dataset": "mnist",
    "model": "simple_cnn",
    "num_workers": 10,

    "non_iid_alpha": 0.5,
    "batch_size": 32,
    "learning_rate": 0.01,
    "reset": true
}
```

## Run experiemnt Label Flipping

```bash
curl -X POST http://localhost:5000/run_round \
     -H "Content-Type: application/json" \
     -d '{
           "scenario_id": 4,
           "attack_type": "LABEL_FLIPPING",
           "malicious_ratio": 0.3,
           "distribution": "RANDOM",
           "source_class": 0,
           "target_class": 2
         }'
```

## Run experiment Backdoor

```bash
curl -X POST http://localhost:5000/run_round \
     -H "Content-Type: application/json" \
     -d '{
           "scenario_id": 4,
           "attack_type": "DBA",
           "malicious_ratio": 0.2,
           "distribution": "CLUSTERED",
           "target_label": 0,
           "poison_rate": 0.2,
           "scaling_factor": 5.0
         }'
```

```bash
{
    "system_mode": "PROPOSED",
    "dataset": "mnist",
    "model": "simple_cnn",
    "num_workers": 10,
    "malicious_ratio": 0.3,

    "attack_type": "BACKDOOR",
    "target_class": 5,

    "batch_size": 32,
    "learning_rate": 0.01
}
```

```bash
{
    "system_mode": "BASELINE",
    "dataset": "mnist",
    "model": "simple_cnn",
    "num_workers": 10,
    "malicious_ratio": 0.7,

    "attack_type": "BACKDOOR",
    "target_class": 5,

    "aggregation_algorithm": "UBAR",

    "batch_size": 32,
    "learning_rate": 0.01,
    "reset": true
}
```

## Run experiment Model Poisoning Gauss Injection

```bash
curl -X POST http://localhost:5000/run_round \
     -H "Content-Type: application/json" \
     -d '{
           "scenario_id": 4,
           "attack_type": "MODEL_POISONING",
           "malicious_ratio": 0.3,
           "distribution": "RANDOM",
           "noise_scale": 5.0
         }'
```

```bash
{
    "system_mode": "PROPOSED",
    "dataset": "mnist",
    "model": "simple_cnn",
    "num_workers": 10,
    "malicious_ratio": 0.3,

    "attack_type": "GAUSSIAN",
    "std": 0.5,

    "batch_size": 32,
    "learning_rate": 0.01
}
```

```bash
{
    "system_mode": "BASELINE",
    "dataset": "mnist",
    "model": "simple_cnn",
    "num_workers": 10,
    "malicious_ratio": 0.7,

    "attack_type": "GAUSSIAN_MODEL_POISONING",
    "std": 0.5,

    "aggregation_algorithm": "UBAR",

    "batch_size": 32,
    "learning_rate": 0.01,
    "reset": true
}
```

## Run proposed experiment in Label Flipping Scenario

```bash
{
    "system_mode": "PROPOSED",
    "dataset": "cifar10",
    "model": "resnet18",
    "num_workers": 20,
    "malicious_ratio": 0.3,

    "attack_type": "LABEL_FLIPPING",
    "source_class": 3,
    "target_class": 5,

    "aggregation_algorithm": "DFCA_BALANCE",

    "batch_size": 32,
    "learning_rate": 0.01
}
```

```bash
{
    "system_mode": "BASELINE",
    "dataset": "mnist",
    "model": "simple_cnn",
    "num_workers": 10,
    "malicious_ratio": 0.7,

    "attack_type": "LABEL_FLIPPING",
    "source_class":3,
    "target_class": 5,

    "aggregation_algorithm": "TRIMMED_MEAN",

    "batch_size": 32,
    "learning_rate": 0.01,
    "reset": true
}
```

```bash
{
    "system_mode": "BASELINE",
    "dataset": "mnist",
    "model": "simple_cnn",
    "num_workers": 10,
    "malicious_ratio": 0.7,

    "attack_type": "LABEL_FLIPPING",
    "source_class": 3,
    "target_class": 5,

    "aggregation_algorithm": "UBAR",

    "batch_size": 32,
    "learning_rate": 0.01,

    "reset": true,

    "distribution": "DIRICHLET",  <-- THÊM DÒNG NÀY
    "beta": 0.5                   <-- Mức độ Non-IID (càng nhỏ càng lệch)
}
```

## Run Baseline experiment with FedAvg in Label Flipping Scenario

```bash
{
    "system_mode": "BASELINE",
    "dataset": "cifar10",
    "model": "resnet18",
    "num_workers": 20,
    "malicious_ratio": 0.3,

    "attack_type": "LABEL_FLIPPING",
    "source_class": 3,
    "target_class": 5,

    "aggregation_algorithm": "FED_AVG",

    "batch_size": 32,
    "learning_rate": 0.01
}
```
