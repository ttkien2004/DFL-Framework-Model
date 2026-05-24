# 🎯 Guia Rápido - Ablation Study com Health Dataset

## ✅ Tudo Pronto!

Dataset gerado: **5.000 registros** com 28 features de saúde pessoal

---

## 🚀 Seu Comando Principal (Recomendado)

```bash
python run_health_ablation.py --bypass_mode 15 --rounds 10 --workers 20
```

### O que isso faz:
- Executa ablation study com **Traditional DFL** (todos os components desabilitados)
- 10 rodadas de treinamento
- 20 workers
- Dataset: Health Personal
- Modelo: HealthMLP

---

## 📊 Comparar Múltiplos Modos

Execute estes 6 comandos para comparação completa:

```bash
# 1. Full Features (CoCo + LDP + BALANCE + Blockchain)
python run_health_ablation.py --bypass_mode 0 --rounds 5 --workers 20

# 2. No Clustering
python run_health_ablation.py --bypass_mode 1 --rounds 5 --workers 20

# 3. No Privacy (sem LDP/SSS)
python run_health_ablation.py --bypass_mode 2 --rounds 5 --workers 20

# 4. No Byzantine (FedAvg)
python run_health_ablation.py --bypass_mode 4 --rounds 5 --workers 20

# 5. No Blockchain (RAM storage)
python run_health_ablation.py --bypass_mode 8 --rounds 5 --workers 20

# 6. Traditional DFL (seu comando)
python run_health_ablation.py --bypass_mode 15 --rounds 5 --workers 20
```

---

## ⚡ Teste Rápido (2 min)

```bash
python run_health_ablation.py --bypass_mode 15 --rounds 2 --workers 5
```

---

## 📋 Listar Todos os Modos

```bash
python run_health_ablation.py --list_modes
```

Output:
```
Available Bypass Modes for Health Ablation Study:
────────────────────────────────────────────────────────────────────────────────
   0: Full_Features - All features enabled (CoCo + LDP + BALANCE + Blockchain)
   1: No_Clustering - Single cluster (traditional DFL)
   2: No_Privacy - No LDP/SSS (clean gradients)
   4: No_Byzantine - FedAvg instead of BALANCE
   8: No_Blockchain - RAM storage instead of blockchain
  15: Traditional_DFL - All features disabled (baseline)
────────────────────────────────────────────────────────────────────────────────
```

---

## 📁 Outputs

Cada execução gera um arquivo JSON em: `histories/ablation_[timestamp]_[scenario_name].json`

Exemplo:
```
histories/ablation_20260515_111533_Traditional_DFL.json
```

### Conteúdo do JSON:
```json
{
  "ablation_scenario": "Traditional_DFL",
  "bypass_mode": 15,
  "total_rounds": 10,
  "num_workers": 20,
  "dataset": "health",
  "model": "health_mlp",
  "metrics": {
    "avg_acc": [0.50, 0.55, 0.60, ...],
    "avg_loss": [0.70, 0.65, 0.60, ...],
    "max_ter": [0.50, 0.45, 0.40, ...],
    "comm_traffic_mb": [25.0, 25.0, 25.0, ...],
    "execution_time": [2.1, 2.1, 2.1, ...]
  },
  "bypass_report": {
    "mode": "Traditional_DFL",
    "clustering_enabled": false,
    "privacy_enabled": false,
    "byzantine_enabled": false,
    "blockchain_enabled": false
  }
}
```

---

## 🔍 Análise de Resultados

### Ver todos os arquivos gerados:
```bash
ls -la histories/ablation_*.json
```

### Carregar e analisar em Python:
```python
import json

with open("histories/ablation_20260515_111533_Traditional_DFL.json") as f:
    data = json.load(f)

# Ver cenário
print(f"Scenario: {data['ablation_scenario']}")

# Ver accuracy final
final_acc = data['metrics']['avg_acc'][-1]
print(f"Final Accuracy: {final_acc:.4f}")

# Ver média de accuracy
avg_acc = sum(data['metrics']['avg_acc']) / len(data['metrics']['avg_acc'])
print(f"Average Accuracy: {avg_acc:.4f}")
```

---

## 📈 Workflow Recomendado

### Passo 1: Teste Rápido (Validar Setup)
```bash
python run_health_ablation.py --bypass_mode 15 --rounds 2 --workers 5
```

### Passo 2: Rodar Modo Principal
```bash
python run_health_ablation.py --bypass_mode 15 --rounds 10 --workers 20
```

### Passo 3: Comparar com Full Features
```bash
python run_health_ablation.py --bypass_mode 0 --rounds 10 --workers 20
```

### Passo 4: Análise Comparativa
```python
# compare.py
import json

results = {}
for mode in [0, 1, 2, 4, 8, 15]:
    # Encontre o arquivo correspondente em histories/
    with open(f"histories/ablation_*_{mode}_*.json") as f:
        data = json.load(f)
        scenario = data['ablation_scenario']
        final_acc = data['metrics']['avg_acc'][-1]
        results[scenario] = final_acc

for scenario, acc in sorted(results.items()):
    print(f"{scenario:20s}: {acc:.4f}")
```

---

## 💡 Interpretação dos Resultados

| Métrica             | Significado                                     |
| ------------------- | ----------------------------------------------- |
| **avg_acc**         | Acurácia média (0.0-1.0) - quanto maior, melhor |
| **avg_loss**        | Perda média - quanto menor, melhor              |
| **max_ter**         | Taxa de erro máximo - quanto menor, melhor      |
| **comm_traffic_mb** | Tráfego de comunicação - quanto menor, melhor   |
| **execution_time**  | Tempo de execução - quanto menor, melhor        |

### Comparação Esperada:

- **Mode 0 vs 15**: Mode 0 melhor accuracy (tem LDP + BALANCE)
- **Mode 1 vs 15**: Mode 1 menos tráfego (1 cluster vs K clusters)
- **Mode 2 vs 15**: Mode 2 melhor accuracy (sem noise de privacy)
- **Mode 4 vs 15**: Mode 4 mais rápido (FedAvg vs BALANCE)
- **Mode 8 vs 15**: Mode 8 mais rápido (RAM vs Blockchain)

---

## 🐛 Troubleshooting

### Q: "Health dataset not found"
```bash
python generate_health_data.py
```

### Q: "Muito lento"
Use menos rounds/workers:
```bash
python run_health_ablation.py --bypass_mode 15 --rounds 3 --workers 10
```

### Q: "CUDA out of memory"
Reduz batch_size (edit `run_health_ablation.py` linha 30)

### Q: "Arquivo JSON não criado"
Verifica se pasta `histories/` existe:
```bash
mkdir -p histories
```

---

## 📚 Estrutura de Arquivos

```
d:\DATN\DFL-Framework-Model\
├── run_health_ablation.py          ← Script principal (use isto!)
├── generate_health_data.py         ← Gera dataset
├── app/
│   ├── core/
│   │   ├── bypass_ablation.py     ← Lógica bypass
│   │   ├── engine.py              ← Engine principal
│   │   └── worker.py              ← Workers
│   ├── models/
│   │   └── cnn.py                 ← HealthMLP model
│   └── utils/
│       └── data_loader.py         ← Dataset loader
├── data/
│   └── personal_health_data.csv   ← Dataset (criado por generate_health_data.py)
└── histories/
    ├── ablation_20260515_111533_Traditional_DFL.json
    ├── ablation_20260515_111604_Full_Features.json
    └── ...
```

---

## ✅ Checklist

- [x] Dataset gerado: `data/personal_health_data.csv` (5.000 registros)
- [ ] Teste Rápido: `python run_health_ablation.py --bypass_mode 15 --rounds 2 --workers 5`
- [ ] Modo Principal: `python run_health_ablation.py --bypass_mode 15 --rounds 10 --workers 20`
- [ ] Comparação: Rodar 6 modos diferentes
- [ ] Análise: Verificar resultados em `histories/`

---

## 🎓 O que Significa Bypass Mode 15?

**Tradicional DFL** = Todas as features desabilitadas:
- ❌ Sem CoCo Clustering (tudo em 1 cluster)
- ❌ Sem LDP Privacy (gradientes limpos)
- ❌ Sem BALANCE Byzantine (usa FedAvg simples)
- ❌ Sem Blockchain Consensus (RAM storage)

= Federated Learning tradicional, sem inovações

---

**Criado:** 2025-05-15  
**Status:** ✅ Pronto para usar  
**Dataset:** Health Personal (5.000 registros, 28 features)  
**Modelos:** HealthMLP  
**Bypass Modes:** 6 (0, 1, 2, 4, 8, 15)
