# Rodar Ablation Study com Health Personal Dataset

## ✅ Dataset Pronto
- **Arquivo:** `data/personal_health_data.csv`
- **Total Registros:** 5.000
- **Features:** 28 features de saúde pessoal
- **Classes:** 2 (Anomaly: 0/1)
- **Anomalias:** 346 registros (6.9%)

---

## 🚀 Comando Principal (Seu Comando)

```bash
python run_ablation_study.py --bypass_mode 15 --rounds 10 --workers 20 --dataset health --model health_mlp --num_classes 2
```

### Explicação dos Parâmetros:
- `--bypass_mode 15`: Traditional DFL (todas features desabilitadas)
- `--rounds 10`: 10 rodadas de treinamento
- `--workers 20`: 20 workers
- `--dataset health`: Dataset Health Personal
- `--model health_mlp`: Model HealthMLP (especial para dados tabulares)
- `--num_classes 2`: 2 classes (Normal/Anomaly)

---

## 📊 Outras Combinations para Comparação

### 1. **Full Features (CoCo + LDP + BALANCE + Blockchain)**
```bash
python run_ablation_study.py --bypass_mode 0 --rounds 10 --workers 20 --dataset health --model health_mlp --num_classes 2
```

### 2. **No Clustering (Single Cluster)**
```bash
python run_ablation_study.py --bypass_mode 1 --rounds 10 --workers 20 --dataset health --model health_mlp --num_classes 2
```

### 3. **No Privacy (No LDP/SSS)**
```bash
python run_ablation_study.py --bypass_mode 2 --rounds 10 --workers 20 --dataset health --model health_mlp --num_classes 2
```

### 4. **No Byzantine (FedAvg instead BALANCE)**
```bash
python run_ablation_study.py --bypass_mode 4 --rounds 10 --workers 20 --dataset health --model health_mlp --num_classes 2
```

### 5. **No Blockchain (RAM Storage)**
```bash
python run_ablation_study.py --bypass_mode 8 --rounds 10 --workers 20 --dataset health --model health_mlp --num_classes 2
```

---

## 🎯 Teste Rápido (3 rounds, 5 workers)

```bash
python run_ablation_study.py --bypass_mode 15 --rounds 3 --workers 5 --dataset health --model health_mlp --num_classes 2
```

---

## 🔧 Otros Parâmetros Opcionais

```bash
python run_ablation_study.py \
  --bypass_mode 15 \
  --rounds 10 \
  --workers 20 \
  --dataset health \
  --model health_mlp \
  --num_classes 2 \
  --batch_size 32 \
  --learning_rate 0.01 \
  --non_iid_alpha 0.5
```

### Descrição dos Parâmetros Adicionais:
- `--batch_size 32`: Tamanho do batch (padrão: 32)
- `--learning_rate 0.01`: Taxa de aprendizado (padrão: 0.01)
- `--non_iid_alpha 0.5`: Parâmetro Dirichlet para dados não-IID (padrão: 0.5)
  - `0.5`: Altamente não-IID (heterogêneo)
  - `1.0`: Moderadamente não-IID
  - `10.0`: Próximo ao IID (homogêneo)

---

## 📊 Saída Esperada

### Arquivo de Resultado:
```
histories/ablation_20250515_143022_Traditional_DFL.json
```

### Conteúdo do JSON:
```json
{
  "ablation_scenario": "Traditional_DFL",
  "bypass_mode": 15,
  "total_rounds": 10,
  "config": {
    "dataset": "health",
    "model": "health_mlp",
    "num_workers": 20,
    ...
  },
  "metrics": {
    "avg_acc": [0.50, 0.62, 0.71, ...],
    "avg_loss": [0.69, 0.58, 0.48, ...],
    "max_ter": [0.50, 0.38, 0.29, ...],
    "execution_time": [5.2, 5.1, 5.3, ...],
    "comm_traffic_mb": [12.5, 12.5, 12.5, ...]
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

## 🔍 Listar Todos os Modos Disponíveis

```bash
python run_ablation_study.py --list_modes
```

Output:
```
Available Bypass Modes:
────────────────────────────────────────────────────────────────────────────────
   0: Full_Features - All features enabled (CoCo + LDP + BALANCE + Blockchain)
   1: No_Clustering - Single cluster (traditional DFL)
   2: No_Privacy - No LDP/SSS (clean gradients)
   4: No_Byzantine - FedAvg instead of BALANCE
   8: No_Blockchain - RAM storage instead of blockchain
  15: Traditional_DFL - All features disabled
────────────────────────────────────────────────────────────────────────────────
```

---

## 💡 Workflow Recomendado

### Passo 1: Teste Rápido (2-3 min)
```bash
python run_ablation_study.py --bypass_mode 15 --rounds 3 --workers 5 --dataset health --model health_mlp --num_classes 2
```

### Passo 2: Rodar Modo Principal (5-10 min)
```bash
python run_ablation_study.py --bypass_mode 15 --rounds 10 --workers 20 --dataset health --model health_mlp --num_classes 2
```

### Passo 3: Comparação (Rodar 2-3 modos diferentes)
```bash
python run_ablation_study.py --bypass_mode 0 --rounds 10 --workers 20 --dataset health --model health_mlp --num_classes 2
python run_ablation_study.py --bypass_mode 15 --rounds 10 --workers 20 --dataset health --model health_mlp --num_classes 2
```

### Passo 4: Análise de Resultados
```python
# analyze_ablation.py
import json
import glob

# Carregar todos os resultados
for file in glob.glob("histories/ablation_*.json"):
    with open(file) as f:
        data = json.load(f)
    
    scenario = data['ablation_scenario']
    final_acc = data['metrics']['avg_acc'][-1]
    avg_time = sum(data['metrics']['execution_time']) / len(data['metrics']['execution_time'])
    
    print(f"{scenario:20s} | Acc: {final_acc:.4f} | Time: {avg_time:.2f}s")
```

---

## 🐛 Troubleshooting

### Q: ImportError: No module named 'torch'
**A:** Instale as dependências:
```bash
pip install -r requirements.txt
```

### Q: FileNotFoundError: personal_health_data.csv
**A:** Gere o dataset primeiro:
```bash
python generate_health_data.py
```

### Q: CUDA out of memory
**A:** Reduza workers ou batch_size:
```bash
python run_ablation_study.py --bypass_mode 15 --rounds 10 --workers 10 --batch_size 16 --dataset health --model health_mlp --num_classes 2
```

### Q: Muito lento
**A:** Use menos rounds e workers para teste:
```bash
python run_ablation_study.py --bypass_mode 15 --rounds 3 --workers 5 --dataset health --model health_mlp --num_classes 2
```

---

## 📋 Checklist

- [x] Dataset gerado: `data/personal_health_data.csv`
- [ ] Teste rápido: `--rounds 3 --workers 5`
- [ ] Ablação completa: `--rounds 10 --workers 20`
- [ ] Comparação múltiplos modos
- [ ] Análise de resultados em `histories/`

---

**Criado:** 2025-05-15  
**Dataset:** Health Personal (5.000 registros, 28 features)  
**Status:** ✅ Pronto para ablation study
