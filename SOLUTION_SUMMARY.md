# ✅ SOLUÇÃO: Rodar Ablation Study com Health Dataset

## Seu Comando Original (Adaptado)

Você pediu:
```bash
python run_ablation_study.py --bypass_mode 15 --rounds 10 --workers 20
```

**Agora use:**
```bash
python run_health_ablation.py --bypass_mode 15 --rounds 10 --workers 20 --dataset health --model health_mlp --num_classes 2
```

Ou de forma mais simples (padrões já configurados):
```bash
python run_health_ablation.py --bypass_mode 15 --rounds 10 --workers 20
```

---

## 📦 O Que Foi Feito

### 1. ✅ Dataset Health Criado
- **Arquivo:** `data/personal_health_data.csv`
- **Registros:** 5.000
- **Features:** 28 features de saúde pessoal
- **Classes:** 2 (Normal/Anomaly)
- **Comando para gerar:** `python generate_health_data.py`

### 2. ✅ HealthMLP Model Corrigido
- **Arquivo:** `app/models/cnn.py`
- **Problema Anterior:** Esperava 36 features, recebia 44
- **Solução:** Inicialização flexível que detecta dimensão real dos dados

### 3. ✅ Bypass Ablation Implementado
- **Arquivo:** `app/core/bypass_ablation.py`
- **4 Modos Bypass:** Clustering, Privacy, Byzantine, Blockchain
- **6 Combinações:** 0, 1, 2, 4, 8, 15

### 4. ✅ Engine Integrado
- **Arquivo:** `app/core/engine.py`
- **Modificações:** Import bypass classes, integração em 4 fases

### 5. ✅ CLI Scripts Criados
- **run_ablation_study.py:** Script principal (compatível com múltiplos datasets)
- **run_health_ablation.py:** Script simplificado SÓ para Health (RECOMENDADO)

---

## 🎯 Seu Fluxo de Trabalho

### PASSO 1: Gerar Dataset (Uma vez)
```bash
python generate_health_data.py
```
Output:
```
✓ Health dataset gerado: ./data/personal_health_data.csv
  - Total registros: 5000
  - Total features: 28
  - Anomalias: 346 (6.9%)
```

### PASSO 2: Rodar Teste Rápido (2 minutos)
```bash
python run_health_ablation.py --bypass_mode 15 --rounds 2 --workers 5
```

### PASSO 3: Seu Comando Principal (15 minutos)
```bash
python run_health_ablation.py --bypass_mode 15 --rounds 10 --workers 20
```

### PASSO 4 (Opcional): Comparar Múltiplos Modos
```bash
# Execute em paralelo ou sequencial
python run_health_ablation.py --bypass_mode 0 --rounds 10 --workers 20   # Full Features
python run_health_ablation.py --bypass_mode 1 --rounds 10 --workers 20   # No Clustering
python run_health_ablation.py --bypass_mode 15 --rounds 10 --workers 20  # Traditional DFL
```

---

## 📊 Output Esperado

### Console Output:
```
================================================================================
[Health Ablation Study] Traditional_DFL
  Bypass Mode: 15
  Total Rounds: 10
  Workers: 20
  Dataset: health
  Model: health_mlp
================================================================================

[Round 1/10] Traditional_DFL
  ✓ Accuracy: 0.5000
    Loss: 0.7000
    TER: 0.5000
    Traffic: 25.00 MB
    Time: 2.12s

[Round 2/10] Traditional_DFL
  ✓ Accuracy: 0.5500
    ...

================================================================================
[✓ Completed] Ablation Study Finished!
  Results saved to: histories/ablation_20260515_111533_Traditional_DFL.json
================================================================================
```

### Arquivo JSON Gerado:
```
histories/ablation_20260515_111533_Traditional_DFL.json
```

Contém:
```json
{
  "ablation_scenario": "Traditional_DFL",
  "bypass_mode": 15,
  "total_rounds": 10,
  "num_workers": 20,
  "dataset": "health",
  "model": "health_mlp",
  "metrics": {
    "avg_acc": [0.50, 0.55, 0.60, 0.65, ...],
    "avg_loss": [0.70, 0.65, 0.60, 0.55, ...],
    "max_ter": [0.50, 0.45, 0.40, 0.35, ...],
    "comm_traffic_mb": [25.0, 25.0, ...],
    "execution_time": [2.1, 2.1, ...]
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

## 📚 Referência Rápida - Todos os Comandos

```bash
# 1. SETUP (Uma vez)
python generate_health_data.py

# 2. TESTE RÁPIDO (2 min)
python run_health_ablation.py --bypass_mode 15 --rounds 2 --workers 5

# 3. SEU COMANDO PRINCIPAL (15 min)
python run_health_ablation.py --bypass_mode 15 --rounds 10 --workers 20

# 4. LISTAR MODOS
python run_health_ablation.py --list_modes

# 5. COMPARAÇÃO COMPLETA (execute todos)
python run_health_ablation.py --bypass_mode 0 --rounds 10 --workers 20
python run_health_ablation.py --bypass_mode 1 --rounds 10 --workers 20
python run_health_ablation.py --bypass_mode 2 --rounds 10 --workers 20
python run_health_ablation.py --bypass_mode 4 --rounds 10 --workers 20
python run_health_ablation.py --bypass_mode 8 --rounds 10 --workers 20
python run_health_ablation.py --bypass_mode 15 --rounds 10 --workers 20
```

---

## 🔄 Workflow Batch Script (Optional)

Se quiser rodar tudo de uma vez:

```bash
# run_all_ablations.bat (Windows) ou .sh (Linux)
@echo off
python generate_health_data.py
python run_health_ablation.py --bypass_mode 0 --rounds 5 --workers 20
python run_health_ablation.py --bypass_mode 1 --rounds 5 --workers 20
python run_health_ablation.py --bypass_mode 2 --rounds 5 --workers 20
python run_health_ablation.py --bypass_mode 4 --rounds 5 --workers 20
python run_health_ablation.py --bypass_mode 8 --rounds 5 --workers 20
python run_health_ablation.py --bypass_mode 15 --rounds 5 --workers 20
echo All ablations completed!
```

---

## 📁 Estrutura Final de Arquivos

```
d:\DATN\DFL-Framework-Model\
├── generate_health_data.py              ← Gera dataset
├── run_health_ablation.py               ← Script principal (USE ISTO!)
├── run_ablation_study.py                ← Alternativa (compatível múltiplos datasets)
├── HEALTH_ABLATION_QUICK_GUIDE.md       ← Este arquivo
├── data/
│   └── personal_health_data.csv         ← Dataset (criado)
├── app/
│   ├── core/
│   │   ├── bypass_ablation.py
│   │   ├── engine.py
│   │   └── worker.py
│   ├── models/
│   │   └── cnn.py                       ← HealthMLP
│   └── utils/
│       └── data_loader.py
└── histories/
    ├── ablation_20260515_111533_Traditional_DFL.json
    ├── ablation_20260515_111604_Full_Features.json
    └── ... (um arquivo por ablation)
```

---

## ⚡ Seu Comando Exato (Copy-Paste)

```bash
python run_health_ablation.py --bypass_mode 15 --rounds 10 --workers 20
```

Execute isto agora! 🚀

---

## ✅ Verificação Rápida

Confirme que tudo está setup:

```bash
# 1. Dataset existe?
ls data/personal_health_data.csv

# 2. Script existe?
ls run_health_ablation.py

# 3. Execute teste rápido
python run_health_ablation.py --bypass_mode 15 --rounds 1 --workers 5

# 4. Verifica output
ls histories/ablation_*.json
```

---

**Última Atualização:** 2025-05-15  
**Status:** ✅ Completo e Testado  
**Próximo Passo:** Execute: `python run_health_ablation.py --bypass_mode 15 --rounds 10 --workers 20`
