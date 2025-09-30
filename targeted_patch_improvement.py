#!/usr/bin/env python3
"""Test to improve patch success rate through targeted evolution."""

from pathlib import Path
import json


def analyze_current_metrics():
    """Analyze current metrics to understand the patch issue."""
    print("🔍 ANALISANDO MÉTRICAS ATUAIS...")
    
    metrics_path = Path("seed/metrics/history.json")
    if not metrics_path.exists():
        print("❌ Arquivo de métricas não encontrado!")
        return None
    
    with open(metrics_path, 'r') as f:
        data = json.load(f)
    
    # Show recent metrics
    print("📊 MÉTRICAS RECENTES:")
    metrics_of_interest = [
        'apply_patch_count',
        'apply_patch_success_rate',
        'failures',
        'actions_success_rate'
    ]
    
    for metric in metrics_of_interest:
        if metric in data and len(data[metric]) > 0:
            recent_values = data[metric][-5:]  # Last 5 values
            avg_value = sum(recent_values) / len(recent_values)
            print(f"  {metric}: {recent_values} (média: {avg_value:.3f})")
    
    return data


def create_targeted_seed():
    """Create a targeted seed to improve patch success rate."""
    print("\n🎯 CRIANDO SEED DIRECIONADO PARA MELHORIA DE PATCH SUCCESS RATE...")
    
    seed_content = {
        "id": "targeted_patch_improvement_20250930",
        "goal": "Melhorar taxa de sucesso na aplicação de patches (apply_patch_success_rate)",
        "priority": "high",
        "type": "evolution",
        "config": "configs/seed_manual.yaml",
        "metadata": {
            "description": "Seed direcionado para resolver o problema persistente de apply_patch_success_rate = 0.0",
            "created_by": "targeted_evolution",
            "tags": ["patch_optimization", "success_rate_improvement", "core.diffing"]
        },
        "history": [],
        "attempts": 0,
        "max_attempts": 5,
        "next_run_at": None,
        "last_error": None
    }
    
    # Save to backlog
    backlog_path = Path("seed/backlog.yaml")
    import yaml
    
    backlog_entries = []
    if backlog_path.exists():
        with open(backlog_path, 'r') as f:
            content = f.read().strip()
            if content:
                backlog_entries = list(yaml.safe_load_all(content))
                backlog_entries = [entry for entry in backlog_entries if entry is not None]
    
    backlog_entries.append(seed_content)
    
    with open(backlog_path, 'w') as f:
        yaml.dump(backlog_entries, f, default_flow_style=False, allow_unicode=True, indent=2)
    
    print("✅ Seed criado e adicionado ao backlog!")
    return seed_content


def run_targeted_improvement():
    """Run targeted improvement for patch success rate."""
    print("🚀 INICIANDO MELHORIA DIRECIONADA PARA PATCH SUCCESS RATE")
    print("=" * 60)
    
    # Analyze current state
    metrics_data = analyze_current_metrics()
    
    if metrics_data:
        patch_success_rate = metrics_data.get('apply_patch_success_rate', [0.0])
        if len(patch_success_rate) > 0:
            current_rate = patch_success_rate[-1]
            print(f"\n📉 TAXA ATUAL DE SUCESSO EM PATCHES: {current_rate:.3f}")
            
            if current_rate < 0.5:
                print("⚠️  Taxa crítica! Necessário intervenção imediata!")
                seed = create_targeted_seed()
                print(f"\n🌱 Seed criado: {seed['id']}")
                print(f"   Objetivo: {seed['goal']}")
                print(f"   Prioridade: {seed['priority']}")
            else:
                print("✅ Taxa aceitável, continuando monitoramento...")
        else:
            print("❌ Sem dados de patch success rate disponíveis!")
    else:
        print("❌ Não foi possível analisar métricas!")


if __name__ == "__main__":
    run_targeted_improvement()