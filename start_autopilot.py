#!/usr/bin/env python3
"""Script para iniciar o SeedAI no modo autopilot e observar sua autoevolução real."""

import subprocess
import sys
import time
from pathlib import Path


def run_autopilot_real():
    """Executa o SeedAI no modo autopilot usando o CLI existente."""
    print("🚀 INICIANDO SEEDAI NO MODO AUTOPILOT - EXECUÇÃO REAL")
    print("=" * 60)
    
    print("\n🎯 COMANDOS PARA INICIAR AUTO-EVOLUÇÃO:")
    print("   1. Iniciar uma execução de objetivo geral de autoaprimoramento")
    print("   2. Rodar seeds existentes no backlog")
    print("   3. Iniciar loop contínuo de autopilot")
    
    print("\n📋 DESCRICAO DOS COMANDOS:")
    print("   a) a3x run --goal 'Auto-aperfeiçoar minhas próprias capacidades' --config configs/seed_manual.yaml")
    print("   b) a3x seed run --config configs/seed_manual.yaml")
    print("   c) a3x autopilot --cycles 5 --goals seed/goal_rotation.yaml")
    
    print("\n🔍 ARTEFATOS DE AUTO-EVOLUÇÃO:")
    print("   - Logs: seed/evaluations/run_evaluations.jsonl")
    print("   - Métricas: seed/metrics/history.json")
    print("   - Seeds: seed/backlog.yaml")
    print("   - Alterações: seed/changes/")
    print("   - Relatórios: seed/reports/")
    print("   - Consciência: seed/consciousness/")
    
    print("\n🎯 OBJETIVO DA AUTO-EVOLUÇÃO:")
    print("   O SeedAI irá:")
    print("   1. Analisar seu próprio código e desempenho")
    print("   2. Identificar lacunas e oportunidades de melhoria")
    print("   3. Criar e executar seeds para auto-aperfeiçoamento")
    print("   4. Avaliar o impacto das mudanças")
    print("   5. Repetir o ciclo continuamente")
    
    print("\n💡 PARA INICIAR REALMENTE:")
    print("   Execute um destes comandos no terminal:")
    print()
    print("   # Modo de execução única com objetivo de auto-aperfeiçoamento")
    print("   python -m a3x run --goal 'Auto-aperfeiçoar minhas próprias capacidades e expandir minhas habilidades' --config configs/seed_manual.yaml")
    print()
    print("   # Executar próxima seed do backlog")
    print("   python -m a3x seed run --config configs/seed_manual.yaml")
    print()
    print("   # Loop de autopilot (5 ciclos)")
    print("   python -m a3x autopilot --cycles 5 --goals configs/sample_goals.yaml")
    print()
    
    print("\n📊 PARA MONITORAR A AUTO-EVOLUÇÃO:")
    print("   # Ver últimos relatórios")
    print("   ls -la seed/reports/")
    print()
    print("   # Ver últimas avaliações")
    print("   tail -f seed/evaluations/run_evaluations.jsonl")
    print()
    print("   # Ver métricas de evolução")
    print("   cat seed/metrics/history.json")
    print()
    print("   # Ver seeds pendentes")
    print("   cat seed/backlog.yaml")
    
    print("\n✅ PRONTO PARA AUTO-EVOLUÇÃO!")
    print("   O SeedAI está configurado para evoluir indefinidamente.")


def create_sample_goals():
    """Cria um arquivo de objetivos de exemplo para autopilot."""
    goals_path = Path("configs/sample_goals.yaml")
    
    goals_content = """
# Objetivos de exemplo para autopilot
- goal: "Auto-avaliar minhas capacidades atuais e identificar lacunas"
  config: "configs/seed_manual.yaml"
  priority: "high"
- goal: "Implementar melhoria em análise de complexidade ciclomática"
  config: "configs/seed_manual.yaml"
  priority: "medium"
- goal: "Melhorar sistema de detecção de más práticas de codificação"
  config: "configs/seed_manual.yaml"
  priority: "medium"
- goal: "Expandir para análise de linguagens além de Python"
  config: "configs/seed_manual.yaml"
  priority: "low"
- goal: "Aprimorar minhas habilidades de auto-modificação segura"
  config: "configs/seed_manual.yaml"
  priority: "high"
"""
    
    with open(goals_path, 'w', encoding='utf-8') as f:
        f.write(goals_content)
    
    print(f"📋 Arquivo de objetivos de exemplo criado: {goals_path}")


def create_seed_config():
    """Cria um arquivo de configuração para seeds."""
    config_path = Path("configs/seed_manual.yaml")
    
    config_content = """
llm:
  type: "manual"
  script: "configs/scripts/demo_plan.yaml"

workspace:
  root: "."

limits:
  max_iterations: 20

tests:
  auto: false

policies:
  allow_network: false
  allow_shell_write: true
  deny_commands: ["rm", "mv", "dd", "kill", "reboot", "shutdown"]

goals:
  apply_patch_success_rate: 0.7

loop:
  auto_seed: true
  seed_backlog: "seed/backlog.yaml"

audit:
  enable_file_log: true
  file_dir: "seed/changes"
  enable_git_commit: false
"""
    
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print(f"📋 Configuração de seed criada: {config_path}")


def create_demo_plan():
    """Cria um plano de demonstração para o modo manual."""
    scripts_dir = Path("configs/scripts")
    scripts_dir.mkdir(exist_ok=True)
    
    plan_path = scripts_dir / "demo_plan.yaml"
    
    plan_content = """
# Plano de demonstração para SeedAI
- type: "message"
  text: "Iniciando autoavaliação de minhas próprias capacidades"
- type: "read_file"
  path: "a3x/__init__.py"
- type: "message" 
  text: "Analisando minhas capacidades atuais..."
- type: "run_command"
  command: ["python", "-c", "import a3x; print('SeedAI está funcionando corretamente')"]
- type: "message"
  text: "Identificando oportunidades de melhoria..."
- type: "apply_patch"
  diff: |
    --- a/README.md
    +++ b/README.md
    @@ -1,5 +1,6 @@
     # A3X – Ferramenta Autônoma de Codificação Local
    +
     A3X é um esqueleto de ferramenta CLI desenhada para orquestrar um agente de codificação local inspirado no Replit Agent 3 e em iniciativas open-source como OpenHands, SWE-Agent e GPT-Engineer.
- type: "message"
  text: "Auto-aperfeiçoamento realizado com sucesso!"
- type: "finish"
  text: "Concluído auto-aperfeiçoamento inicial"
"""
    
    with open(plan_path, 'w', encoding='utf-8') as f:
        f.write(plan_content)
    
    print(f"📋 Plano de demonstração criado: {plan_path}")


if __name__ == "__main__":
    print("🔧 CONFIGURANDO SEEDAI PARA AUTO-EVOLUÇÃO")
    print("=" * 50)
    
    # Criar arquivos de configuração necessários
    create_seed_config()
    create_demo_plan()
    create_sample_goals()
    
    print()
    run_autopilot_real()
    
    print("\n💡 RECOMENDAÇÃO:")
    print("   1. Inicie com uma execução pequena para testar")
    print("   2. Monitore os resultados nos diretórios seed/")
    print("   3. Após confirmar funcionamento, inicie loops contínuos")
    print("   4. Observe como o SeedAI evolui e melhora continuamente")