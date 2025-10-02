#!/usr/bin/env python3
"""Script para iniciar o SeedAI em modo completamente autônomo para autoevolução."""

import sys
from pathlib import Path

# Adiciona o diretório do projeto ao path
sys.path.insert(0, str(Path(__file__).parent))

from a3x.seed_runner import main as seed_main


def start_autonomous_evolution():
    """Inicia o processo de evolução autônoma do SeedAI."""
    print("🚀 INICIANDO AUTO-EVOLUÇÃO AUTÔNOMA DO SEEDAI")
    print("=" * 60)

    # Verificar se temos seeds pendentes
    backlog_path = Path("seed/backlog.yaml")

    if not backlog_path.exists():
        print(f"⚠️  Arquivo de backlog não encontrado: {backlog_path}")
        print("   Criando backlog inicial...")
        backlog_path.parent.mkdir(parents=True, exist_ok=True)
        backlog_content = """# Backlog de seeds para autoevolução do SeedAI
- id: initial.self_awareness
  goal: "Analisar minhas próprias capacidades e identificar oportunidades de auto-aperfeiçoamento"
  priority: high
  type: evolution
  config: configs/seed_manual.yaml
  metadata:
    description: "Seed inicial para autoavaliação e autoevolução"
    created_by: "initial_setup"
    tags: ["self_awareness", "evolution", "optimization"]
  history: []
  attempts: 0
  max_attempts: 3
  next_run_at: null
  last_error: null
"""
        backlog_path.write_text(backlog_content)
        print(f"✅ Backlog inicial criado: {backlog_path}")

    print(f"\n📊 VERIFICANDO BACKLOG EM: {backlog_path}")
    backlog_content = backlog_path.read_text()
    print(f"Conteúdo atual do backlog:\n{backlog_content}")

    print("\n📂 VERIFICANDO CONFIGURAÇÃO EM: configs/seed_manual.yaml")
    if Path("configs/seed_manual.yaml").exists():
        config_content = Path("configs/seed_manual.yaml").read_text()
        print("Configuração encontrada:")
        print(config_content[:500] + "..." if len(config_content) > 500 else config_content)
    else:
        print("❌ Arquivo de configuração não encontrado!")

    print("\n🔍 VERIFICANDO SCRIPT EM: configs/scripts/demo_plan.yaml")
    if Path("configs/scripts/demo_plan.yaml").exists():
        script_content = Path("configs/scripts/demo_plan.yaml").read_text()
        print("Script encontrado:")
        print(script_content[:500] + "..." if len(script_content) > 500 else script_content)
    else:
        print("❌ Arquivo de script não encontrado!")

    print("\n🎯 INICIANDO PRIMEIRA SEED DO BACKLOG...")
    print("(Esta execução iniciará o SeedAI no modo autônomo)")

    try:
        # Executar o seed runner com sys.argv
        sys.argv = ["seed_runner", "--backlog", str(backlog_path), "--config", "configs/seed_manual.yaml"]
        result_code = seed_main(sys.argv[1:])

        print(f"\n✅ EXECUÇÃO CONCLUÍDA COM CÓDIGO: {result_code}")

        return result_code

    except Exception as e:
        print(f"\n❌ ERRO DURANTE EXECUÇÃO: {e}")
        import traceback
        traceback.print_exc()
        return -1


def check_evolution_artifacts():
    """Verifica os artefatos da evolução do SeedAI."""
    print("\n🔍 VERIFICANDO ARTEFATOS DE AUTO-EVOLUÇÃO:")

    dirs_to_check = [
        "seed/evaluations/",
        "seed/metrics/",
        "seed/reports/",
        "seed/changes/",
        "seed/consciousness/"
    ]

    for dir_path in dirs_to_check:
        path = Path(dir_path)
        if path.exists():
            files = list(path.glob("*"))
            print(f"  {dir_path}: {len(files)} arquivos")
            for file in files[:3]:  # Mostrar apenas os 3 primeiros
                print(f"    - {file.name}")
        else:
            print(f"  {dir_path}: (não existe ainda)")


def main():
    """Função principal para iniciar a autoevolução."""
    print("🌟 INICIANDO SISTEMA DE AUTO-EVOLUÇÃO AUTÔNOMA")
    print("O SeedAI está prestes a começar sua jornada de autoevolução contínua...")
    print()

    # Iniciar a autoevolução
    result = start_autonomous_evolution()

    # Verificar artefatos
    check_evolution_artifacts()

    print("\n" + "=" * 60)
    print("🎯 O SEEDAI ESTÁ EM MODO AUTO-EVOLUTIVO!")
    print("   - Ele continuará gerando e executando seeds automaticamente")
    print("   - Monitorando métricas de inteligência emergente")
    print("   - Expandindo suas próprias capacidades")
    print("   - Auto-avaliando seu próprio desempenho")
    print("=" * 60)

    return result


if __name__ == "__main__":
    main()
