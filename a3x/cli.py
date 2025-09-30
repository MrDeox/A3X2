"""Interface de linha de comando para o A3X."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .agent import AgentOrchestrator
from .config import load_config
from .llm import build_llm_client
from .seed_runner import main as seed_main
from .seed_daemon import main as seed_daemon_main
from .memory.store import SemanticMemory
from .autoloop import load_goal_rotation, run_autopilot
from .autonomous_planner import run_autonomous_planning  # Nova importação


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Agente autônomo de codificação local")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Executa o agente com um objetivo")
    run_parser.add_argument(
        "--goal", required=True, help="Objetivo a ser perseguido pelo agente"
    )
    run_parser.add_argument(
        "--config", default="configs/sample.yaml", help="Arquivo de configuração YAML"
    )
    run_parser.add_argument(
        "--max-steps",
        type=int,
        help="Sobrescreve max_iterations do arquivo de configuração",
    )
    run_parser.add_argument(
        "--show-history", action="store_true", help="Exibe histórico completo ao final"
    )
    # Auto-watch: roda o daemon de seeds após a execução principal
    run_parser.add_argument(
        "--auto-watch",
        action="store_true",
        help="Após a run, executa seeds pendentes automaticamente",
    )
    run_parser.add_argument(
        "--watch-backlog",
        default="seed/backlog.yaml",
        help="Backlog a ser observado pelo daemon",
    )
    run_parser.add_argument(
        "--watch-interval",
        type=float,
        default=30.0,
        help="Intervalo entre execuções no daemon",
    )
    run_parser.add_argument(
        "--watch-max-runs", type=int, help="Limite de execuções do daemon"
    )
    run_parser.add_argument(
        "--watch-no-stop-when-idle",
        action="store_true",
        help="Não encerrar quando não houver seeds elegíveis",
    )

    seed_parser = subparsers.add_parser("seed", help="Operações relacionadas a seeds")
    seed_sub = seed_parser.add_subparsers(dest="seed_command")
    seed_run_parser = seed_sub.add_parser(
        "run", help="Executa a próxima seed pendente do backlog"
    )
    seed_run_parser.add_argument(
        "--backlog", default="seed/backlog.yaml", help="Arquivo YAML do backlog"
    )
    seed_run_parser.add_argument(
        "--config",
        default="configs/sample.yaml",
        help="Config padrão caso seed não defina a sua",
    )
    seed_run_parser.add_argument(
        "--max-steps", type=int, help="Override de max_iterations"
    )

    seed_watch_parser = seed_sub.add_parser(
        "watch", help="Executa seeds continuamente (daemon de loop)"
    )
    seed_watch_parser.add_argument("--backlog", default="seed/backlog.yaml")
    seed_watch_parser.add_argument("--config", default="configs/sample.yaml")
    seed_watch_parser.add_argument("--interval", type=float, default=30.0)
    seed_watch_parser.add_argument("--max-runs", type=int)
    seed_watch_parser.add_argument("--no-stop-when-idle", action="store_true")

    memory_parser = subparsers.add_parser("memory", help="Memória semântica SeedAI")
    memory_sub = memory_parser.add_subparsers(dest="memory_command")
    memory_search = memory_sub.add_parser(
        "search", help="Pesquisa semântica por lembranças"
    )
    memory_search.add_argument(
        "--query", required=True, help="Texto para busca semântica"
    )
    memory_search.add_argument(
        "--top-k", type=int, default=5, help="Número de resultados"
    )
    memory_search.add_argument("--memory-path", default="seed/memory/memory.jsonl")

    autopilot_parser = subparsers.add_parser(
        "autopilot", help="Executa objetivos e seeds em sequência"
    )
    autopilot_parser.add_argument(
        "--goals",
        default="seed/goal_rotation.yaml",
        help="Arquivo YAML com rotação de objetivos",
    )
    autopilot_parser.add_argument(
        "--cycles", type=int, default=1, help="Número de ciclos de objetivos a executar"
    )
    autopilot_parser.add_argument(
        "--backlog", default="seed/backlog.yaml", help="Backlog de seeds"
    )
    autopilot_parser.add_argument(
        "--seed-default-config",
        default="configs/sample.yaml",
        help="Config padrão para seeds sem configuração própria",
    )
    autopilot_parser.add_argument(
        "--seed-max", type=int, help="Limite de seeds por ciclo (default: até esvaziar)"
    )
    autopilot_parser.add_argument(
        "--seed-max-steps",
        type=int,
        help="Override de max_iterations para seeds executadas",
    )

    # Novo comando plan
    plan_parser = subparsers.add_parser("plan", help="Planejamento autônomo de evolução")
    plan_sub = plan_parser.add_subparsers(dest="plan_command")
    plan_run_parser = plan_sub.add_parser("run", help="Executa planejamento autônomo")
    plan_run_parser.add_argument(
        "--workspace", 
        default=".", 
        help="Diretório raiz do workspace"
    )

    # Novo comando daemon
    daemon_parser = subparsers.add_parser("daemon", help="Daemon de auto-evolução contínua")
    daemon_sub = daemon_parser.add_subparsers(dest="daemon_command")
    daemon_start_parser = daemon_sub.add_parser("start", help="Inicia o daemon de auto-evolução")
    daemon_start_parser.add_argument(
        "--workspace", 
        default=".", 
        help="Diretório raiz do workspace"
    )
    daemon_start_parser.add_argument(
        "--config", 
        help="Arquivo de configuração"
    )
    daemon_start_parser.add_argument(
        "--max-cycles", 
        type=int, 
        help="Número máximo de ciclos (padrão: infinito)"
    )
    daemon_start_parser.add_argument(
        "--interval", 
        type=int, 
        default=300, 
        help="Intervalo entre ciclos em segundos (padrão: 300)"
    )

    args = parser.parse_args(argv)

    if args.command == "memory":
        if args.memory_command == "search":
            store = SemanticMemory(args.memory_path)
            try:
                results = store.query(args.query, top_k=args.top_k)
            except RuntimeError as exc:
                print(f"Memória indisponível: {exc}")
                return 1
            if not results:
                print("Nenhuma lembrança encontrada.")
                return 0
            for entry, score in results:
                print(f"[{score:.3f}] {entry.title}")
                print(entry.content)
                print("---")
            return 0
        parser.error("Comando memory requer subcomando válido (ex.: search)")

    if args.command == "autopilot":
        try:
            goals = load_goal_rotation(args.goals)
        except Exception as exc:
            print(f"Erro ao carregar rotação de objetivos: {exc}")
            return 1
        exit_code = run_autopilot(
            goals,
            cycles=args.cycles,
            backlog_path=args.backlog,
            seed_default_config=args.seed_default_config,
            seed_max=args.seed_max,
            seed_max_steps=args.seed_max_steps,
        )
        return exit_code

    if args.command == "plan":
        if args.plan_command == "run":
            workspace_root = Path(args.workspace).resolve()
            print(f"🤖 Executando planejamento autônomo no workspace: {workspace_root}")
            
            try:
                seeds = run_autonomous_planning(workspace_root)
                print(f"✅ Planejamento autônomo concluído! {len(seeds)} seeds gerados.")
                return 0
            except Exception as e:
                print(f"❌ Erro durante planejamento autônomo: {e}")
                import traceback
                traceback.print_exc()
                return 1
        else:
            plan_parser.print_help()
            return 1

    # Comando daemon (NOVO)
    if args.command == "daemon":
        if args.daemon_command == "start":
            from .continuous_evolution_daemon import start_continuous_evolution_daemon
            workspace_root = Path(args.workspace).resolve()
            config_path = Path(args.config) if args.config else None
            
            print(f"🚀 Iniciando daemon de auto-evolução contínua...")
            print(f"   Workspace: {workspace_root}")
            print(f"   Config: {config_path or 'padrão'}")
            print(f"   Ciclos máximos: {args.max_cycles or 'infinito'}")
            print(f"   Intervalo: {args.interval} segundos")
            
            try:
                start_continuous_evolution_daemon(
                    workspace_root=workspace_root,
                    max_cycles=args.max_cycles,
                    analysis_interval=args.interval,
                    config_path=config_path
                )
                return 0
            except Exception as e:
                print(f"❌ Erro no daemon: {e}")
                import traceback
                traceback.print_exc()
                return 1
        else:
            daemon_parser.print_help()
            return 1

    if args.command == "seed":
        if args.seed_command == "run":
            return seed_main(
                [
                    "--backlog",
                    args.backlog,
                    "--config",
                    args.config,
                    *(["--max-steps", str(args.max_steps)] if args.max_steps else []),
                ]
            )
        if args.seed_command == "watch":
            return seed_daemon_main(
                [
                    "--backlog",
                    args.backlog,
                    "--config",
                    args.config,
                    "--interval",
                    str(args.interval),
                    *(["--max-runs", str(args.max_runs)] if args.max_runs else []),
                    *(["--no-stop-when-idle"] if args.no_stop_when_idle else []),
                ]
            )
        parser.error("Comando seed requer subcomando válido (ex.: run)")

    if args.command != "run":
        parser.print_help()
        return 0

    config = load_config(args.config)
    if args.max_steps:
        config.limits.max_iterations = args.max_steps

    llm_client = build_llm_client(config.llm)
    orchestrator = AgentOrchestrator(config, llm_client)

    result = orchestrator.run(args.goal)

    print("=== Resultado ===")
    print(f"Objetivo        : {args.goal}")
    print(f"Concluído       : {result.completed}")
    print(f"Iterações       : {result.iterations}")
    print(f"Falhas          : {result.failures}")
    if result.errors:
        print("Erros           :")
        for err in result.errors:
            print(f"  - {err}")

    if args.show_history:
        print("\n=== Histórico ===")
        print(result.history.snapshot())

    exit_code = 0 if result.completed else 1

    loop_cfg = config.loop

    # Auto-watch daemon para fechar o loop sem humano
    if args.auto_watch:
        exit_code = seed_daemon_main(
            [
                "--backlog",
                args.watch_backlog,
                "--config",
                args.config,
                "--interval",
                str(args.watch_interval),
                *(
                    ["--max-runs", str(args.watch_max_runs)]
                    if args.watch_max_runs
                    else []
                ),
                *(["--no-stop-when-idle"] if args.watch_no_stop_when_idle else []),
            ]
        )
    elif loop_cfg.auto_seed:
        seed_args = [
            "--backlog",
            str(loop_cfg.seed_backlog),
            "--config",
            str(loop_cfg.seed_config or Path(args.config).resolve()),
            "--interval",
            str(loop_cfg.seed_interval),
            *(
                ["--max-runs", str(loop_cfg.seed_max_runs)]
                if loop_cfg.seed_max_runs is not None
                else []
            ),
            *(["--no-stop-when-idle"] if not loop_cfg.stop_when_idle else []),
        ]
        seed_exit = seed_daemon_main(seed_args)
        exit_code = exit_code or seed_exit

    return exit_code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

