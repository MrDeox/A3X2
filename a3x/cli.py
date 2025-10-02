"""Interface de linha de comando para o A3X."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .agent import AgentOrchestrator
from .autoloop import load_goal_rotation, run_autopilot
from .autonomous_planner import run_autonomous_planning
from .config import load_config
from .llm import build_llm_client
from .memory.store import SemanticMemory
from .seed_daemon import main as seed_daemon_main
from .seed_runner import main as seed_main

try:
    from .config.validation import validate_config_file, ValidationError, migrate_config_file
    from .config.testing import ConfigTester
    from .config.utils import ConfigAnalyzer, ConfigGenerator
    HAS_CONFIG_TOOLS = True
except ImportError:
    HAS_CONFIG_TOOLS = False


def _handle_run_command(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if args.max_steps:
        config.limits.max_iterations = args.max_steps
    if args.interactive:
        config.loop.interactive = True
    if args.use_memory is not None:
        config.loop.use_memory = args.use_memory
    if args.memory_top_k is not None:
        config.loop.memory_top_k = max(0, args.memory_top_k)

    if args.deterministic_script:
        script_path = Path(args.deterministic_script)
        if not script_path.is_absolute():
            script_path = script_path.resolve()
        config.llm.type = "manual"
        config.llm.script = script_path
        config.llm.model = None
        config.llm.endpoint = None
        config.llm.base_url = None
        config.llm.api_key_env = None

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

    if args.auto_watch:
        return seed_daemon_main(
            [
                "--backlog",
                args.watch_backlog,
                "--config",
                args.config,
                "--interval",
                str(args.watch_interval),
                *(["--max-runs", str(args.watch_max_runs)] if args.watch_max_runs else []),
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
            *(["--max-runs", str(loop_cfg.seed_max_runs)] if loop_cfg.seed_max_runs is not None else []),
            *(["--no-stop-when-idle"] if not loop_cfg.stop_when_idle else []),
        ]
        seed_exit = seed_daemon_main(seed_args)
        return exit_code or seed_exit

    return exit_code


def _handle_seed_command(args: argparse.Namespace) -> int:
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
    return 1  # Should be unreachable if parser is configured correctly


def _handle_memory_command(args: argparse.Namespace) -> int:
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
    return 1  # Should be unreachable


def _handle_autopilot_command(args: argparse.Namespace) -> int:
    try:
        goals = load_goal_rotation(args.goals)
    except Exception as exc:
        print(f"Erro ao carregar rotação de objetivos: {exc}")
        return 1
    return run_autopilot(
        goals,
        cycles=args.cycles,
        backlog_path=args.backlog,
        seed_default_config=args.seed_default_config,
        seed_max=args.seed_max,
        seed_max_steps=args.seed_max_steps,
    )


def _handle_plan_command(args: argparse.Namespace) -> int:
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
    return 1  # Should be unreachable


def _handle_daemon_command(args: argparse.Namespace) -> int:
    if args.daemon_command == "start":
        from .continuous_evolution_daemon import start_continuous_evolution_daemon
        workspace_root = Path(args.workspace).resolve()
        config_path = Path(args.config) if args.config else None

        print("🚀 Iniciando daemon de auto-evolução contínua...")
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
    return 1  # Should be unreachable


def _handle_config_command(args: argparse.Namespace) -> int:
    """Handle configuration management commands."""
    if not HAS_CONFIG_TOOLS:
        print("❌ Configuration tools not available. Install jsonschema: pip install jsonschema")
        return 1

    if args.config_command == "validate":
        return _handle_config_validate(args)
    elif args.config_command == "test":
        return _handle_config_test(args)
    elif args.config_command == "analyze":
        return _handle_config_analyze(args)
    elif args.config_command == "migrate":
        return _handle_config_migrate(args)
    elif args.config_command == "generate":
        return _handle_config_generate(args)

    return 1


def _handle_config_validate(args: argparse.Namespace) -> int:
    """Handle configuration validation."""
    try:
        validate_config_file(args.config_file, strict=not args.lenient)
        print(f"✅ Configuration is valid: {args.config_file}")
        return 0
    except ValidationError as e:
        print(f"❌ Configuration validation failed: {args.config_file}")
        print(str(e))
        return 1
    except Exception as e:
        print(f"❌ Error validating configuration: {e}")
        return 1


def _handle_config_test(args: argparse.Namespace) -> int:
    """Handle configuration testing."""
    try:
        tester = ConfigTester(args.config_file)

        if args.functional:
            results = tester.run_functional_tests()
        elif args.validation_only:
            results = tester.run_validation_tests()
        else:
            results = tester.run_all_tests()

        tester.print_results(results)

        # Return non-zero if any tests failed
        failed_tests = [r for r in results if not r.passed]
        return len(failed_tests)
    except Exception as e:
        print(f"❌ Error testing configuration: {e}")
        return 1


def _handle_config_analyze(args: argparse.Namespace) -> int:
    """Handle configuration analysis."""
    try:
        analyzer = ConfigAnalyzer(args.config_file)
        analysis = analyzer.analyze_comprehensive()

        if args.output:
            analyzer.export_analysis(args.output)
            print(f"📊 Analysis exported to: {args.output}")
        else:
            analyzer.print_analysis(analysis)

        summary = analysis["summary"]
        return 1 if summary["error_count"] > 0 else 0
    except Exception as e:
        print(f"❌ Error analyzing configuration: {e}")
        return 1


def _handle_config_migrate(args: argparse.Namespace) -> int:
    """Handle configuration migration."""
    try:
        target_version = args.target_version or "1.3.0"

        migrated_config, messages, backup_path = migrate_config_file(
            Path(args.config_file),
            target_version=target_version,
            create_backup=not args.no_backup
        )

        print(f"🔄 Configuration migrated to version {target_version}")
        print(f"📁 File: {args.config_file}")

        if backup_path:
            print(f"💾 Backup created: {backup_path}")

        for message in messages:
            print(f"   • {message}")

        return 0
    except Exception as e:
        print(f"❌ Error migrating configuration: {e}")
        return 1


def _handle_config_generate(args: argparse.Namespace) -> int:
    """Handle configuration generation."""
    try:
        generator = ConfigGenerator()

        if args.generate_type == "minimal":
            generator.generate_minimal_config(args.output)
            print(f"📄 Minimal configuration generated: {args.output}")
        elif args.generate_type == "development":
            generator.generate_development_config(args.output)
            print(f"📄 Development configuration generated: {args.output}")
        elif args.generate_type == "secure":
            generator.generate_secure_config(args.output)
            print(f"📄 Secure configuration generated: {args.output}")
        elif args.generate_type == "llm-examples":
            generator.generate_llm_configs(args.output)
            print(f"📄 LLM example configurations generated in: {args.output}")
        elif args.generate_type == "schema-docs":
            generator.generate_schema_docs(args.output)
            print(f"📄 Schema documentation generated: {args.output}")

        return 0
    except Exception as e:
        print(f"❌ Error generating configuration: {e}")
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Agente autônomo de codificação local")
    subparsers = parser.add_subparsers(dest="command", required=True)

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
        "--interactive",
        action="store_true",
        help="Habilita modo interativo com prompts para refinamento de objetivos durante a execução",
    )
    run_parser.add_argument(
        "--show-history", action="store_true", help="Exibe histórico completo ao final"
    )
    run_parser.add_argument(
        "--deterministic-script",
        help="Força execução determinística via ManualLLMClient usando roteiro YAML",
    )
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
    run_parser.add_argument(
        "--use-memory",
        dest="use_memory",
        action="store_true",
        help="Ativa consulta à memória semântica antes de cada ação",
    )
    run_parser.add_argument(
        "--no-memory",
        dest="use_memory",
        action="store_false",
        help="Desativa consulta à memória semântica",
    )
    run_parser.add_argument(
        "--memory-top-k",
        type=int,
        dest="memory_top_k",
        help="Número de lembranças semânticas a incluir no contexto",
    )
    run_parser.set_defaults(use_memory=None, memory_top_k=None)

    seed_parser = subparsers.add_parser("seed", help="Operações relacionadas a seeds")
    seed_sub = seed_parser.add_subparsers(dest="seed_command", required=True)
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
    memory_sub = memory_parser.add_subparsers(dest="memory_command", required=True)
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

    plan_parser = subparsers.add_parser("plan", help="Planejamento autônomo de evolução")
    plan_sub = plan_parser.add_subparsers(dest="plan_command", required=True)
    plan_run_parser = plan_sub.add_parser("run", help="Executa planejamento autônomo")
    plan_run_parser.add_argument(
        "--workspace",
        default=".",
        help="Diretório raiz do workspace"
    )

    daemon_parser = subparsers.add_parser("daemon", help="Daemon de auto-evolução contínua")
    daemon_sub = daemon_parser.add_subparsers(dest="daemon_command", required=True)
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

    # Configuration management commands
    if HAS_CONFIG_TOOLS:
        config_parser = subparsers.add_parser("config", help="Gerenciamento de configuração")
        config_sub = config_parser.add_subparsers(dest="config_command", required=True)

        # Config validate command
        config_validate_parser = config_sub.add_parser("validate", help="Valida arquivo de configuração")
        config_validate_parser.add_argument(
            "config_file",
            help="Arquivo de configuração a validar"
        )
        config_validate_parser.add_argument(
            "--lenient",
            action="store_true",
            help="Modo de validação menos rigoroso"
        )

        # Config test command
        config_test_parser = config_sub.add_parser("test", help="Testa configuração completamente")
        config_test_parser.add_argument(
            "config_file",
            help="Arquivo de configuração a testar"
        )
        config_test_parser.add_argument(
            "--functional",
            action="store_true",
            help="Executa apenas testes funcionais"
        )
        config_test_parser.add_argument(
            "--validation-only",
            action="store_true",
            help="Executa apenas testes de validação"
        )

        # Config analyze command
        config_analyze_parser = config_sub.add_parser("analyze", help="Analisa configuração em busca de problemas")
        config_analyze_parser.add_argument(
            "config_file",
            help="Arquivo de configuração a analisar"
        )
        config_analyze_parser.add_argument(
            "--output",
            help="Arquivo de saída para resultados da análise (JSON)"
        )

        # Config migrate command
        config_migrate_parser = config_sub.add_parser("migrate", help="Migra configuração para versão mais recente")
        config_migrate_parser.add_argument(
            "config_file",
            help="Arquivo de configuração a migrar"
        )
        config_migrate_parser.add_argument(
            "--target-version",
            help="Versão alvo para migração (padrão: 1.3.0)"
        )
        config_migrate_parser.add_argument(
            "--no-backup",
            action="store_true",
            help="Não criar backup antes da migração"
        )

        # Config generate command
        config_generate_parser = config_sub.add_parser("generate", help="Gera configurações de exemplo")
        config_generate_parser.add_argument(
            "generate_type",
            choices=["minimal", "development", "secure", "llm-examples", "schema-docs"],
            help="Tipo de configuração a gerar"
        )
        config_generate_parser.add_argument(
            "output",
            help="Arquivo ou diretório de saída"
        )

    args = parser.parse_args(argv)

    if args.command == "run":
        return _handle_run_command(args)
    elif args.command == "seed":
        return _handle_seed_command(args)
    elif args.command == "memory":
        return _handle_memory_command(args)
    elif args.command == "autopilot":
        return _handle_autopilot_command(args)
    elif args.command == "plan":
        return _handle_plan_command(args)
    elif args.command == "daemon":
        return _handle_daemon_command(args)
    elif args.command == "config" and HAS_CONFIG_TOOLS:
        return _handle_config_command(args)

    # Fallback in case no command is matched (should not happen with required=True)
    parser.print_help()
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

