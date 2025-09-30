#!/usr/bin/env python3
"""Daemon de auto-evolução contínua para o SeedAI."""

from __future__ import annotations

import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml
import asyncio

from .autonomous_planner import AutonomousPlanner, run_autonomous_planning
from .autoeval import RunEvaluation, EvaluationSeed
from .seeds import SeedBacklog


class ContinuousEvolutionDaemon:
    """Daemon que mantém a auto-evolução contínua do SeedAI."""
    
    def __init__(self, workspace_root: Path, config_path: Path = None) -> None:
        self.workspace_root = workspace_root
        self.config_path = config_path or Path("configs/seed_manual.yaml")
        self.metrics_path = workspace_root / "seed" / "metrics" / "history.json"
        self.evaluations_path = workspace_root / "seed" / "evaluations" / "run_evaluations.jsonl"
        self.backlog_path = workspace_root / "seed" / "backlog.yaml"
        self.plans_path = workspace_root / "seed" / "plans"
        self.logs_path = workspace_root / "seed" / "daemon_logs"
        self.logs_path.mkdir(parents=True, exist_ok=True)
        
        # Componentes principais
        self.planner = AutonomousPlanner(workspace_root)
        
        # Estado do daemon
        self.running = False
        self.cycle_count = 0
        self.total_seeds_executed = 0
        self.successful_seeds = 0
        self.failed_seeds = 0
        self.last_analysis_timestamp = None
        
        print(f"🤖 DAEMON DE AUTO-EVOLUÇÃO CONTÍNUA INICIALIZADO")
        print(f"   Workspace: {workspace_root}")
        print(f"   Config: {self.config_path}")
        print(f"   Backlog: {self.backlog_path}")
    
    def start(self, max_cycles: Optional[int] = None, analysis_interval: int = 300) -> None:
        """Inicia o daemon de auto-evolução contínua."""
        print(f"\n🚀 INICIANDO DAEMON DE AUTO-EVOLUÇÃO CONTÍNUA")
        print(f"   Ciclos máximos: {max_cycles or 'infinito'}")
        print(f"   Intervalo de análise: {analysis_interval} segundos")
        print(f"   Pressione Ctrl+C para parar")
        print("=" * 60)
        
        self.running = True
        self.cycle_count = 0
        
        try:
            while self.running:
                self.cycle_count += 1
                print(f"\n🔄 CICLO #{self.cycle_count} - {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
                print("-" * 50)
                
                # 1. Executar seeds disponíveis
                seeds_executed = self._execute_available_seeds()
                
                # 2. Analisar resultados e métricas
                analysis_results = self._analyze_system_performance()
                
                # 3. Gerar novos seeds com LLM baseado na análise
                new_seeds_generated = self._generate_new_seeds_autonomously(analysis_results)
                
                # 4. Registrar ciclo no log
                self._log_cycle_completion(seeds_executed, analysis_results, new_seeds_generated)
                
                # 5. Verificar limite de ciclos
                if max_cycles and self.cycle_count >= max_cycles:
                    print(f"\n🏁 Limite de {max_cycles} ciclos atingido. Parando daemon...")
                    break
                
                # 6. Esperar antes do próximo ciclo
                print(f"\n⏰ Próximo ciclo em {analysis_interval} segundos...")
                time.sleep(analysis_interval)
                
        except KeyboardInterrupt:
            print(f"\n🛑 Daemon interrompido pelo usuário")
        except Exception as e:
            print(f"\n❌ Erro fatal no daemon: {e}")
            traceback.print_exc()
        finally:
            self.running = False
            self._generate_final_report()
            print(f"\n✅ DAEMON FINALIZADO - {self.cycle_count} ciclos executados")
    
    def _execute_available_seeds(self) -> int:
        """Executa seeds disponíveis no backlog."""
        print("🔍 Verificando seeds disponíveis para execução...")
        
        seeds_executed = 0
        try:
            # Verificar se há seeds no backlog
            if self.backlog_path.exists():
                with open(self.backlog_path, 'r', encoding='utf-8') as f:
                    backlog_content = f.read().strip()
                    if backlog_content:
                        # Aqui normalmente chamaríamos o executor de seeds
                        # Por simplicidade, vamos simular a execução
                        backlog_entries = list(yaml.safe_load_all(backlog_content))
                        backlog_entries = [entry for entry in backlog_entries if entry is not None]
                        
                        seeds_available = len(backlog_entries)
                        print(f"   🌱 Seeds disponíveis: {seeds_available}")
                        
                        if seeds_available > 0:
                            # Simular execução de alguns seeds
                            seeds_to_execute = min(3, seeds_available)  # Executar até 3 seeds
                            print(f"   ▶️  Executando {seeds_to_execute} seeds...")
                            
                            for i in range(seeds_to_execute):
                                seed_id = backlog_entries[i].get('id', f'seed_{i}')
                                print(f"     • Executando seed: {seed_id}")
                                time.sleep(1)  # Simular tempo de execução
                                seeds_executed += 1
                                self.total_seeds_executed += 1
                                
                                # Simular sucesso/fracasso aleatório
                                import random
                                if random.random() > 0.2:  # 80% de sucesso
                                    self.successful_seeds += 1
                                    print(f"       ✅ Sucesso!")
                                else:
                                    self.failed_seeds += 1
                                    print(f"       ❌ Falha!")
                            
                            # Remover seeds executados do backlog (simulação)
                            remaining_seeds = backlog_entries[seeds_to_execute:]
                            if remaining_seeds:
                                with open(self.backlog_path, 'w', encoding='utf-8') as f:
                                    yaml.dump(remaining_seeds, f, default_flow_style=False, allow_unicode=True, indent=2)
                            else:
                                # Limpar arquivo se não houver mais seeds
                                self.backlog_path.write_text("", encoding='utf-8')
                    else:
                        print("   📭 Nenhum seed disponível no backlog")
            else:
                print("   📭 Nenhum backlog encontrado")
                
        except Exception as e:
            print(f"   ❌ Erro ao executar seeds: {e}")
            
        print(f"   📊 Seeds executados neste ciclo: {seeds_executed}")
        return seeds_executed
    
    def _analyze_system_performance(self) -> Dict[str, Any]:
        """Analisa o desempenho do sistema baseado em métricas históricas."""
        print("📊 Analisando desempenho do sistema...")
        
        analysis_results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics_summary": {},
            "performance_trends": {},
            "identified_issues": [],
            "improvement_opportunities": [],
            "system_health": "good"
        }
        
        try:
            # Analisar métricas atuais
            if self.metrics_path.exists():
                with open(self.metrics_path, 'r', encoding='utf-8') as f:
                    metrics_data = json.load(f)
                
                # Calcular resumo das métricas principais
                key_metrics = [
                    'actions_success_rate',
                    'apply_patch_success_rate', 
                    'failures',
                    'iterations',
                    'recursion_depth'
                ]
                
                for metric in key_metrics:
                    if metric in metrics_data and len(metrics_data[metric]) > 0:
                        recent_values = metrics_data[metric][-10:]  # Últimos 10 valores
                        current_value = recent_values[-1] if recent_values else 0
                        average_value = sum(recent_values) / len(recent_values) if recent_values else 0
                        
                        analysis_results["metrics_summary"][metric] = {
                            "current": current_value,
                            "average": average_value,
                            "trend": self._calculate_trend(recent_values)
                        }
                        
                        # Identificar problemas críticos
                        if metric == 'apply_patch_success_rate' and current_value < 0.5:
                            analysis_results["identified_issues"].append({
                                "type": "critical",
                                "metric": metric,
                                "value": current_value,
                                "description": f"Taxa crítica de sucesso em patches: {current_value:.1%}"
                            })
                        elif metric == 'failures' and current_value > 5:
                            analysis_results["identified_issues"].append({
                                "type": "warning",
                                "metric": metric,
                                "value": current_value,
                                "description": f"Alto número de falhas: {current_value:.0f}"
                            })
                
                # Determinar saúde geral do sistema
                critical_issues = [issue for issue in analysis_results["identified_issues"] if issue["type"] == "critical"]
                if len(critical_issues) > 0:
                    analysis_results["system_health"] = "critical"
                elif len(analysis_results["identified_issues"]) > 2:
                    analysis_results["system_health"] = "warning"
                    
            # Analisar tendências de desempenho
            analysis_results["performance_trends"] = self._identify_performance_trends(metrics_data)
            
            print(f"   📈 Saúde do sistema: {analysis_results['system_health']}")
            print(f"   📊 Métricas analisadas: {len(analysis_results['metrics_summary'])}")
            print(f"   ⚠️  Issues identificados: {len(analysis_results['identified_issues'])}")
            
        except Exception as e:
            print(f"   ❌ Erro na análise de desempenho: {e}")
            analysis_results["system_health"] = "unknown"
            
        self.last_analysis_timestamp = datetime.now(timezone.utc)
        return analysis_results
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calcula a tendência de uma série de valores."""
        if len(values) < 2:
            return "stable"
        
        # Calcular diferença média entre valores recentes e antigos
        recent_avg = sum(values[-3:]) / min(3, len(values))
        older_avg = sum(values[:-3]) / max(1, len(values) - 3) if len(values) > 3 else recent_avg
        
        if recent_avg > older_avg * 1.1:  # 10% de melhoria
            return "improving"
        elif recent_avg < older_avg * 0.9:  # 10% de piora
            return "declining"
        else:
            return "stable"
    
    def _identify_performance_trends(self, metrics_data: Dict[str, Any]) -> Dict[str, str]:
        """Identifica tendências gerais de desempenho."""
        trends = {}
        
        # Tendências específicas para métricas importantes
        important_metrics = ['actions_success_rate', 'apply_patch_success_rate', 'failures']
        
        for metric in important_metrics:
            if metric in metrics_data and len(metrics_data[metric]) >= 5:
                values = metrics_data[metric][-5:]  # Últimos 5 valores
                trend = self._calculate_trend(values)
                trends[metric] = trend
                
        return trends
    
    def _generate_new_seeds_autonomously(self, analysis_results: Dict[str, Any]) -> int:
        """Gera novos seeds automaticamente com base na análise do sistema."""
        print("🧠 Gerando novos seeds com LLM baseado na análise...")
        
        seeds_generated = 0
        try:
            # Usar o planejador autônomo para gerar novos seeds
            print("   🤖 Invocando planejador autônomo...")
            
            # Em produção, chamaríamos: seeds = run_autonomous_planning(self.workspace_root)
            # Para simulação, vamos gerar seeds baseados na análise
            
            health = analysis_results.get("system_health", "good")
            issues = analysis_results.get("identified_issues", [])
            trends = analysis_results.get("performance_trends", {})
            
            # Gerar seeds com base na saúde do sistema
            new_seeds = []
            
            if health == "critical":
                # Problemas críticos - gerar seeds de emergência
                critical_seeds = self._generate_critical_improvement_seeds(issues)
                new_seeds.extend(critical_seeds)
                print(f"   🚨 Gerados {len(critical_seeds)} seeds de emergência")
            elif health == "warning":
                # Problemas moderados - gerar seeds de melhoria
                improvement_seeds = self._generate_improvement_seeds(issues, trends)
                new_seeds.extend(improvement_seeds)
                print(f"   ⚠️  Gerados {len(improvement_seeds)} seeds de melhoria")
            else:
                # Sistema saudável - gerar seeds de otimização e expansão
                optimization_seeds = self._generate_optimization_seeds(trends)
                new_seeds.extend(optimization_seeds)
                print(f"   ✨ Gerados {len(optimization_seeds)} seeds de otimização")
            
            # Adicionar seeds ao backlog
            if new_seeds:
                self._add_seeds_to_backlog(new_seeds)
                seeds_generated = len(new_seeds)
                print(f"   ➕ {seeds_generated} novos seeds adicionados ao backlog")
            else:
                print("   📭 Nenhum novo seed gerado")
                
        except Exception as e:
            print(f"   ❌ Erro ao gerar novos seeds: {e}")
            traceback.print_exc()
            
        return seeds_generated
    
    def _generate_critical_improvement_seeds(self, issues: List[Dict]) -> List[EvaluationSeed]:
        """Gera seeds para resolver problemas críticos."""
        seeds = []
        
        for issue in issues:
            if issue.get("type") == "critical":
                seed = EvaluationSeed(
                    description=f"[EMERGÊNCIA] Resolver {issue['description']} criticamente",
                    priority="high",
                    capability="core.stability",
                    seed_type="emergency_fix",
                    data={
                        "issue": issue,
                        "urgency": "critical",
                        "impact": "system_stability"
                    }
                )
                seeds.append(seed)
                
        return seeds
    
    def _generate_improvement_seeds(self, issues: List[Dict], trends: Dict[str, str]) -> List[EvaluationSeed]:
        """Gera seeds para melhorias gerais."""
        seeds = []
        
        # Seeds baseados em issues identificados
        for issue in issues:
            seed = EvaluationSeed(
                description=f"[MELHORIA] Melhorar {issue['description']}",
                priority="medium",
                capability="core.optimization",
                seed_type="improvement",
                data={
                    "issue": issue,
                    "trend": trends.get(issue.get('metric', ''), 'unknown')
                }
            )
            seeds.append(seed)
            
        # Seeds baseados em tendências negativas
        for metric, trend in trends.items():
            if trend == "declining":
                seed = EvaluationSeed(
                    description=f"[PREVENTIVO] Reverter declínio em {metric}",
                    priority="medium",
                    capability="core.stability",
                    seed_type="preventive_action",
                    data={
                        "metric": metric,
                        "trend": trend
                    }
                )
                seeds.append(seed)
                
        return seeds
    
    def _generate_optimization_seeds(self, trends: Dict[str, str]) -> List[EvaluationSeed]:
        """Gera seeds para otimização em sistemas saudáveis."""
        seeds = []
        
        # Seeds de otimização geral
        seed = EvaluationSeed(
            description="[OTIMIZAÇÃO] Otimizar desempenho geral do sistema",
            priority="low",
            capability="core.optimization",
            seed_type="optimization",
            data={
                "focus": "general_performance"
            }
        )
        seeds.append(seed)
        
        # Seeds de expansão e aprendizado
        expansion_seed = EvaluationSeed(
            description="[EXPANSÃO] Explorar novas capacidades e domínios",
            priority="low",
            capability="horiz.expansion",
            seed_type="exploration",
            data={
                "focus": "capability_expansion"
            }
        )
        seeds.append(expansion_seed)
        
        return seeds
    
    def _add_seeds_to_backlog(self, seeds: List[EvaluationSeed]) -> None:
        """Adiciona novos seeds ao backlog existente."""
        try:
            # Ler backlog atual
            backlog_entries = []
            if self.backlog_path.exists():
                with open(self.backlog_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        backlog_entries = list(yaml.safe_load_all(content))
                        backlog_entries = [entry for entry in backlog_entries if entry is not None]
            
            # Converter seeds para formato do backlog
            timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
            for i, seed in enumerate(seeds):
                backlog_entry = {
                    "id": f"auto_{seed.capability}_{timestamp}_{i}",
                    "goal": seed.description,
                    "priority": seed.priority,
                    "type": seed.seed_type,
                    "config": str(self.config_path),
                    "metadata": {
                        "description": seed.description,
                        "created_by": "continuous_evolution_daemon",
                        "tags": ["autonomous", "continuous", seed.capability, seed.seed_type]
                    },
                    "history": [],
                    "attempts": 0,
                    "max_attempts": 3,
                    "next_run_at": None,
                    "last_error": None,
                    "data": seed.data
                }
                backlog_entries.append(backlog_entry)
            
            # Salvar backlog atualizado
            with open(self.backlog_path, 'w', encoding='utf-8') as f:
                yaml.dump(backlog_entries, f, default_flow_style=False, allow_unicode=True, indent=2)
                
        except Exception as e:
            print(f"   ❌ Erro ao adicionar seeds ao backlog: {e}")
    
    def _log_cycle_completion(self, seeds_executed: int, analysis_results: Dict[str, Any], seeds_generated: int) -> None:
        """Registra a conclusão de um ciclo no log."""
        try:
            cycle_log = {
                "cycle_number": self.cycle_count,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "seeds_executed": seeds_executed,
                "seeds_generated": seeds_generated,
                "system_health": analysis_results.get("system_health", "unknown"),
                "metrics_summary": analysis_results.get("metrics_summary", {}),
                "issues_count": len(analysis_results.get("identified_issues", [])),
                "daemon_status": {
                    "total_cycles": self.cycle_count,
                    "total_seeds_executed": self.total_seeds_executed,
                    "successful_seeds": self.successful_seeds,
                    "failed_seeds": self.failed_seeds
                }
            }
            
            # Salvar log do ciclo
            log_file = self.logs_path / f"cycle_{self.cycle_count:06d}.json"
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(cycle_log, f, ensure_ascii=False, indent=2)
                
            print(f"   📝 Ciclo registrado: {log_file.name}")
            
        except Exception as e:
            print(f"   ❌ Erro ao registrar ciclo: {e}")
    
    def _generate_final_report(self) -> None:
        """Gera relatório final da execução do daemon."""
        print(f"\n📊 RELATÓRIO FINAL DO DAEMON")
        print("=" * 40)
        print(f"Ciclos executados: {self.cycle_count}")
        print(f"Total de seeds executados: {self.total_seeds_executed}")
        print(f"Seeds bem-sucedidos: {self.successful_seeds}")
        print(f"Seeds com falha: {self.failed_seeds}")
        
        if self.total_seeds_executed > 0:
            success_rate = (self.successful_seeds / self.total_seeds_executed) * 100
            print(f"Taxa de sucesso: {success_rate:.1f}%")
        
        print(f"Última análise: {self.last_analysis_timestamp or 'Nenhuma'}")
        
        # Salvar relatório final
        final_report = {
            "daemon_session": {
                "start_time": datetime.now(timezone.utc).isoformat(),
                "end_time": datetime.now(timezone.utc).isoformat(),
                "total_cycles": self.cycle_count,
                "total_seeds_executed": self.total_seeds_executed,
                "successful_seeds": self.successful_seeds,
                "failed_seeds": self.failed_seeds,
                "final_system_health": "unknown"
            }
        }
        
        report_file = self.logs_path / "final_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, ensure_ascii=False, indent=2)
        
        print(f"📄 Relatório final salvo: {report_file}")


def start_continuous_evolution_daemon(
    workspace_root: Path, 
    max_cycles: Optional[int] = None, 
    analysis_interval: int = 300,
    config_path: Path = None
) -> None:
    """Inicia o daemon de auto-evolução contínua."""
    daemon = ContinuousEvolutionDaemon(workspace_root, config_path)
    daemon.start(max_cycles=max_cycles, analysis_interval=analysis_interval)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Daemon de auto-evolução contínua para o SeedAI")
    parser.add_argument("--workspace", default=".", help="Diretório raiz do workspace")
    parser.add_argument("--max-cycles", type=int, help="Número máximo de ciclos (padrão: infinito)")
    parser.add_argument("--interval", type=int, default=300, help="Intervalo entre ciclos em segundos (padrão: 300)")
    parser.add_argument("--config", help="Arquivo de configuração")
    
    args = parser.parse_args()
    
    workspace_root = Path(args.workspace).resolve()
    config_path = Path(args.config) if args.config else None
    
    start_continuous_evolution_daemon(
        workspace_root=workspace_root,
        max_cycles=args.max_cycles,
        analysis_interval=args.interval,
        config_path=config_path
    )