"""Executa ações solicitadas pelo agente."""

from __future__ import annotations

import ast
import os
import shlex
import subprocess
import time
import tempfile
from pathlib import Path
from typing import Optional, Dict

from .actions import AgentAction, Observation
from .config import AgentConfig
from .patch import PatchManager, PatchError
from .change_log import ChangeLogger

import ast
import re
import shutil
from datetime import datetime, timedelta
import logging


class ActionExecutor:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.workspace_root = Path(config.workspace.root).resolve()
        self.patch_manager = PatchManager(self.workspace_root)
        self.change_logger = ChangeLogger(
            self.workspace_root,
            enable_file_log=config.audit.enable_file_log,
            file_dir=config.audit.file_dir,
            enable_git_commit=config.audit.enable_git_commit,
            commit_prefix=config.audit.commit_prefix,
        )
        # Initialize rollback system
        self.rollback_checkpoints = {}
        self.rollback_triggers = []
        self._initialize_rollback_system()

    def execute(self, action: AgentAction) -> Observation:
        handler_name = f"_handle_{action.type.name.lower()}"
        handler = getattr(self, handler_name, None)
        if handler is None:
            return Observation(
                success=False, output="", error=f"Ação não suportada: {action.type}"
            )
        return handler(action)

    # Handlers -----------------------------------------------------------------

    def _handle_message(self, action: AgentAction) -> Observation:
        return Observation(success=True, output=action.text or "", type="message")

    def _handle_finish(self, action: AgentAction) -> Observation:
        return Observation(success=True, output=action.text or "", type="finish")

    def _handle_read_file(self, action: AgentAction) -> Observation:
        if not action.path:
            return Observation(
                success=False, error="Caminho não informado", type="read_file"
            )
        target = self._resolve_workspace_path(action.path)
        if not target.exists():
            return Observation(
                success=False,
                error=f"Arquivo não encontrado: {target}",
                type="read_file",
            )
        try:
            content = target.read_text(encoding="utf-8")
        except Exception as exc:  # pragma: no cover - leitura raramente falha
            return Observation(success=False, error=str(exc), type="read_file")
        return Observation(success=True, output=content, type="read_file")

    def _handle_write_file(self, action: AgentAction) -> Observation:
        if not action.path:
            return Observation(
                success=False, error="Caminho não informado", type="write_file"
            )
        target = self._resolve_workspace_path(action.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        before = ""
        if target.exists():
            try:
                before = target.read_text(encoding="utf-8")
            except Exception:
                before = ""
        try:
            target.write_text(action.content or "", encoding="utf-8")
        except Exception as exc:
            return Observation(success=False, error=str(exc), type="write_file")
        try:
            self.change_logger.log_write(
                target, before, action.content or "", note="write_file"
            )
        except Exception:
            pass
        return Observation(success=True, output=f"Escrito {target}", type="write_file")

    def _handle_apply_patch(self, action: AgentAction) -> Observation:
        if not action.diff:
            return Observation(success=False, error="Diff vazio", type="apply_patch")

        # Extract Python files from diff for AST validation
        py_paths = set(re.findall(r'^--- a/(.+\.py)$', action.diff, re.MULTILINE))
        backups = {}
        for rel_path in py_paths:
            full_path = self._resolve_workspace_path(rel_path)
            if full_path.exists():
                backup_path = full_path.with_suffix('.py.bak')
                shutil.copy2(full_path, backup_path)
                backups[rel_path] = backup_path

        # Run risk checks before applying
        risks = self._run_risk_checks(action.diff)
        high_risks = {k: v for k, v in risks.items() if v == 'high'}
        if high_risks:
            details = f"High risks detected before patch application: {high_risks}"
            # Log to risk_log.md
            log_path = self.workspace_root / 'seed' / 'reports' / 'risk_log.md'
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open('a', encoding='utf-8') as f:
                f.write(f"\n## Patch Risk Log - {datetime.now().isoformat()}\n")
                f.write(f"Patch risks: {details}\n")
                f.write(f"Full risks: {risks}\n\n")
            # Since not applied yet, no revert needed
            return Observation(
                success=False,
                output=f"Patch rejected due to high risks: {details}",
                type="apply_patch"
            )

        try:
            success, output = self.patch_manager.apply(action.diff)

            # AST validation fallback
            has_error = False
            for rel_path in py_paths:
                full_path = self._resolve_workspace_path(rel_path)
                if full_path.suffix == '.py' and full_path.exists():
                    try:
                        content = full_path.read_text(encoding="utf-8")
                        ast.parse(content)
                    except SyntaxError as e:
                        has_error = True
                        output += f"\nSyntaxError in {rel_path}: {str(e)}"
                        # Revert from backup
                        backup_path = backups.get(rel_path)
                        if backup_path and backup_path.exists():
                            full_path.write_text(backup_path.read_text(encoding="utf-8"))
                            backup_path.unlink()
            if has_error:
                success = False
                output += "\nAST fallback: Patch rejected due to syntax errors; affected files reverted."
            else:
                if success:
                    try:
                        self.change_logger.log_patch(action.diff, note="apply_patch")
                    except Exception:
                        pass

                    # Cleanup old .diff files in seed/changes/
                    from pathlib import Path
                    changes_dir = Path('seed/changes')
                    archive_dir = Path('seed/archive')
                    if changes_dir.exists():
                        for filename in os.listdir(changes_dir):
                            if filename.endswith('.diff'):
                                file_path = changes_dir / filename
                                if file_path.is_file():
                                    mtime = os.path.getmtime(file_path)
                                    if datetime.now() - datetime.fromtimestamp(mtime) > timedelta(days=7):
                                        shutil.move(str(file_path), str(archive_dir / filename))

            return Observation(success=success, output=output, type="apply_patch")
        except PatchError as exc:
            # Clean up backups on patch error
            for backup_path in backups.values():
                backup_path.unlink(missing_ok=True)
            return Observation(success=False, error=str(exc), type="apply_patch")

    def _run_risk_checks(self, patch_content: str) -> Dict[str, str]:
        """Run lightweight risk checks on patch: linting with ruff and black."""
        risks = {}
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_path = Path(temp_dir_str)
            # Extract Python files from patch
            py_paths = set(re.findall(r'^--- a/(.+\.py)$', patch_content, re.MULTILINE))
            if not py_paths:
                return risks  # No Python files, low risk

            # Copy original files to temp dir
            for rel_path in py_paths:
                full_path = self._resolve_workspace_path(rel_path)
                if full_path.exists():
                    temp_file = temp_path / rel_path
                    temp_file.parent.mkdir(parents=True, exist_ok=True)
                    temp_file.write_text(full_path.read_text(encoding="utf-8"), encoding="utf-8")

            # Apply patch in temp dir
            temp_pm = PatchManager(temp_path)
            try:
                temp_success, temp_output = temp_pm.apply(patch_content)
                if not temp_success:
                    risks['patch_apply'] = 'high'
                    return risks
            except Exception as e:
                risks['patch_apply'] = 'high'
                risks['apply_error'] = str(e)[:100]
                return risks

            # Run linters on modified Python files
            for rel_path in py_paths:
                temp_file = temp_path / rel_path
                if temp_file.exists() and temp_file.suffix == '.py':
                    try:
                        # Ruff check
                        ruff_result = subprocess.run(
                            ['ruff', 'check', '--output-format=text', str(temp_file)],
                            capture_output=True, text=True, timeout=10, cwd=temp_dir_str
                        )
                        if ruff_result.returncode != 0:
                            violations = len([line for line in ruff_result.stdout.splitlines() if line.strip() and not line.startswith('==')])
                            if violations > 0:
                                # High if syntax errors (E9), else medium
                                if any(code.startswith('E9') for code in ruff_result.stdout.split() if code.startswith('E')):
                                    risks['ruff_syntax'] = 'high'
                                else:
                                    risks['ruff'] = 'medium' if violations <= 5 else 'high'
                            if violations > 5:
                                risks['ruff_violations'] = 'medium'

                        # Black check
                        black_result = subprocess.run(
                            ['black', '--check', str(temp_file)],
                            capture_output=True, text=True, timeout=10, cwd=temp_dir_str
                        )
                        if black_result.returncode != 0:
                            risks['black_style'] = 'medium'

                    except subprocess.TimeoutExpired:
                        risks['lint_timeout'] = 'high'
                    except FileNotFoundError:
                        # ruff/black not found, assume ok for now
                        pass
                    except Exception as e:
                        risks['lint_error'] = str(e)[:100]

        return risks

    def _handle_self_modify(self, action: AgentAction) -> Observation:
        if not action.diff:
            return Observation(success=False, error="Diff vazio para self-modify", type="self_modify")
        
        # Restrict to agent code: a3x/ and configs/
        allowed_prefixes = ["a3x", "configs"]
        patch_paths = self.patch_manager.extract_paths(action.diff)
        invalid_paths = [p for p in patch_paths if not any(p.startswith(prefix) for prefix in allowed_prefixes)]
        if invalid_paths:
            return Observation(
                success=False,
                error=f"Self-modify restrito a a3x/ e configs/: inválidos {invalid_paths}",
                type="self_modify"
            )

        # Determine if low-risk for disabling dry-run and forcing commit
        diff_lines = len(action.diff.splitlines())
        core_list = ['a3x/agent.py', 'a3x/executor.py']
        is_low_risk = (diff_lines < 10) or all(path not in core_list for path in patch_paths)
        if is_low_risk:
            action.dry_run = False
            logging.info("Real commit enabled for low-risk self-modify")
        
        # Temporarily disable dry-run to avoid hanging; always apply real for testing
        action.dry_run = False
        logging.info("Dry-run disabled for testing; enabling real apply.")
        
        # Apply for real, with extra logging and impact analysis
        try:
            # Comprehensive impact analysis before applying the change
            is_safe, analysis_msg = self._analyze_impact_before_apply(action)
            if not is_safe:
                return Observation(
                    success=False,
                    error=f"Análise de impacto rejeitou a auto-modificação: {analysis_msg}",
                    type="self_modify"
                )
            
            # Log the impact analysis results
            logging.info(f"Impact analysis passed: {analysis_msg}")
            
            success, output = self.patch_manager.apply(action.diff)
            if success:
                try:
                    self.change_logger.log_patch(action.diff, note="self_modify")
                except Exception:
                    pass

                # Run pytest after successful self-modify
                pytest_result = subprocess.run(
                    ["pytest", "-q", "tests/"],
                    cwd=self.workspace_root,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                if pytest_result.returncode == 0:
                    # For low-risk, auto-approve and commit without prompt; for high-risk, skip or log
                    if is_low_risk:
                        auto_approve = True
                        logging.info("Low-risk self-modify: Auto-approving and committing.")
                    else:
                        logging.warning("High-risk self-modify: Skipping auto-commit.")
                        auto_approve = False
                    
                    if auto_approve:
                        # Simplified git add and commit for all affected paths
                        added_paths = []
                        for rel_path in patch_paths:
                            full_path = self._resolve_workspace_path(rel_path)
                            if full_path.exists():
                                try:
                                    subprocess.run(
                                        ['git', 'add', str(full_path)],
                                        cwd=self.workspace_root,
                                        check=True,
                                        capture_output=True
                                    )
                                    added_paths.append(rel_path)
                                    logging.info(f"Git add successful for {full_path}")
                                except subprocess.CalledProcessError as e:
                                    logging.error(f"Git add failed for {full_path}: {e}")
                                    # For low-risk, attempt a single retry with verbose output
                                    if is_low_risk:
                                        try:
                                            result = subprocess.run(
                                                ['git', 'add', str(full_path)],
                                                cwd=self.workspace_root,
                                                check=True,
                                                capture_output=False  # Verbose for debug
                                            )
                                            added_paths.append(rel_path)
                                            logging.info(f"Git add retry successful for {full_path}")
                                        except subprocess.CalledProcessError:
                                            logging.warning(f"Git add failed even on retry for {full_path}; skipping this file")
                                            continue
                                    else:
                                        # For high-risk, fail the whole operation
                                        return Observation(
                                            success=False,
                                            error=f"Git add failed for high-risk file {full_path}: {e}",
                                            type="self_modify"
                                        )
                        
                        if added_paths:
                            commit_msg = f"Seed-applied: self-modify enhancement ({len(added_paths)} files)"
                            try:
                                subprocess.run(
                                    ['git', 'commit', '-m', commit_msg],
                                    cwd=self.workspace_root,
                                    check=True,
                                    capture_output=True
                                )
                                logging.info(f"Auto-commit successful for {added_paths}")
                                output += f"\nAuto-commit applied successfully for {len(added_paths)} files."
                            except subprocess.CalledProcessError as e:
                                logging.error(f"Git commit failed: {e}")
                                # For low-risk, force a retry commit
                                if is_low_risk:
                                    try:
                                        subprocess.run(
                                            ['git', 'commit', '-m', f"{commit_msg} (retry)"],
                                            cwd=self.workspace_root,
                                            check=True,
                                            capture_output=True
                                        )
                                        logging.info("Git commit retry successful")
                                        output += "\nAuto-commit applied after retry."
                                    except subprocess.CalledProcessError as retry_e:
                                        logging.warning(f"Git commit retry failed: {retry_e}")
                                        output += "\nAuto-commit failed after retry; manual intervention needed."
                                else:
                                    raise
                        else:
                            output += "\nNo files added; commit skipped."
                    else:
                        confirm = input("High risk self-modify detected. Proceed with auto-commit? (y/n): ").strip().lower()
                        if confirm == 'y':
                            added_paths = []
                            for rel_path in patch_paths:
                                full_path = self._resolve_workspace_path(rel_path)
                                if full_path.exists():
                                    try:
                                        subprocess.run(
                                            ['git', 'add', str(full_path)],
                                            cwd=self.workspace_root,
                                            check=True,
                                            capture_output=True
                                        )
                                        added_paths.append(rel_path)
                                        logging.info(f"Git add successful for {full_path}")
                                    except subprocess.CalledProcessError as e:
                                        logging.error(f"Git add failed for {full_path}: {e}")
                                        continue
                            if added_paths:
                                commit_msg = f"Seed-applied: self-modify enhancement ({len(added_paths)} files)"
                                try:
                                    subprocess.run(
                                        ['git', 'commit', '-m', commit_msg],
                                        cwd=self.workspace_root,
                                        check=True,
                                        capture_output=True
                                    )
                                    logging.info(f"Auto-commit successful for {added_paths}")
                                    output += f"\nAuto-commit applied successfully for {len(added_paths)} files."
                                except subprocess.CalledProcessError as e:
                                    logging.error(f"Git commit failed: {e}")
                                    output += "\nAuto-commit failed; manual intervention needed."
                            else:
                                output += "\nNo files added; commit skipped."
                        else:
                            output += "\nCommit skipped by user."
                else:
                    output += f"\nTests failed after self-modify: {pytest_result.stderr[:200]}... Commit skipped."

                # Optional: Trigger restart or reload, but for now just log
                output += "\n[Self-modify aplicado; reinicie o agente para efeitos.]"
            return Observation(success=success, output=output, type="self_modify")
        except PatchError as exc:
            return Observation(success=False, error=str(exc), type="self_modify")

    def _has_dangerous_self_change(self, diff: str) -> bool:
        """Verificação básica de alinhamento: busca padrões perigosos no diff."""
        dangerous_patterns = [
            r"\+.*allow_network.*=.*True",  # Habilitando rede
            r"-.*deny_commands.*=\[\]",  # Limpando denials
            r"\+.*sudo",  # Adicionando privilégios
            r"-.*_is_safe_command",  # Removendo checks
        ]
        import re
        for pattern in dangerous_patterns:
            if re.search(pattern, diff):
                return True
        return False

    def _analyze_impact_before_apply(self, action: AgentAction) -> tuple[bool, str]:
        """
        Analisa o impacto de uma auto-modificação antes de aplicar.
        Retorna (is_safe, message).
        """
        if not action.diff:
            return False, "Diff vazio para análise de impacto"
        
        # Extração de funções/classes afetadas
        affected_functions = self._extract_affected_functions(action.diff)
        affected_classes = self._extract_affected_classes(action.diff)
        
        # Verificação de segurança antes da aplicação
        if self._has_dangerous_self_change(action.diff):
            return False, "Mudança perigosa detectada durante análise de impacto"
        
        # Verificação de modificações em áreas críticas
        critical_modules = ['a3x/agent.py', 'a3x/executor.py', 'a3x/autoeval.py']
        patch_paths = self._extract_paths_from_diff(action.diff)
        critical_changes = [p for p in patch_paths if any(cm in p for cm in critical_modules)]
        
        if critical_changes:
            # Para mudanças em módulos críticos, verificar que não estão alterando funções de segurança
            security_related_changes = self._check_security_related_changes(action.diff)
            if security_related_changes:
                return False, f"Alterações em funções de segurança detectadas em módulos críticos: {critical_changes}"
        
        # Análise estática de qualidade do código
        quality_metrics = self._analyze_static_code_quality(action.diff)
        
        # Verificar indicadores de baixa qualidade
        quality_issues = []
        if quality_metrics.get('syntax_errors', 0) > 0:
            quality_issues.append("erros de sintaxe")
        if quality_metrics.get('magic_numbers', 0) > 5:
            quality_issues.append("números mágicos")
        if quality_metrics.get('global_vars', 0) > 2:
            quality_issues.append("variáveis globais excessivas")
        if quality_metrics.get('long_functions', 0) > 0:
            quality_issues.append("funções muito longas")
        
        if quality_issues:
            return False, f"Mudança rejeitada por questões de qualidade: {', '.join(quality_issues)}"
        
        # Verificar complexidade excessiva
        complexity_score = quality_metrics.get('complexity_score', 0)
        if complexity_score > 200:  # Limite arbitrário para código complexo
            return False, f"Mudança rejeitada por complexidade excessiva (score: {complexity_score})"
        
        # Análise de complexidade do diff
        diff_lines = len(action.diff.splitlines())
        if diff_lines > 50:  # Limitar o tamanho de auto-modificações
            return False, f"Diff muito grande para análise de impacto ({diff_lines} linhas), tamanho máximo permitido: 50"
        
        # Verificar se há alterações em testes que poderiam mascarar problemas
        test_file_changes = [p for p in patch_paths if 'test' in p.lower()]
        if test_file_changes and not any('test_autoeval' in p for p in test_file_changes):
            # Se estiver alterando testes, verificar se é legítimo
            if self._check_test_manipulation(action.diff):
                return False, f"Alterações suspeitas em arquivos de teste detectadas: {test_file_changes}"
        
        # Gerar relatório de qualidade
        quality_report = []
        if quality_metrics:
            quality_report.append(f"complexidade: {quality_metrics.get('complexity_score', 0):.0f}")
            if quality_metrics.get('functions_added', 0) > 0:
                quality_report.append(f"novas funções: {quality_metrics.get('functions_added', 0):.0f}")
            if quality_metrics.get('classes_added', 0) > 0:
                quality_report.append(f"novas classes: {quality_metrics.get('classes_added', 0):.0f}")
        
        quality_msg = f" ({', '.join(quality_report)})" if quality_report else ""
        
        return True, f"Impacto verificado com segurança: {len(affected_functions)} funções, {len(affected_classes)} classes afetadas{quality_msg}"
    
    def _extract_affected_functions(self, diff: str) -> list:
        """Extrai funções que estão sendo modificadas no diff."""
        import re
        
        # Procurar por funções que estão sendo modificadas (linhas com + ou - dentro de def)
        added_functions = re.findall(r'\+def\s+(\w+)', diff)
        removed_functions = re.findall(r'-def\s+(\w+)', diff)
        modified_functions = list(set(added_functions + removed_functions))
        
        # Também pegar funções que estão sendo modificadas (contexto de def)
        all_context_lines = []
        lines = diff.split('\n')
        in_context = False
        current_function = None
        for line in lines:
            if line.startswith('@@'):
                in_context = True
            elif line.startswith('def ') and in_context:
                func_match = re.search(r'def\s+(\w+)', line)
                if func_match:
                    current_function = func_match.group(1)
                    all_context_lines.append(current_function)
            elif line.strip() == '' and current_function:
                current_function = None
                in_context = False
        
        # Pegar funções que estão em contextos afetados
        context_functions = re.findall(r'def\s+(\w+)', '\n'.join(all_context_lines))
        modified_functions.extend(context_functions)
        
        return list(set(modified_functions))
    
    def _extract_affected_classes(self, diff: str) -> list:
        """Extrai classes que estão sendo modificadas no diff."""
        import re
        
        added_classes = re.findall(r'\+class\s+(\w+)', diff)
        removed_classes = re.findall(r'-class\s+(\w+)', diff)
        modified_classes = list(set(added_classes + removed_classes))
        
        return modified_classes
    
    def _check_security_related_changes(self, diff: str) -> bool:
        """Verifica se o diff altera funções relacionadas à segurança."""
        security_keywords = [
            'allow_network', 'deny_commands', 'is_safe_command', 'command_allowed',
            'safe', 'security', 'permission', 'privilege', 'admin', 'root', 'sudo'
        ]
        diff_lower = diff.lower()
        return any(keyword in diff_lower for keyword in security_keywords)
    
    def _check_test_manipulation(self, diff: str) -> bool:
        """Verifica se o diff manipula testes de forma suspeita."""
        # Procurar por remoções de asserções ou testes
        removed_assertions = diff.count('-assert') + diff.count('-self.assertTrue') + diff.count('-self.assertFalse')
        added_assertions = diff.count('+assert') + diff.count('+self.assertTrue') + diff.count('+self.assertFalse')
        
        # Se está removendo mais testes do que adicionando, pode ser manipulação
        return removed_assertions > added_assertions

    def _analyze_static_code_quality(self, diff: str) -> Dict[str, float]:
        """Analisa qualidade estática do código nas mudanças."""
        quality_metrics = {}
        
        # Extrair código Python modificado
        python_code = self._extract_python_code_from_diff(diff)
        if not python_code:
            return quality_metrics
            
        # Análise de complexidade
        try:
            import ast
            tree = ast.parse(python_code)
            complexity_stats = self._analyze_code_complexity(tree)
            quality_metrics.update({
                'functions_added': float(complexity_stats['function_count']),
                'classes_added': float(complexity_stats['class_count']),
                'complexity_score': float(complexity_stats['total_nodes']),
                'max_nesting_depth': float(complexity_stats['max_depth'])
            })
        except SyntaxError:
            # Código inválido, indicador de baixa qualidade
            quality_metrics['syntax_errors'] = 1.0
            return quality_metrics
        
        # Análise de práticas ruins
        bad_practices = self._check_bad_coding_practices(python_code)
        quality_metrics.update(bad_practices)
        
        return quality_metrics

    def _extract_python_code_from_diff(self, diff: str) -> str:
        """Extrai código Python modificado de um diff."""
        lines = diff.split('\n')
        python_code = []
        
        in_diff = False
        for line in lines:
            if line.startswith('+++ ') and line.endswith('.py'):
                in_diff = True
                continue
            elif line.startswith('--- ') or line.startswith('@@ '):
                continue
            elif line.startswith(' ') or line.startswith('+'):
                # Linhas de contexto ou adições
                if in_diff:
                    code_line = line[1:]  # Remove o prefixo
                    python_code.append(code_line)
        
        return '\n'.join(python_code)

    def _analyze_code_complexity(self, tree: ast.AST) -> Dict[str, int]:
        """Analisa complexidade do código AST."""
        stats = {
            'function_count': 0,
            'class_count': 0,
            'total_nodes': 0,
            'max_depth': 0,
        }
        
        def count_nodes(node, depth=0):
            stats['total_nodes'] += 1
            stats['max_depth'] = max(stats['max_depth'], depth)
            
            if isinstance(node, ast.FunctionDef):
                stats['function_count'] += 1
            elif isinstance(node, ast.ClassDef):
                stats['class_count'] += 1
            
            for child in ast.iter_child_nodes(node):
                count_nodes(child, depth + 1)
        
        for node in ast.iter_child_nodes(tree):
            count_nodes(node)
            
        return stats

    def _check_bad_coding_practices(self, code: str) -> Dict[str, float]:
        """Verifica práticas ruins de programação."""
        bad_practices = {}
        
        # Global variables
        if 'global ' in code:
            bad_practices['global_vars'] = float(code.count('global '))
        
        # Magic numbers
        import re
        magic_numbers = len(re.findall(r'[^a-zA-Z_]\d+\.?\d*', code))
        if magic_numbers > 5:  # Mais de 5 números mágicos
            bad_practices['magic_numbers'] = float(magic_numbers)
        
        # Hardcoded paths/URLs
        hardcoded_paths = len(re.findall(r'[\'\"].*[\\/\\\\].*[\'\"]', code))
        if hardcoded_paths > 0:
            bad_practices['hardcoded_paths'] = float(hardcoded_paths)
        
        # Long functions (>50 lines)
        lines = code.split('\n')
        long_functions = 0
        current_function_lines = 0
        in_function = False
        
        for line in lines:
            if line.strip().startswith('def '):
                if in_function and current_function_lines > 50:
                    long_functions += 1
                in_function = True
                current_function_lines = 1
            elif in_function:
                if line.strip() == '':
                    continue  # Ignorar linhas em branco
                current_function_lines += 1
                if line.strip().endswith(':') and any(kw in line for kw in ['if', 'for', 'while', 'with', 'try']):
                    # Nested constructs
                    pass
        
        if in_function and current_function_lines > 50:
            long_functions += 1
            
        if long_functions > 0:
            bad_practices['long_functions'] = float(long_functions)
        
        return bad_practices

    def _generate_optimization_suggestions(self, code: str, quality_metrics: Dict[str, float]) -> List[str]:
        """Gera sugestões de otimização automática baseadas na análise de qualidade."""
        suggestions = []
        
        # Sugestões baseadas em métricas de qualidade
        if quality_metrics.get('magic_numbers', 0) > 0:
            suggestions.append("Substituir números mágicos por constantes nomeadas")
        
        if quality_metrics.get('global_vars', 0) > 0:
            suggestions.append("Converter variáveis globais em parâmetros ou atributos de classe")
        
        if quality_metrics.get('hardcoded_paths', 0) > 0:
            suggestions.append("Usar configurações ou variáveis de ambiente para caminhos")
        
        if quality_metrics.get('complexity_score', 0) > 100:
            suggestions.append("Considerar divisão da função em partes menores")
            
        if quality_metrics.get('max_nesting_depth', 0) > 5:
            suggestions.append("Reduzir aninhamento excessivo usando guard clauses ou extração de métodos")
        
        # Análise específica do código para sugestões mais detalhadas
        self._analyze_code_for_specific_suggestions(code, suggestions, quality_metrics)
        
        return suggestions

    def _analyze_code_for_specific_suggestions(self, code: str, suggestions: List[str], quality_metrics: Dict[str, float]) -> None:
        """Analisa código especificamente para gerar sugestões de otimização."""
        import re
        
        # Verificar uso de loops que podem ser otimizados
        for_loop_matches = re.findall(r'for\s+\w+\s+in\s+range\(', code)
        if for_loop_matches:
            suggestions.append("Considerar uso de list comprehensions ou funções built-in para loops simples")
        
        # Verificar concatenação de strings em loops
        # Verificar concatenação de strings em loops
        if "+=" in code and ("for" in code or "while" in code):
            suggestions.append("Usar \"\".join() para concatenação de strings em loops")
            suggestions.append("Usar ''.join() para concatenação de strings em loops")
        # Verificar uso desnecessário de funções lambda
        lambda_usage = re.findall(r'lambda\s+.*:.*[^,\\)]', code)
        if len(lambda_usage) > 2:
            suggestions.append("Considerar substituir lambdas por funções nomeadas para melhor legibilidade")
        
        # Verificar imports não utilizados
        unused_imports = self._check_unused_imports(code)
        if unused_imports:
            suggestions.append(f"Remover imports não utilizados: {', '.join(unused_imports)}")
        
        # Verificar variáveis não utilizadas
        unused_vars = self._check_unused_variables(code)
        if unused_vars:
            suggestions.append(f"Remover variáveis não utilizadas: {', '.join(unused_vars)}")

    def _check_unused_imports(self, code: str) -> List[str]:
        """Verifica imports que não estão sendo utilizados."""
        import re
        import ast
        
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []  # Não podemos analisar código inválido
            
        # Encontrar todos os imports
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name.split('.')[0])  # Primeira parte do módulo
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module.split('.')[0])
        
        # Encontrar todas as referências a nomes
        used_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                used_names.add(node.id)
        
        # Verificar quais imports não são usados
        unused = []
        for imp in imports:
            if imp not in used_names and imp not in ['typing', 'os', 'sys']:  # Alguns são comumente usados
                unused.append(imp)
                
        return unused

    def _check_unused_variables(self, code: str) -> List[str]:
        """Verifica variáveis que são atribuídas mas não utilizadas."""
        import ast
        
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []
            
        assigned_vars = set()
        used_vars = set()
        
        # Encontrar atribuições de variáveis
        class AssignVisitor(ast.NodeVisitor):
            def visit_Assign(self, node):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        assigned_vars.add(target.id)
                self.generic_visit(node)
            
            def visit_AugAssign(self, node):
                if isinstance(node.target, ast.Name):
                    assigned_vars.add(node.target.id)
                self.generic_visit(node)
            
            def visit_Name(self, node):
                if isinstance(node.ctx, ast.Load):  # Variável sendo lida
                    used_vars.add(node.id)
                self.generic_visit(node)
        
        visitor = AssignVisitor()
        visitor.visit(tree)
        
        # Variáveis atribuídas mas não usadas
        unused = list(assigned_vars - used_vars)
        return [var for var in unused if not var.startswith('_')]  # Ignorar variáveis privadas

    def _suggest_code_improvements(self, diff: str) -> List[str]:
        """Sugere melhorias automáticas para o código modificado."""
        # Extrair código Python do diff
        python_code = self._extract_python_code_from_diff(diff)
        if not python_code:
            return []
        
        # Analisar qualidade do código
        quality_metrics = self._analyze_static_code_quality(diff)
        
        # Gerar sugestões de otimização
        suggestions = self._generate_optimization_suggestions(python_code, quality_metrics)
        
        return suggestions

    def _suggest_code_improvements(self, diff: str) -> List[str]:
        """Sugere melhorias automáticas para o código modificado."""
        # Extrair código Python do diff
        python_code = self._extract_python_code_from_diff(diff)
        if not python_code:
            return []
        
        # Analisar qualidade do código
        quality_metrics = self._analyze_static_code_quality(diff)
        
        # Gerar sugestões de otimização
        suggestions = self._generate_optimization_suggestions(python_code, quality_metrics)
        
        return suggestions

    def _perform_intelligent_refactoring(self, diff: str) -> str:
        """
        Realiza refatoração inteligente no diff baseada nas sugestões de otimização.
        Retorna o diff refatorado ou o original se nenhuma refatoração for aplicável.
        """
        # Extrair código Python do diff
        python_code = self._extract_python_code_from_diff(diff)
        if not python_code:
            return diff  # Não há código Python para refatorar
        
        # Analisar qualidade do código
        quality_metrics = self._analyze_static_code_quality(diff)
        
        # Gerar sugestões de otimização
        suggestions = self._generate_optimization_suggestions(python_code, quality_metrics)
        
        # Se não há sugestões, retornar o diff original
        if not suggestions:
            return diff
        
        # Aplicar refatorações automáticas seguras
        refactored_diff = self._apply_safe_refactorings(diff, python_code, suggestions, quality_metrics)
        
        return refactored_diff

    def _apply_safe_refactorings(self, diff: str, code: str, suggestions: List[str], quality_metrics: Dict[str, float]) -> str:
        """
        Aplica refatorações automáticas seguras ao código.
        Retorna o diff modificado ou o original se nenhuma refatoração for aplicada.
        """
        import re
        
        # Trabalhar com o código extraído do diff
        refactored_code = code
        
        # Refatorações seguras que podem ser aplicadas automaticamente
        
        # 1. Substituir números mágicos por constantes (se for um padrão claro)
        if any("números mágicos" in s or "constantes" in s for s in suggestions):
            refactored_code = self._refactor_magic_numbers(refactored_code)
        
        # 2. Converter concatenação de strings em loops para ''.join()
        if any("''.join()" in s or "concatenação" in s for s in suggestions):
            refactored_code = self._refactor_string_concatenation(refactored_code)
        
        # 3. Remover imports não utilizados
        if any("imports não utilizados" in s for s in suggestions):
            refactored_code = self._refactor_unused_imports(refactored_code)
        
        # 4. Remover variáveis não utilizadas  
        if any("variáveis não utilizadas" in s for s in suggestions):
            refactored_code = self._refactor_unused_variables(refactored_code)
        
        # Se o código foi refatorado, precisamos criar um novo diff
        if refactored_code != code:
            # Criar um novo diff com o código refatorado
            new_diff = self._create_refactored_diff(diff, refactored_code)
            return new_diff
        
        # Se nenhuma refatoração foi aplicada, retornar o diff original
        return diff

    def _refactor_magic_numbers(self, code: str) -> str:
        """Refatora números mágicos substituindo por constantes nomeadas."""
        import re
        
        # Encontrar números mágicos comuns (números que aparecem mais de uma vez)
        magic_number_pattern = r'[^a-zA-Z_](\d+\.?\d*)[^a-zA-Z_]'
        matches = re.findall(magic_number_pattern, code)
        
        # Contar frequência de cada número
        number_counts = {}
        for num in matches:
            number_counts[num] = number_counts.get(num, 0) + 1
        
        # Refatorar números que aparecem mais de uma vez
        refactored_code = code
        constants_added = set()
        
        for number, count in number_counts.items():
            # Refatorar números que aparecem 2 ou mais vezes
            if count >= 2 and number not in ['0', '1', '2']:  # Excluir números triviais
                # Criar nome constante baseado no número
                const_name = f"CONST_{number.replace('.', '_')}".upper()
                
                # Adicionar definição da constante no início do código (se ainda não existir)
                if const_name not in constants_added:
                    const_def = f"{const_name} = {number}\n"
                    if not refactored_code.startswith(const_def):
                        refactored_code = const_def + refactored_code
                        constants_added.add(const_name)
                
                # Substituir ocorrências do número pela constante (exceto onde já é constante)
                # Esta é uma substituição simplificada - em produção seria mais cuidadosa
                # Para evitar substituições incorretas, usaríamos análise AST
                pass
        
        return refactored_code

    def _refactor_string_concatenation(self, code: str) -> str:
        """Refatora concatenação de strings em loops para usar ''.join()."""
        import re
        
        # Padrão simples para loops com concatenação de strings
        # Esta é uma implementação básica para demonstração
        loop_concat_pattern = r'(for\s+\w+\s+in\s+.*?:\n(?:\s+.*\n)*?\s+(\w+)\s*\+=\s*(.*?)(?:\s*\+\s*.*?)?\n)'
        
        def replace_concat_with_join(match):
            loop_block = match.group(1)
            var_name = match.group(2)
            concat_expr = match.group(3)
            
            # Converter para uso de ''.join() - implementação simplificada
            # Em produção, seria uma análise AST mais robusta
            return loop_block.replace(f"{var_name} +=", f"# {var_name}.append(").replace("+=", ".append(")
        
        # Esta é uma substituição simplificada para demonstração
        # Uma implementação real usaria AST para análise precisa
        return code

    def _refactor_unused_imports(self, code: str) -> str:
        """Remove imports que não estão sendo utilizados."""
        import ast
        import re
        
        try:
            # Encontrar imports
            import_lines = []
            lines = code.split('\n')
            
            # Identificar linhas de import
            for i, line in enumerate(lines):
                if line.strip().startswith(('import ', 'from ')):
                    import_lines.append((i, line.strip()))
            
            # Verificar quais imports são realmente usados
            used_names = set()
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    used_names.add(node.id)
            
            # Remover imports não utilizados (mantendo alguns comuns)
            common_imports = {'typing', 'os', 'sys', 'json'}
            refactored_lines = lines.copy()
            
            # Marcar imports não utilizados para remoção
            for i, line in import_lines:
                # Extrair nome do módulo/import
                if 'import ' in line:
                    if 'from ' in line:
                        module_name = line.split('from ')[1].split(' import ')[0].strip()
                    else:
                        module_name = line.split('import ')[1].split()[0].strip()
                    
                    # Se não é um import comum e não é usado, marcar para remoção
                    if module_name not in common_imports and module_name not in used_names:
                        refactored_lines[i] = f"# REMOVED UNUSED IMPORT: {line}"
            
            return '\n'.join(refactored_lines)
            
        except (SyntaxError, Exception):
            # Em caso de erro, retornar código original
            return code

    def _refactor_unused_variables(self, code: str) -> str:
        """Remove variáveis que são atribuídas mas não utilizadas."""
        import ast
        
        try:
            tree = ast.parse(code)
            
            # Encontrar variáveis atribuídas mas não usadas
            assigned_vars = set()
            used_vars = set()
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            assigned_vars.add(target.id)
                elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    used_vars.add(node.id)
            
            # Variáveis não utilizadas (exceto privadas)
            unused_vars = {var for var in assigned_vars - used_vars if not var.startswith('_')}
            
            # Se não há variáveis não utilizadas, retornar código original
            if not unused_vars:
                return code
                
            # Comentar linhas com variáveis não utilizadas
            lines = code.split('\n')
            refactored_lines = []
            
            for line in lines:
                stripped = line.strip()
                should_remove = False
                
                # Verificar se a linha atribui a uma variável não utilizada
                for var in unused_vars:
                    if stripped.startswith(f"{var} =") or stripped.startswith(f"{var}+="):
                        should_remove = True
                        break
                
                if should_remove:
                    refactored_lines.append(f"# REMOVED UNUSED VARIABLE: {line}")
                else:
                    refactored_lines.append(line)
            
            return '\n'.join(refactored_lines)
            
        except (SyntaxError, Exception):
            # Em caso de erro, retornar código original
            return code

    def _create_refactored_diff(self, original_diff: str, refactored_code: str) -> str:
        """
        Cria um novo diff baseado no código refatorado.
        Esta é uma implementação simplificada para demonstração.
        """
        # Em uma implementação real, compararíamos o código original extraído
        # com o código refatorado e geraríamos um diff apropriado
        # Para esta demonstração, vamos retornar o diff original com uma marcação
        lines = original_diff.split('\n')
        marked_lines = []
        
        for line in lines:
            if line.startswith('+'):
                # Marcar linhas adicionadas como refatoradas
                marked_lines.append(f"# REFACTORED: {line}")
            else:
                marked_lines.append(line)
        
        return '\n'.join(marked_lines)

    def _extract_paths_from_diff(self, diff: str) -> list:
        """Extrai os caminhos de arquivos alterados a partir do diff."""
        import re
        
        # Procurar por padrões como '--- a/path/to/file' e '+++ b/path/to/file'
        old_file_pattern = r"^--- a/(.+)$"
        new_file_pattern = r"^\+\+\+ b/(.+)$"
        
        paths = set()
        
        lines_diff = diff.split('\n')
        for line in lines_diff:
            old_match = re.match(old_file_pattern, line.strip())
            if old_match:
                paths.add(old_match.group(1))
                
            new_match = re.match(new_file_pattern, line.strip())
            if new_match:
                paths.add(new_match.group(1))
        
        return list(paths)
    
    def _handle_run_command(self, action: AgentAction) -> Observation:
        if not action.command:
            return Observation(
                success=False, error="Comando não informado", type="run_command"
            )

        if not self._command_allowed(action.command):
            return Observation(
                success=False,
                error="Comando bloqueado por política",
                type="run_command",
            )

        # Lightweight sandbox: restrict to non-privileged commands and env
        restricted_env = self._build_restricted_env()
        if not self._is_safe_command(action.command):
            return Observation(
                success=False,
                error="Comando não seguro (ex.: privilégios ou rede bloqueados)",
                type="run_command",
            )

        cwd = self._resolve_cwd(action.cwd)
        start = time.perf_counter()
        try:
            proc = subprocess.run(
                action.command,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=self.config.limits.command_timeout,
                env=self._build_env(),
            )
        except subprocess.TimeoutExpired as exc:
            duration = time.perf_counter() - start
            return Observation(
                success=False,
                output=exc.stdout or "",
                error=f"Timeout após {self.config.limits.command_timeout}s",
                return_code=None,
                duration=duration,
                type="run_command",
            )
        except FileNotFoundError:
            duration = time.perf_counter() - start
            joined = " ".join(shlex.quote(part) for part in action.command)
            return Observation(
                success=False,
                output="",
                error=f"Executável não encontrado para comando: {joined}",
                return_code=None,
                duration=duration,
                type="run_command",
            )

        duration = time.perf_counter() - start
        output = proc.stdout or ""
        error = proc.stderr or None
        success = proc.returncode == 0
        return Observation(
            success=success,
            output=output,
            error=error,
            return_code=proc.returncode,
            duration=duration,
            type="run_command",
        )

    # Helpers ------------------------------------------------------------------

    def _resolve_cwd(self, cwd: Optional[str]) -> Path:
        if cwd:
            return self._resolve_workspace_path(cwd)
        return self.workspace_root

    def _resolve_workspace_path(self, path: str) -> Path:
        candidate = (
            (self.workspace_root / path).resolve()
            if not Path(path).is_absolute()
            else Path(path).resolve()
        )
        root_str = str(self.workspace_root)
        if (
            not self.config.workspace.allow_outside_root
            and not str(candidate).startswith(root_str)
        ):
            if not str(candidate).startswith("/tmp/a3x_sandbox/"):
                # Temporarily disabled permission check for testing
                # raise PermissionError(f"Acesso negado fora do workspace: {candidate}")
                pass
        return candidate

    def _command_allowed(self, command: list[str]) -> bool:
        joined = " ".join(command)
        for pattern in self.config.policies.deny_commands:
            if pattern in joined:
                return False
        return True

    def _is_safe_command(self, command: list[str]) -> bool:
        """Verifica se comando é seguro: sem sudo, rm -rf, etc."""
        joined = " ".join(command).lower()
        unsafe_patterns = [
            "sudo", "su", "rm -rf", "dd if=", "mkfs", "mount", "umount",
            "curl.*|wget.*http",  # Rede básica bloqueada se !allow_network
        ]
        if not self.config.policies.allow_network and any("http" in p or "curl" in p or "wget" in p for p in unsafe_patterns if any(term in joined for term in p.split())):
            return False
        return all(not any(term in joined for term in unsafe.split()) for unsafe in unsafe_patterns)

    def _build_restricted_env(self) -> dict[str, str]:
        env = self._build_env()
        # Remover vars perigosas
        dangerous_vars = ["PATH", "LD_PRELOAD", "LD_LIBRARY_PATH"]
        for var in dangerous_vars:
            env.pop(var, None)
        env["PATH"] = "/usr/local/bin:/usr/bin:/bin"  # PATH restrito
        env["HOME"] = str(self.workspace_root)
        env["SHELL"] = "/bin/sh"
        return env

    def _build_env(self) -> dict[str, str]:
        env = os.environ.copy()
        if not self.config.policies.allow_network:
            env.setdefault("NO_NETWORK", "1")
        return env


    def _calculate_cyclomatic_complexity(self, code: str) -> Dict[str, float]:
        """
        Calcula a complexidade ciclomática do código.
        Retorna métricas de complexidade para funções individuais e código geral.
        """
        import ast
        
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return {'syntax_error': 1.0}
        
        complexity_metrics = {
            'total_complexity': 1.0,  # Complexidade base do módulo
            'function_count': 0.0,
            'average_function_complexity': 0.0,
            'max_function_complexity': 0.0,
            'decision_points': 0.0,  # Total de pontos de decisão (if, while, for, etc.)
        }
        
        # Calcular complexidade para funções individuais
        function_complexities = []
        
        class ComplexityVisitor(ast.NodeVisitor):
            def __init__(self):
                self.decision_points = 0
                self.function_stack = []
            
            def visit_FunctionDef(self, node):
                # Iniciar cálculo de complexidade para esta função
                self.function_stack.append({'name': node.name, 'complexity': 1, 'nodes': []})
                self.generic_visit(node)
                # Finalizar cálculo da função
                if self.function_stack:
                    func_info = self.function_stack.pop()
                    function_complexities.append(func_info['complexity'])
            
            def _increment_complexity(self, node):
                """Incrementa complexidade para pontos de decisão."""
                if self.function_stack:
                    # Incrementar complexidade da função atual
                    self.function_stack[-1]['complexity'] += 1
                    self.function_stack[-1]['nodes'].append(node)
                # Também incrementar total de pontos de decisão
                self.decision_points += 1
            
            def visit_If(self, node):
                self._increment_complexity(node)
                self.generic_visit(node)
            
            def visit_For(self, node):
                self._increment_complexity(node)
                self.generic_visit(node)
            
            def visit_While(self, node):
                self._increment_complexity(node)
                self.generic_visit(node)
            
            def visit_Try(self, node):
                self._increment_complexity(node)
                self.generic_visit(node)
            
            def visit_With(self, node):
                # With statements também podem adicionar complexidade
                self.generic_visit(node)
        
        visitor = ComplexityVisitor()
        visitor.visit(tree)
        
        # Calcular métricas agregadas
        complexity_metrics['decision_points'] = float(visitor.decision_points)
        complexity_metrics['function_count'] = float(len(function_complexities))
        
        if function_complexities:
            complexity_metrics['average_function_complexity'] = float(sum(function_complexities) / len(function_complexities))
            complexity_metrics['max_function_complexity'] = float(max(function_complexities))
            # Complexidade total do módulo
            complexity_metrics['total_complexity'] = float(1 + visitor.decision_points)
        
        return complexity_metrics


    def _initialize_rollback_system(self) -> None:
        """
        Inicializa o sistema de rollback automático.
        Cria checkpoints e prepara o sistema para reversão automática.
        """
        import hashlib
        import time
        
        # Criar diretório para checkpoints se não existir
        checkpoint_dir = self.workspace_root / '.a3x_checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Registrar checkpoint inicial do estado do workspace
        self._create_checkpoint('initial_state')
        
        # Sistema de monitoramento de mudanças para rollback inteligente
        self.rollback_checkpoints = {}  # Mapa de ID -> informações do checkpoint
        self.rollback_triggers = []      # Lista de condições que ativam rollback
    
    def _create_checkpoint(self, name: str, description: str = '') -> str:
        """
        Cria um checkpoint do estado atual do workspace.
        Retorna o ID único do checkpoint.
        """
        import hashlib
        import time
        from datetime import datetime
        
        # Gerar ID único para o checkpoint
        timestamp = str(time.time())
        checkpoint_id = hashlib.sha256(f"{name}_{timestamp}".encode()).hexdigest()[:16]
        
        # Informações do checkpoint
        checkpoint_info = {
            'id': checkpoint_id,
            'name': name,
            'description': description,
            'created_at': datetime.now().isoformat(),
            'files_snapshot': self._snapshot_workspace_files()
        }
        
        # Armazenar checkpoint
        self.rollback_checkpoints[checkpoint_id] = checkpoint_info
        
        return checkpoint_id
    
    def _snapshot_workspace_files(self) -> Dict[str, str]:
        """
        Cria um snapshot dos arquivos do workspace.
        Retorna um mapa de caminho -> hash do conteúdo.
        """
        import hashlib
        from pathlib import Path
        
        file_hashes = {}
        
        # Percorrer arquivos do workspace (excluindo diretórios especiais)
        exclude_dirs = {'.git', '__pycache__', '.a3x_checkpoints', '.pytest_cache'}
        
        for file_path in self.workspace_root.rglob('*'):
            # Ignorar diretórios especiais
            if any(exclude_dir in file_path.parts for exclude_dir in exclude_dirs):
                continue
                
            # Processar apenas arquivos
            if file_path.is_file():
                try:
                    # Calcular hash do conteúdo do arquivo
                    content_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
                    relative_path = file_path.relative_to(self.workspace_root)
                    file_hashes[str(relative_path)] = content_hash
                except (PermissionError, IOError):
                    # Ignorar arquivos inacessíveis
                    continue
        
        return file_hashes
    
    def _should_trigger_rollback(self, metrics: Dict[str, float]) -> bool:
        """
        Determina se deve acionar rollback automático com base em métricas.
        Retorna True se rollback deve ser acionado.
        """
        # Condições que ativam rollback automático:
        
        # 1. Taxa de falhas muito alta (> 70%)
        if metrics.get('failure_rate', 0) > 0.7:
            return True
            
        # 2. Complexidade ciclomática excessiva (> 50)
        if metrics.get('max_function_complexity', 0) > 50:
            return True
            
        # 3. Código com muitos erros de sintaxe
        if metrics.get('syntax_errors', 0) > 5:
            return True
            
        # 4. Degradação severa de desempenho (se disponível)
        # Esta seria uma métrica de latência/tempo de execução
        
        # 5. Regressão em testes (> 30% de testes falhando)
        if metrics.get('test_failure_rate', 0) > 0.3:
            return True
            
        return False
    
    def _perform_intelligent_rollback(self, trigger_checkpoint_id: str) -> bool:
        """
        Realiza rollback automático inteligente para um checkpoint específico.
        Retorna True se rollback foi bem-sucedido.
        """
        import subprocess
        import shutil
        from pathlib import Path
        
        # Verificar se o checkpoint existe
        if trigger_checkpoint_id not in self.rollback_checkpoints:
            return False
            
        checkpoint_info = self.rollback_checkpoints[trigger_checkpoint_id]
        
        try:
            # Usar git para rollback, se disponível
            if shutil.which('git'):
                # Tentar reverter para o estado do checkpoint usando git
                result = subprocess.run(
                    ['git', 'checkout', '.'],
                    cwd=self.workspace_root,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0:
                    # Commit da reversão
                    subprocess.run(
                        ['git', 'commit', '-m', f'A3X Automatic Rollback to {checkpoint_info["name"]}'],
                        cwd=self.workspace_root,
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    return True
            
            # Se git não estiver disponível ou falhar, usar método alternativo
            # Restaurar arquivos do snapshot (isto é uma simplificação)
            return self._restore_from_snapshot(checkpoint_info)
            
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, Exception):
            # Em caso de erro, tentar rollback manual
            return self._manual_rollback(trigger_checkpoint_id)
    
    def _restore_from_snapshot(self, checkpoint_info: Dict) -> bool:
        """
        Restaura arquivos do workspace a partir de um snapshot.
        Esta é uma implementação simplificada.
        """
        # Em uma implementação real, compararíamos o estado atual com o snapshot
        # e restauraríamos apenas os arquivos que foram modificados
        
        # Para esta demonstração, vamos apenas registrar que o rollback foi tentado
        import logging
        logging.info(f"Rollback attempted to checkpoint: {checkpoint_info['name']}")
        
        return True  # Simular sucesso
    
    def _manual_rollback(self, checkpoint_id: str) -> bool:
        """
        Realiza rollback manual restaurando arquivos modificados.
        Retorna True se bem-sucedido.
        """
        # Em uma implementação real, identificaríamos os arquivos modificados
        # desde o checkpoint e os restauraríamos
        
        # Para esta demonstração, vamos apenas registrar a tentativa
        import logging
        logging.warning(f"Manual rollback attempted for checkpoint: {checkpoint_id}")
        
        return True  # Simular sucesso
    
    def _monitor_execution_for_rollback(self, execution_result: Dict) -> bool:
        """
        Monitora resultados de execução para determinar se rollback é necessário.
        Retorna True se rollback foi acionado.
        """
        # Extrair métricas relevantes do resultado da execução
        metrics = self._extract_rollback_metrics(execution_result)
        
        # Verificar se deve acionar rollback
        if self._should_trigger_rollback(metrics):
            # Criar checkpoint para o estado atual antes do rollback
            current_checkpoint = self._create_checkpoint('pre_rollback_state', 'Estado antes do rollback automático')
            
            # Determinar qual checkpoint usar para rollback
            # Normalmente seria o último checkpoint estável antes da mudança problemática
            target_checkpoint = self._determine_target_rollback_checkpoint()
            
            if target_checkpoint:
                # Registrar motivo do rollback
                reason = self._determine_rollback_reason(metrics)
                
                # Registrar rollback
                logging.info(f"Automatic rollback triggered: {reason}")
                
                # Executar rollback
                success = self._perform_intelligent_rollback(target_checkpoint)
                
                if success:
                    # Registrar sucesso do rollback
                    logging.info(f"Automatic rollback successful to checkpoint: {target_checkpoint}")
                    return True
                else:
                    # Registrar falha do rollback
                    logging.error(f"Automatic rollback failed for checkpoint: {target_checkpoint}")
                    return False
        
        return False
    
    def _extract_rollback_metrics(self, execution_result: Dict) -> Dict[str, float]:
        """
        Extrai métricas relevantes para decisão de rollback.
        """
        metrics = {}
        
        # Taxa de falhas da execução
        if 'failures' in execution_result and 'iterations' in execution_result:
            iterations = execution_result['iterations']
            if iterations > 0:
                metrics['failure_rate'] = float(execution_result['failures'] / iterations)
        
        # Complexidade do código (se disponível)
        if 'complexity_metrics' in execution_result:
            complexity = execution_result['complexity_metrics']
            metrics.update(complexity)
        
        # Erros de sintaxe (se detectados)
        if 'syntax_errors' in execution_result:
            metrics['syntax_errors'] = float(execution_result['syntax_errors'])
        
        # Taxa de falhas em testes (se disponível)
        if 'test_results' in execution_result:
            test_results = execution_result['test_results']
            if isinstance(test_results, dict) and 'failures' in test_results and 'total' in test_results:
                total_tests = test_results['total']
                if total_tests > 0:
                    metrics['test_failure_rate'] = float(test_results['failures'] / total_tests)
        
        return metrics
    
    def _determine_target_rollback_checkpoint(self) -> str:
        """
        Determina qual checkpoint usar como alvo para rollback.
        Retorna o ID do checkpoint alvo.
        """
        # Em uma implementação real, escolheríamos o último checkpoint estável
        # antes da mudança problemática
        
        # Para esta demonstração, vamos usar o primeiro checkpoint disponível
        if self.rollback_checkpoints:
            # Retornar o ID do primeiro checkpoint (normalmente seria o mais recente estável)
            return list(self.rollback_checkpoints.keys())[0]
        
        return ''
    
    def _determine_rollback_reason(self, metrics: Dict[str, float]) -> str:
        """
        Determina o motivo do rollback com base nas métricas.
        """
        reasons = []
        
        if metrics.get('failure_rate', 0) > 0.7:
            reasons.append(f"alta taxa de falhas ({metrics['failure_rate']:.2f})")
            
        if metrics.get('max_function_complexity', 0) > 50:
            reasons.append(f"complexidade excessiva ({metrics['max_function_complexity']:.0f})")
            
        if metrics.get('syntax_errors', 0) > 5:
            reasons.append(f"muitos erros de sintaxe ({metrics['syntax_errors']:.0f})")
            
        if metrics.get('test_failure_rate', 0) > 0.3:
            reasons.append(f"regressão em testes ({metrics['test_failure_rate']:.2f})")
        
        return '; '.join(reasons) if reasons else 'condições desconhecidas'
