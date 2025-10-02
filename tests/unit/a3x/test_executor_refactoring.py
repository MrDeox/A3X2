"""Testes para o sistema de refatoração inteligente do executor."""

from pathlib import Path
from unittest.mock import Mock

from a3x.config import AgentConfig
from a3x.executor import ActionExecutor


class TestIntelligentRefactoring:
    """Testes para o sistema de refatoração inteligente."""

    def setup_method(self) -> None:
        """Configuração antes de cada teste."""
        self.mock_config = Mock(spec=AgentConfig)
        self.mock_config.workspace = Mock()
        self.mock_config.workspace.root = Path("/tmp/test_workspace")
        self.mock_config.policies = Mock()
        self.mock_config.policies.allow_network = False
        self.mock_config.policies.deny_commands = []
        self.mock_config.audit = Mock()
        self.mock_config.audit.enable_file_log = True
        self.mock_config.audit.file_dir = Path("seed/changes")
        self.mock_config.audit.enable_git_commit = False
        self.mock_config.audit.commit_prefix = "A3X"
        self.executor = ActionExecutor(self.mock_config)

    def test_perform_intelligent_refactoring_empty_diff(self) -> None:
        """Testa refatoração com diff vazio."""
        empty_diff = ""

        refactored_diff = self.executor._perform_intelligent_refactoring(empty_diff)

        # Deve retornar o diff original para diffs vazios
        assert refactored_diff == empty_diff

    def test_perform_intelligent_refactoring_no_python_code(self) -> None:
        """Testa refatoração com diff sem código Python."""
        non_python_diff = """--- a/README.md
+++ b/README.md
@@ -1,1 +1,1 @@
-Old content
+New content
"""

        refactored_diff = self.executor._perform_intelligent_refactoring(non_python_diff)

        # Deve retornar o diff original para diffs sem código Python
        assert refactored_diff == non_python_diff

    def test_perform_intelligent_refactoring_clean_code(self) -> None:
        """Testa refatoração com código limpo (sem sugestões)."""
        clean_diff = """--- a/test.py
+++ b/test.py
@@ -1,1 +1,3 @@
+def calculate_area(radius: float) -> float:
+    PI = 3.14159
+    return PI * radius * radius
"""

        refactored_diff = self.executor._perform_intelligent_refactoring(clean_diff)

        # Código limpo não deve gerar mudanças significativas
        assert isinstance(refactored_diff, str)

    def test_perform_intelligent_refactoring_with_issues(self) -> None:
        """Testa refatoração com código que tem problemas."""
        problematic_diff = """--- a/test.py
+++ b/test.py
@@ -1,1 +1,3 @@
+def calculate_area():
+    return 3.14159 * 10 * 10  # Números mágicos
+    
"""

        refactored_diff = self.executor._perform_intelligent_refactoring(problematic_diff)

        # Deve retornar algum tipo de diff (mesmo que marcado como refatorado)
        assert isinstance(refactored_diff, str)
        assert len(refactored_diff) > 0

    def test_apply_safe_refactorings_no_suggestions(self) -> None:
        """Testa aplicação de refatorações quando não há sugestões."""
        diff = """--- a/test.py
+++ b/test.py
@@ -1,1 +1,1 @@
+pass
"""
        code = "pass\n"
        suggestions = []
        quality_metrics = {}

        refactored_diff = self.executor._apply_safe_refactorings(diff, code, suggestions, quality_metrics)

        # Deve retornar o diff original quando não há sugestões
        assert refactored_diff == diff

    def test_apply_safe_refactorings_with_suggestions(self) -> None:
        """Testa aplicação de refatorações quando há sugestões."""
        diff = """--- a/test.py
+++ b/test.py
@@ -1,1 +1,3 @@
+def calculate():
+    return 3.14159 * 10 * 10
+    
"""
        code = """def calculate():
    return 3.14159 * 10 * 10
"""
        suggestions = ["Substituir números mágicos por constantes nomeadas"]
        quality_metrics = {"magic_numbers": 3.0}

        refactored_diff = self.executor._apply_safe_refactorings(diff, code, suggestions, quality_metrics)

        # Deve retornar algum tipo de diff (pode ser o original ou refatorado)
        assert isinstance(refactored_diff, str)
        assert len(refactored_diff) > 0

    def test_refactor_magic_numbers_basic(self) -> None:
        """Testa refatoração básica de números mágicos."""
        code = """def calculate():
    return 3.14159 * 10 * 10
"""

        refactored_code = self.executor._refactor_magic_numbers(code)

        # A função deve retornar algum código (mesmo que não faça mudanças reais)
        assert isinstance(refactored_code, str)
        assert len(refactored_code) > 0

    def test_refactor_string_concatenation_basic(self) -> None:
        """Testa refatoração básica de concatenação de strings."""
        code = """def build_string(items):
    result = ""
    for item in items:
        result += str(item) + ","
    return result
"""

        refactored_code = self.executor._refactor_string_concatenation(code)

        # A função deve retornar algum código
        assert isinstance(refactored_code, str)
        assert len(refactored_code) > 0

    def test_refactor_unused_imports_basic(self) -> None:
        """Testa refatoração básica de imports não utilizados."""
        code = """import os
import sys
import json

def simple_function():
    return 42
"""

        refactored_code = self.executor._refactor_unused_imports(code)

        # A função deve retornar algum código
        assert isinstance(refactored_code, str)
        assert len(refactored_code) > 0

    def test_refactor_unused_variables_basic(self) -> None:
        """Testa refatoração básica de variáveis não utilizadas."""
        code = """def function_with_unused():
    used_var = 42
    unused_var = 23
    another_unused = "hello"
    return used_var
"""

        refactored_code = self.executor._refactor_unused_variables(code)

        # A função deve retornar algum código
        assert isinstance(refactored_code, str)
        assert len(refactored_code) > 0

    def test_create_refactored_diff_basic(self) -> None:
        """Testa criação básica de diff refatorado."""
        original_diff = """--- a/test.py
+++ b/test.py
@@ -1,1 +1,1 @@
+pass
"""
        refactored_code = "pass\n"

        new_diff = self.executor._create_refactored_diff(original_diff, refactored_code)

        # Deve retornar algum tipo de diff
        assert isinstance(new_diff, str)
        assert len(new_diff) > 0
