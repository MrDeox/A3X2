"""Testes para o sistema de análise de impacto do executor."""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path

from a3x.executor import ActionExecutor
from a3x.config import AgentConfig, WorkspaceConfig, AuditConfig
from a3x.actions import AgentAction, ActionType


class TestExecutorImpactAnalysis:
    """Testes para as funções de análise de impacto."""
    
    def setup_method(self) -> None:
        """Configuração antes de cada teste."""
        self.workspace_path = Path("/tmp/test_workspace")
        self.config = AgentConfig(
            llm=Mock(),  # Mock since we don't need LLM for executor tests
            workspace=WorkspaceConfig(root=self.workspace_path),
            limits=Mock(),
            tests=Mock(),
            policies=Mock(),
            goals=Mock(),
            loop=Mock(),
            audit=AuditConfig()
        )
        self.executor = ActionExecutor(self.config)
    
    def test_analyze_impact_before_apply_with_empty_diff(self) -> None:
        """Testa a análise de impacto com diff vazio."""
        action = AgentAction(type=ActionType.SELF_MODIFY, diff="")
        
        is_safe, message = self.executor._analyze_impact_before_apply(action)
        
        assert is_safe is False
        assert "Diff vazio" in message
    
    def test_analyze_impact_before_apply_dangerous_change(self) -> None:
        """Testa a análise de impacto com mudança perigosa."""
        dangerous_diff = """--- a/test.py
+++ b/test.py
@@ -10,3 +10,3 @@
-    allow_network = False
+    allow_network = True
"""
        action = AgentAction(type=ActionType.SELF_MODIFY, diff=dangerous_diff)
        
        is_safe, message = self.executor._analyze_impact_before_apply(action)
        
        assert is_safe is False
        assert "Mudança perigosa detectada" in message
    
    def test_analyze_impact_before_apply_large_diff(self) -> None:
        """Testa a análise de impacto com diff muito grande."""
        large_diff = "\n".join([f"+line {i}" for i in range(60)])  # More than 50 lines
        
        action = AgentAction(type=ActionType.SELF_MODIFY, diff=large_diff)
        
        is_safe, message = self.executor._analyze_impact_before_apply(action)
        
        assert is_safe is False
        assert "Diff muito grande" in message
    
    def test_analyze_impact_before_apply_safe_change(self) -> None:
        """Testa a análise de impacto com mudança segura."""
        safe_diff = """--- a/test.py
+++ b/test.py
@@ -1,5 +1,5 @@
 def hello():
-    print("Hello")
+    print("Hello, World!")
     return True
"""
        action = AgentAction(type=ActionType.SELF_MODIFY, diff=safe_diff)
        
        is_safe, message = self.executor._analyze_impact_before_apply(action)
        
        assert is_safe is True
        assert "Impacto verificado com segurança" in message
    
    def test_extract_paths_from_diff(self) -> None:
        """Testa a extração de caminhos de arquivos de um diff."""
        diff = """--- a/a3x/test.py
+++ b/a3x/test.py
@@ -1,5 +1,5 @@
 def hello():
-    print("Hello")
+    print("Hello, World!")
     return True
"""
        paths = self.executor._extract_paths_from_diff(diff)
        
        assert "a3x/test.py" in paths
        assert len(paths) == 1
    
    def test_extract_affected_functions(self) -> None:
        """Testa a extração de funções afetadas."""
        diff = """--- a/test.py
+++ b/test.py
@@ -1,10 +1,10 @@
-def old_function():
+def new_function():
     pass

+def added_function():
+    return 42
"""
        functions = self.executor._extract_affected_functions(diff)
        
        assert "new_function" in functions
        assert "added_function" in functions
    
    def test_extract_affected_classes(self) -> None:
        """Testa a extração de classes afetadas."""
        diff = """--- a/test.py
+++ b/test.py
@@ -1,10 +1,10 @@
-class OldClass:
+class NewClass:
     pass

+class AddedClass:
+    def method(self):
+        pass
"""
        classes = self.executor._extract_affected_classes(diff)
        
        assert "NewClass" in classes
        assert "AddedClass" in classes
    
    def test_check_security_related_changes(self) -> None:
        """Testa a detecção de mudanças relacionadas à segurança."""
        security_diff = """--- a/config.py
+++ b/config.py
@@ -5,3 +5,3 @@
-    allow_network = False
+    allow_network = True
"""
        
        has_security_changes = self.executor._check_security_related_changes(security_diff)
        
        assert has_security_changes is True
    
    def test_check_test_manipulation(self) -> None:
        """Testa a detecção de manipulação de testes."""
        # Test that removes assertions without adding new ones
        test_diff = """--- a/test_example.py
+++ b/test_example.py
@@ -10,3 +10,1 @@
-self.assertTrue(result)
-self.assertFalse(condition)
+pass
"""
        
        is_manipulation = self.executor._check_test_manipulation(test_diff)
        
        assert is_manipulation is True
    
    def test_check_test_manipulation_legitimate(self) -> None:
        """Testa que mudanças legítimas em testes não são detectadas como manipulação."""
        # Test that removes one assertion but adds a better one
        test_diff = """--- a/test_example.py
+++ b/test_example.py
@@ -10,2 +10,3 @@
-        self.assertTrue(result)
+        assert result is True
+        assert isinstance(result, bool)
"""
        
        is_manipulation = self.executor._check_test_manipulation(test_diff)
        
        assert is_manipulation is False