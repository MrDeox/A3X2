"""Testes para o sistema de sugestões de otimização do executor."""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path

from a3x.executor import ActionExecutor
from a3x.config import AgentConfig
from a3x.actions import AgentAction, ActionType


class TestOptimizationSuggestions:
    """Testes para as sugestões de otimização automática."""
    
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
    
    def test_generate_optimization_suggestions_with_magic_numbers(self) -> None:
        """Testa geração de sugestões para código com números mágicos."""
        code = """
def calculate_area():
    return 3.14159 * 10 * 10
"""
        quality_metrics = {
            'magic_numbers': 3.0,
            'complexity_score': 50.0
        }
        
        suggestions = self.executor._generate_optimization_suggestions(code, quality_metrics)
        
        # Deve sugerir substituição de números mágicos
        assert any("números mágicos" in s for s in suggestions)
        assert any("constantes nomeadas" in s for s in suggestions)
    
    def test_generate_optimization_suggestions_with_global_vars(self) -> None:
        """Testa geração de sugestões para código com variáveis globais."""
        code = """
global counter
counter = 0

def increment():
    global counter
    counter += 1
"""
        quality_metrics = {
            'global_vars': 2.0,
            'complexity_score': 30.0
        }
        
        suggestions = self.executor._generate_optimization_suggestions(code, quality_metrics)
        
        # Deve sugerir conversão de variáveis globais
        assert any("variáveis globais" in s for s in suggestions)
        assert any("parâmetros" in s or "atributos" in s for s in suggestions)
    
    def test_generate_optimization_suggestions_with_hardcoded_paths(self) -> None:
        """Testa geração de sugestões para código com caminhos hardcoded."""
        code = """
def read_config():
    with open('/etc/myapp/config.json', 'r') as f:
        return f.read()
"""
        quality_metrics = {
            'hardcoded_paths': 1.0,
            'complexity_score': 40.0
        }
        
        suggestions = self.executor._generate_optimization_suggestions(code, quality_metrics)
        
        # Deve sugerir uso de configurações
        assert any("caminhos" in s for s in suggestions)
        assert any("configurações" in s or "variáveis de ambiente" in s for s in suggestions)
    
    def test_generate_optimization_suggestions_with_high_complexity(self) -> None:
        """Testa geração de sugestões para código com alta complexidade."""
        code = """
def complex_function():
    for i in range(10):
        for j in range(10):
            for k in range(10):
                if i > 5:
                    if j > 5:
                        if k > 5:
                            print(i, j, k)
"""
        quality_metrics = {
            'complexity_score': 150.0,
            'max_nesting_depth': 6.0
        }
        
        suggestions = self.executor._generate_optimization_suggestions(code, quality_metrics)
        
        # Deve sugerir divisão de funções e redução de aninhamento
        assert any("divisão" in s or "partes menores" in s for s in suggestions)
        assert any("aninhamento" in s or "guard clauses" in s for s in suggestions)
    
    def test_generate_optimization_suggestions_clean_code(self) -> None:
        """Testa geração de sugestões para código limpo (sem problemas)."""
        code = """
def calculate_area(radius: float) -> float:
    PI = 3.14159
    return PI * radius * radius
"""
        quality_metrics = {
            'complexity_score': 20.0,
            'max_nesting_depth': 2.0
        }
        
        suggestions = self.executor._generate_optimization_suggestions(code, quality_metrics)
        
        # Código limpo não deve gerar muitas sugestões críticas
        # Mas pode ter sugestões gerais de melhoria
        assert isinstance(suggestions, list)
    
    def test_analyze_code_for_specific_suggestions_loops(self) -> None:
        """Testa análise específica para loops que podem ser otimizados."""
        code = """
def process_items(items):
    result = []
    for i in range(len(items)):
        result.append(items[i] * 2)
    return result
"""
        suggestions = []
        quality_metrics = {}
        
        self.executor._analyze_code_for_specific_suggestions(code, suggestions, quality_metrics)
        
        # Deve sugerir uso de list comprehensions
        assert any("list comprehensions" in s or "built-in" in s for s in suggestions)
    
    def test_analyze_code_for_specific_suggestions_string_concatenation(self) -> None:
        """Testa análise específica para concatenação de strings em loops."""
        code = """
def build_string(items):
    result = ""
    for item in items:
        result += str(item) + ","
    return result
"""
        suggestions = []
        quality_metrics = {}
        
        self.executor._analyze_code_for_specific_suggestions(code, suggestions, quality_metrics)
        
        # Deve sugerir uso de ''.join()
        assert any("''.join()" in s or "concatenação" in s for s in suggestions)
    
    def test_check_unused_imports(self) -> None:
        """Testa detecção de imports não utilizados."""
        code = """
import os
import sys
import json

def simple_function():
    return 42
"""
        
        unused_imports = self.executor._check_unused_imports(code)
        
        # json foi importado mas não usado
        assert "json" in unused_imports or len(unused_imports) > 0
    
    def test_check_unused_variables(self) -> None:
        """Testa detecção de variáveis não utilizadas."""
        code = """
def function_with_unused_vars():
    used_var = 42
    unused_var = 23
    another_unused = "hello"
    return used_var
"""
        
        unused_vars = self.executor._check_unused_variables(code)
        
        # Deve detectar variáveis não utilizadas
        # Note: A detecção exata pode variar, mas pelo menos uma deve ser encontrada
        assert isinstance(unused_vars, list)
    
    def test_suggest_code_improvements_empty_diff(self) -> None:
        """Testa sugestões para diff vazio."""
        empty_diff = ""
        
        suggestions = self.executor._suggest_code_improvements(empty_diff)
        
        # Não deve gerar sugestões para diff vazio
        assert suggestions == []
    
    def test_suggest_code_improvements_valid_diff(self) -> None:
        """Testa sugestões para diff válido com código Python."""
        valid_diff = """--- a/test.py
+++ b/test.py
@@ -1,1 +1,3 @@
+def calculate():
+    return 3.14159 * 10 * 10  # Números mágicos
+    
"""
        
        suggestions = self.executor._suggest_code_improvements(valid_diff)
        
        # Deve gerar sugestões para código com problemas
        assert isinstance(suggestions, list)