"""Testes para a análise de complexidade ciclomática do executor."""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path

from a3x.executor import ActionExecutor
from a3x.config import AgentConfig


class TestCyclomaticComplexity:
    """Testes para a análise de complexidade ciclomática."""
    
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
    
    def test_calculate_cyclomatic_complexity_empty_code(self) -> None:
        """Testa cálculo de complexidade com código vazio."""
        empty_code = ""
        
        complexity = self.executor._calculate_cyclomatic_complexity(empty_code)
        
        # Deve ter complexidade mínima para código vazio
        assert 'total_complexity' in complexity
        assert complexity['total_complexity'] >= 1.0
    
    def test_calculate_cyclomatic_complexity_syntax_error(self) -> None:
        """Testa cálculo de complexidade com erro de sintaxe."""
        invalid_code = "def invalid(:  # Missing closing parenthesis"
        
        complexity = self.executor._calculate_cyclomatic_complexity(invalid_code)
        
        # Deve indicar erro de sintaxe
        assert 'syntax_error' in complexity
        assert complexity['syntax_error'] == 1.0
    
    def test_calculate_cyclomatic_complexity_simple_function(self) -> None:
        """Testa cálculo de complexidade com função simples."""
        simple_code = """
def simple_function():
    return 42
"""
        
        complexity = self.executor._calculate_cyclomatic_complexity(simple_code)
        
        # Deve ter métricas básicas
        assert 'function_count' in complexity
        assert 'total_complexity' in complexity
        assert complexity['function_count'] >= 1.0
    
    def test_calculate_cyclomatic_complexity_if_statements(self) -> None:
        """Testa cálculo de complexidade com declarações if."""
        if_code = """
def conditional_function(x):
    if x > 0:
        return x
    elif x < 0:
        return -x
    else:
        return 0
"""
        
        complexity = self.executor._calculate_cyclomatic_complexity(if_code)
        
        # Deve ter maior complexidade devido aos ifs
        assert 'decision_points' in complexity
        assert 'total_complexity' in complexity
        assert complexity['total_complexity'] >= 3.0  # 1 base + 2 ifs (if, elif)
    
    def test_calculate_cyclomatic_complexity_loop_statements(self) -> None:
        """Testa cálculo de complexidade com loops."""
        loop_code = """
def loop_function(items):
    result = []
    for item in items:
        if item > 0:
            result.append(item)
    return result
"""
        
        complexity = self.executor._calculate_cyclomatic_complexity(loop_code)
        
        # Deve ter complexidade aumentada por loop e condição
        assert 'decision_points' in complexity
        assert 'total_complexity' in complexity
        assert complexity['total_complexity'] >= 3.0  # 1 base + 1 for + 1 if
    
    def test_calculate_cyclomatic_complexity_nested_structures(self) -> None:
        """Testa cálculo de complexidade com estruturas aninhadas."""
        nested_code = """
def nested_function(data):
    results = []
    for item in data:
        if item['valid']:
            while item['count'] > 0:
                try:
                    processed = item['value'] * 2
                    results.append(processed)
                    item['count'] -= 1
                except Exception as e:
                    print(f"Error: {e}")
                    break
    return results
"""
        
        complexity = self.executor._calculate_cyclomatic_complexity(nested_code)
        
        # Deve ter alta complexidade devido ao aninhamento
        assert 'decision_points' in complexity
        assert 'total_complexity' in complexity
        assert complexity['total_complexity'] >= 4.0  # 1 base + for + if + while
    
    def test_calculate_cyclomatic_complexity_multiple_functions(self) -> None:
        """Testa cálculo de complexidade com múltiplas funções."""
        multi_code = """
def simple_func():
    return 1

def complex_func(x):
    if x > 0:
        for i in range(x):
            if i % 2 == 0:
                print(i)
    return x

def another_func():
    try:
        result = 10 / 0
    except ZeroDivisionError:
        result = 0
    return result
"""
        
        complexity = self.executor._calculate_cyclomatic_complexity(multi_code)
        
        # Deve ter métricas para múltiplas funções
        assert 'function_count' in complexity
        assert 'average_function_complexity' in complexity
        assert 'max_function_complexity' in complexity
        assert complexity['function_count'] >= 3.0
    
    def test_calculate_cyclomatic_complexity_high_complexity_warning(self) -> None:
        """Testa detecção de complexidade alta."""
        # Código com complexidade muito alta
        high_complexity_code = """
def overly_complex_function(data):
    result = []
    for item in data:
        if item.get('type') == 'A':
            for subitem in item.get('subitems', []):
                if subitem.get('active'):
                    while subitem.get('count', 0) > 0:
                        try:
                            if subitem.get('value') > 100:
                                for i in range(10):
                                    if i % 2 == 0:
                                        result.append(subitem.get('value') * i)
                        except Exception:
                            pass
                        finally:
                            subitem['count'] -= 1
                elif subitem.get('pending'):
                    with open('temp.txt', 'w') as f:
                        f.write(str(subitem))
    return result
"""
        
        complexity = self.executor._calculate_cyclomatic_complexity(high_complexity_code)
        
        # Deve detectar alta complexidade
        assert 'total_complexity' in complexity
        assert 'max_function_complexity' in complexity
        # A complexidade deve ser alta para este código
        assert complexity['total_complexity'] >= 5.0
    
    def test_calculate_cyclomatic_complexity_realistic_example(self) -> None:
        """Testa cálculo com exemplo realista de código."""
        realistic_code = """
def process_user_data(users):
    valid_users = []
    invalid_count = 0
    
    for user in users:
        if user.get('active') and user.get('verified'):
            if user.get('age', 0) >= 18:
                if user.get('subscription') != 'expired':
                    valid_users.append(user)
                else:
                    invalid_count += 1
            else:
                invalid_count += 1
        else:
            invalid_count += 1
    
    return {
        'valid_users': valid_users,
        'invalid_count': invalid_count,
        'total_processed': len(users)
    }
"""
        
        complexity = self.executor._calculate_cyclomatic_complexity(realistic_code)
        
        # Deve ter métricas razoáveis para código realista
        assert isinstance(complexity, dict)
        assert 'total_complexity' in complexity
        assert 'function_count' in complexity
        assert 'decision_points' in complexity
        
        # Verificar valores razoáveis
        assert complexity['function_count'] >= 1.0
        assert complexity['decision_points'] >= 3.0  # Vários ifs
        assert complexity['total_complexity'] >= 5.0  # Complexidade moderada