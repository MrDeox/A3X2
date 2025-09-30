"""Testes para o módulo de autoavaliação do SeedAI."""

import ast
import json
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
import tempfile

from a3x.autoeval import AutoEvaluator, EvaluationSeed, RunEvaluation


class TestAutoEvaluator:
    """Testes para a classe AutoEvaluator."""
    
    def setup_method(self) -> None:
        """Configuração antes de cada teste."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.log_dir = self.test_dir / "evaluations"
        self.evaluator = AutoEvaluator(log_dir=self.log_dir)
    
    def teardown_method(self) -> None:
        """Limpeza após cada teste."""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_record_creates_evaluation(self) -> None:
        """Testa a criação de uma avaliação."""
        evaluation = self.evaluator.record(
            goal="Test goal",
            completed=True,
            iterations=5,
            failures=1,
            duration_seconds=10.5
        )
        
        assert evaluation.goal == "Test goal"
        assert evaluation.completed is True
        assert evaluation.iterations == 5
        assert evaluation.failures == 1
        assert evaluation.duration_seconds == 10.5
        assert len(evaluation.seeds) == 0
        # After our enhancement, the system automatically adds quality metrics
        assert "failure_rate" in evaluation.metrics
        assert "success_rate" in evaluation.metrics
        assert evaluation.metrics["failure_rate"] == 0.2  # 1 failure / 5 iterations
        assert evaluation.metrics["success_rate"] == 0.8  # 1 - 0.2
    
    def test_record_with_metrics(self) -> None:
        """Testa a gravação de uma avaliação com métricas."""
        metrics = {
            "apply_patch_count": 2.0,
            "unique_file_extensions": 3.0,
            "actions_success_rate": 0.8
        }
        evaluation = self.evaluator.record(
            goal="Test goal",
            completed=True,
            iterations=5,
            failures=1,
            duration_seconds=10.5,
            metrics=metrics
        )
        
        # After our enhancement, the system automatically adds quality metrics to the existing ones
        expected_metrics = metrics.copy()
        expected_metrics["failure_rate"] = 0.2  # 1 failure / 5 iterations
        expected_metrics["success_rate"] = 0.8  # 1 - 0.2
        expected_metrics["file_diversity"] = 3.0  # from original metrics
        
        for key, value in expected_metrics.items():
            assert evaluation.metrics[key] == value
    
    def test_analyze_code_quality_basic_metrics(self) -> None:
        """Testa a análise básica de qualidade de código."""
        evaluation = RunEvaluation(
            goal="Test goal",
            completed=True,
            iterations=10,
            failures=2,
            duration_seconds=15.0,
            timestamp="2023-01-01T00:00:00Z",
            seeds=[],
            metrics={
                "apply_patch_count": 5.0,
                "unique_file_extensions": 2.0
            },
            capabilities=[]
        )
        
        quality_metrics = self.evaluator._analyze_code_quality(evaluation)
        
        assert "apply_patch_count" in quality_metrics
        assert "file_diversity" in quality_metrics
        assert "failure_rate" in quality_metrics
        assert "success_rate" in quality_metrics
        assert quality_metrics["failure_rate"] == 0.2  # 2 failures / 10 iterations
        assert quality_metrics["success_rate"] == 0.8  # 1 - 0.2
    
    def test_check_code_quality_issues_high_failure_rate(self) -> None:
        """Testa a detecção de problemas de qualidade com alta taxa de falha."""
        quality_metrics = {
            "failure_rate": 0.4,  # 40% failure rate - too high
            "success_rate": 0.6
        }
        
        seeds = self.evaluator._check_code_quality_issues(quality_metrics)
        
        # Should have a seed about high failure rate
        failure_seeds = [s for s in seeds if "falhas" in s.description]
        assert len(failure_seeds) > 0
        assert failure_seeds[0].priority == "high"
        assert failure_seeds[0].capability == "core.execution"
    
    def test_check_code_quality_issues_low_success_patches(self) -> None:
        """Testa a detecção de problemas com patches de baixa qualidade."""
        quality_metrics = {
            "apply_patch_count": 10.0,  # Many patches
            "success_rate": 0.5,       # But low success rate
        }
        
        seeds = self.evaluator._check_code_quality_issues(quality_metrics)
        
        # Should have a seed about patch quality
        patch_seeds = [s for s in seeds if "patch" in s.description.lower()]
        assert len(patch_seeds) > 0
        assert patch_seeds[0].priority == "medium"
        assert patch_seeds[0].capability == "core.diffing"
    
    def test_check_code_quality_issues_low_diversity(self) -> None:
        """Testa a detecção de problemas com baixa diversidade de arquivos."""
        quality_metrics = {
            "file_diversity": 1.0,      # Low diversity
            "apply_patch_count": 15.0,  # Many patches applied
        }
        
        seeds = self.evaluator._check_code_quality_issues(quality_metrics)
        
        # Should have a seed about file diversity
        diversity_seeds = [s for s in seeds if "diversidade" in s.description]
        assert len(diversity_seeds) > 0
        assert diversity_seeds[0].priority == "low"
        assert diversity_seeds[0].capability == "horiz.file_handling"
    
    def test_check_code_quality_issues_no_issues(self) -> None:
        """Testa que não são geradas seeds quando não há problemas de qualidade."""
        quality_metrics = {
            "failure_rate": 0.1,        # Low failure rate
            "success_rate": 0.9,        # High success rate
            "apply_patch_count": 2.0,   # Few patches
            "file_diversity": 5.0,      # Good diversity
        }
        
        seeds = self.evaluator._check_code_quality_issues(quality_metrics)
        
        # Should have no quality-related seeds
        assert len(seeds) == 0


class TestCodeComplexityAnalysis:
    """Testes para a análise de complexidade de código."""
    
    def test_extract_python_code_from_patch(self) -> None:
        """Testa a extração de código Python de um patch."""
        evaluator = AutoEvaluator()
        patch_content = """--- a/test.py
+++ b/test.py
@@ -1,1 +1,3 @@
-old code
+def hello():
+    print("Hello, world!")
+    return True
"""
        
        extracted = evaluator._extract_python_code_from_patch(patch_content)
        
        # Should extract the added lines without the '+' prefix
        assert "def hello():" in extracted
        assert 'print("Hello, world!")' in extracted
        assert "return True" in extracted
    
    def test_analyze_ast_complexity(self) -> None:
        """Testa a análise de complexidade AST."""
        evaluator = AutoEvaluator()
        code = """
def simple_function():
    return 1

def complex_function(x, y):
    if x > y:
        result = x + y
    else:
        result = x - y
    return result

class SimpleClass:
    def method(self):
        pass
"""
        try:
            tree = ast.parse(code)
            complexity = evaluator._analyze_ast_complexity(tree)
            
            # Should detect functions and classes
            assert complexity["ast_function_count"] >= 2
            assert complexity["ast_class_count"] >= 1
            assert complexity["ast_total_nodes"] > 0
            assert complexity["ast_max_depth"] > 0
        except NameError:  # ast not defined in this scope
            import ast
            tree = ast.parse(code)
            complexity = evaluator._analyze_ast_complexity(tree)
            
            # Should detect functions and classes
            assert complexity["ast_function_count"] >= 2
            assert complexity["ast_class_count"] >= 1
            assert complexity["ast_total_nodes"] > 0
            assert complexity["ast_max_depth"] > 0
    
    def test_analyze_code_complexity_from_patch(self) -> None:
        """Testa a análise de complexidade a partir de um patch."""
        evaluator = AutoEvaluator()
        patch_content = """--- a/test.py
+++ b/test.py
@@ -1,1 +1,5 @@
 def old_func():
+    def new_func():
+        x = 1
+        y = 2
+        return x + y
     pass
"""
        
        complexity = evaluator.analyze_code_complexity_from_patch(patch_content)
        
        # Should analyze the added Python code
        assert "ast_function_count" in complexity
        assert "ast_total_nodes" in complexity