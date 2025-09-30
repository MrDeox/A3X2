"""Testes para o módulo de análise de dados do SeedAI."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np

from a3x.data_analysis import DataAnalyzer, DataAnalysisResult
from a3x.actions import AgentAction, ActionType
from a3x.config import AgentConfig, WorkspaceConfig


class TestDataAnalyzer:
    """Testes para o DataAnalyzer."""
    
    def setup_method(self) -> None:
        """Configuração antes de cada teste."""
        # Criar diretório temporário para testes
        self.temp_dir = Path(tempfile.mkdtemp())
        self.workspace_root = self.temp_dir / "workspace"
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        
        # Criar configuração de teste
        self.mock_config = Mock(spec=AgentConfig)
        self.mock_config.workspace = Mock(spec=WorkspaceConfig)
        self.mock_config.workspace.root = str(self.workspace_root)
        self.mock_config.policies = Mock()
        self.mock_config.policies.allow_network = False
        self.mock_config.policies.deny_commands = []
        self.mock_config.audit = Mock()
        self.mock_config.audit.enable_file_log = True
        self.mock_config.audit.file_dir = Path("seed/changes")
        self.mock_config.audit.enable_git_commit = False
        self.mock_config.audit.commit_prefix = "A3X"
        
        # Criar analyzer
        self.analyzer = DataAnalyzer(self.mock_config)
    
    def teardown_method(self) -> None:
        """Limpeza após cada teste."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init_creates_directories(self) -> None:
        """Testa que o inicializador cria os diretórios necessários."""
        # Verificar que os diretórios foram criados
        assert self.analyzer.data_dir.exists()
        assert self.analyzer.reports_dir.exists()
        assert self.analyzer.visualizations_dir.exists()
        
        # Verificar que são subdiretórios do workspace
        assert str(self.workspace_root) in str(self.analyzer.data_dir)
        assert str(self.workspace_root) in str(self.analyzer.reports_dir)
        assert str(self.workspace_root) in str(self.analyzer.visualizations_dir)
    
    def test_analyze_with_unsupported_action(self) -> None:
        """Testa análise com ação não suportada."""
        action = AgentAction(type=ActionType.MESSAGE, text="Unsupported action")
        
        result = self.analyzer.analyze(action)
        
        assert isinstance(result, DataAnalysisResult)
        assert result.success is False
        assert "não suportada" in result.error
    
    def test_analyze_data_with_missing_path(self) -> None:
        """Testa análise de dados sem caminho especificado."""
        action = AgentAction(type=ActionType.ANALYZE_DATA)
        
        result = self.analyzer.analyze(action)
        
        assert isinstance(result, DataAnalysisResult)
        assert result.success is False
        assert "não informado" in result.error
    
    def test_analyze_data_with_nonexistent_file(self) -> None:
        """Testa análise de dados com arquivo inexistente."""
        action = AgentAction(type=ActionType.ANALYZE_DATA, path="nonexistent.csv")
        
        result = self.analyzer.analyze(action)
        
        assert isinstance(result, DataAnalysisResult)
        assert result.success is False
        assert "não foi possível carregar" in result.error.lower()
    
    def test_analyze_data_with_valid_csv(self) -> None:
        """Testa análise de dados com CSV válido."""
        # Criar dataset de teste
        test_data = pd.DataFrame({
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [25, 30, 35],
            'salary': [50000, 60000, 70000]
        })
        
        # Salvar como CSV
        csv_path = self.workspace_root / "test_data.csv"
        test_data.to_csv(csv_path, index=False)
        
        # Criar ação
        action = AgentAction(type=ActionType.ANALYZE_DATA, path=str(csv_path))
        
        # Mock para evitar salvar o relatório durante o teste
        with patch.object(self.analyzer, '_generate_analysis_report', return_value="# Test Report"):
            result = self.analyzer.analyze(action)
        
        assert isinstance(result, DataAnalysisResult)
        assert result.success is True
        assert "Análise concluída" in result.output
        assert result.metrics is not None
        assert "data_columns" in result.metrics
    
    def test_visualize_data_with_missing_path(self) -> None:
        """Testa visualização de dados sem caminho especificado."""
        action = AgentAction(type=ActionType.VISUALIZE_DATA)
        
        result = self.analyzer.analyze(action)
        
        assert isinstance(result, DataAnalysisResult)
        assert result.success is False
        assert "não informado" in result.error
    
    def test_visualize_data_with_valid_csv(self) -> None:
        """Testa visualização de dados com CSV válido."""
        # Criar dataset de teste
        test_data = pd.DataFrame({
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [25, 30, 35],
            'salary': [50000, 60000, 70000]
        })
        
        # Salvar como CSV
        csv_path = self.workspace_root / "test_data.csv"
        test_data.to_csv(csv_path, index=False)
        
        # Criar ação
        action = AgentAction(type=ActionType.VISUALIZE_DATA, path=str(csv_path))
        
        # Mock para evitar salvar visualizações durante o teste
        with patch.object(self.analyzer, '_generate_visualizations', return_value=["/fake/path/viz.png"]):
            result = self.analyzer.analyze(action)
        
        assert isinstance(result, DataAnalysisResult)
        assert result.success is True
        assert "Visualizações geradas" in result.output
        assert result.visualization_path is not None
    
    def test_clean_data_with_missing_path(self) -> None:
        """Testa limpeza de dados sem caminho especificado."""
        action = AgentAction(type=ActionType.CLEAN_DATA)
        
        result = self.analyzer.analyze(action)
        
        assert isinstance(result, DataAnalysisResult)
        assert result.success is False
        assert "não informado" in result.error
    
    def test_clean_data_with_valid_csv(self) -> None:
        """Testa limpeza de dados com CSV válido."""
        # Criar dataset de teste com dados faltantes
        test_data = pd.DataFrame({
            'name': ['Alice', 'Bob', None, 'Charlie', 'Alice'],
            'age': [25, 30, np.nan, 35, 25],
            'salary': [50000, 60000, 70000, np.nan, 50000]
        })
        
        # Salvar como CSV
        csv_path = self.workspace_root / "dirty_data.csv"
        test_data.to_csv(csv_path, index=False)
        
        # Criar ação
        action = AgentAction(type=ActionType.CLEAN_DATA, path=str(csv_path))
        
        # Mock para evitar salvar durante o teste
        with patch.object(self.analyzer, '_save_dataset') as mock_save:
            result = self.analyzer.analyze(action)
        
        assert isinstance(result, DataAnalysisResult)
        assert result.success is True
        assert "Dados limpos" in result.output
        assert result.visualization_path is not None
    
    def test_statistics_with_missing_path(self) -> None:
        """Testa cálculo de estatísticas sem caminho especificado."""
        action = AgentAction(type=ActionType.STATISTICS)
        
        result = self.analyzer.analyze(action)
        
        assert isinstance(result, DataAnalysisResult)
        assert result.success is False
        assert "não informado" in result.error
    
    def test_statistics_with_valid_csv(self) -> None:
        """Testa cálculo de estatísticas com CSV válido."""
        # Criar dataset de teste
        test_data = pd.DataFrame({
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [25, 30, 35],
            'salary': [50000, 60000, 70000]
        })
        
        # Salvar como CSV
        csv_path = self.workspace_root / "test_data.csv"
        test_data.to_csv(csv_path, index=False)
        
        # Criar ação
        action = AgentAction(type=ActionType.STATISTICS, path=str(csv_path))
        
        result = self.analyzer.analyze(action)
        
        assert isinstance(result, DataAnalysisResult)
        assert result.success is True
        assert "Estatísticas do Dataset" in result.output
        assert result.metrics is not None
        assert "data_columns" in result.metrics
    
    def test_resolve_path_relative(self) -> None:
        """Testa resolução de caminho relativo."""
        relative_path = "test.csv"
        resolved_path = self.analyzer._resolve_path(relative_path)
        
        # Deve resolver para um caminho absoluto dentro do workspace
        assert resolved_path.is_absolute()
        assert str(self.workspace_root) in str(resolved_path)
    
    def test_resolve_path_absolute(self) -> None:
        """Testa resolução de caminho absoluto."""
        absolute_path = "/tmp/test.csv"
        resolved_path = self.analyzer._resolve_path(absolute_path)
        
        # Deve resolver para o caminho absoluto
        assert resolved_path.is_absolute()
        assert str(resolved_path) == absolute_path
    
    def test_load_dataset_csv(self) -> None:
        """Testa carregamento de dataset CSV."""
        # Criar dataset de teste
        test_data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        
        # Salvar como CSV
        csv_path = self.workspace_root / "test.csv"
        test_data.to_csv(csv_path, index=False)
        
        # Carregar dataset
        df = self.analyzer._load_dataset(csv_path)
        
        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert list(df.columns) == ['col1', 'col2']
    
    def test_load_dataset_json(self) -> None:
        """Testa carregamento de dataset JSON."""
        # Criar dataset de teste
        test_data = {
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        }
        
        # Salvar como JSON
        json_path = self.workspace_root / "test.json"
        with open(json_path, 'w') as f:
            json.dump(test_data, f)
        
        # Carregar dataset
        df = self.analyzer._load_dataset(json_path)
        
        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert list(df.columns) == ['col1', 'col2']
    
    def test_load_dataset_invalid(self) -> None:
        """Testa carregamento de dataset inválido."""
        # Criar arquivo inválido
        invalid_path = self.workspace_root / "invalid.txt"
        with open(invalid_path, 'w') as f:
            f.write("invalid content that is not a valid dataset")
        
        # Tentar carregar dataset
        df = self.analyzer._load_dataset(invalid_path)
        
        # Pode retornar None ou DataFrame vazio dependendo da implementação
        # O importante é que não lance exceção
        assert df is None or isinstance(df, pd.DataFrame)
    
    def test_clean_dataset_basic(self) -> None:
        """Testa limpeza básica de dataset."""
        # Criar dataset com dados faltantes e duplicatas
        dirty_data = pd.DataFrame({
            'name': ['Alice', 'Bob', None, 'Charlie', 'Alice'],
            'age': [25, 30, np.nan, 35, 25],
            'salary': [50000, 60000, 70000, np.nan, 50000]
        })
        
        # Limpar dataset
        clean_data = self.analyzer._clean_dataset(dirty_data)
        
        assert isinstance(clean_data, pd.DataFrame)
        # Deve remover duplicatas e linhas com todos os valores NaN
        assert len(clean_data) <= len(dirty_data)
    
    def test_calculate_statistics_basic(self) -> None:
        """Testa cálculo básico de estatísticas."""
        # Criar dataset de teste
        test_data = pd.DataFrame({
            'numeric_col': [1, 2, 3, 4, 5],
            'categorical_col': ['a', 'b', 'c', 'a', 'b']
        })
        
        # Calcular estatísticas
        stats = self.analyzer._calculate_statistics(test_data)
        
        assert isinstance(stats, dict)
        assert 'numeric_col' in stats
        assert 'categorical_col' in stats
        
        # Verificar estatísticas numéricas
        numeric_stats = stats['numeric_col']
        assert 'mean' in numeric_stats
        assert 'std' in numeric_stats
        assert 'min' in numeric_stats
        assert 'max' in numeric_stats
        assert 'count' in numeric_stats
        
        # Verificar estatísticas categóricas
        categorical_stats = stats['categorical_col']
        assert 'unique' in categorical_stats
        assert 'most_frequent' in categorical_stats
        assert 'most_frequent_count' in categorical_stats
    
    def test_stats_to_metrics_conversion(self) -> None:
        """Testa conversão de estatísticas para métricas."""
        # Estatísticas de teste
        test_stats = {
            'col1': {
                'count': 100,
                'missing': 5,
                'missing_percentage': 5.0,
                'mean': 25.5,
                'std': 10.2,
                'unique': 50,
                'most_frequent': 'value1',
                'most_frequent_count': 20
            },
            'col2': {
                'count': 100,
                'missing': 10,
                'missing_percentage': 10.0,
                'mean': 30.0,
                'std': 15.5,
                'unique': 75,
                'most_frequent': 'value2',
                'most_frequent_count': 30
            }
        }
        
        # Converter para métricas
        metrics = self.analyzer._stats_to_metrics(test_stats)
        
        assert isinstance(metrics, dict)
        assert 'data_columns' in metrics
        assert 'data_missing_values' in metrics
        assert 'data_missing_percentage' in metrics
        assert 'data_numeric_columns' in metrics
        assert 'data_categorical_columns' in metrics
        assert metrics['data_columns'] == 2.0  # Duas colunas
        assert metrics['data_missing_values'] == 15.0  # 5 + 10
        assert metrics['data_missing_percentage'] == 7.5  # Média de 5.0 e 10.0
    
    def test_generate_analysis_report(self) -> None:
        """Testa geração de relatório de análise."""
        # Criar dataset de teste
        test_data = pd.DataFrame({
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [25, 30, 35],
            'salary': [50000, 60000, 70000]
        })
        
        # Gerar relatório
        report = self.analyzer._generate_analysis_report(test_data, Path("test.csv"))
        
        assert isinstance(report, str)
        assert len(report) > 0
        assert "# Relatório de Análise de Dados" in report
        assert "Resumo Geral" in report
        assert "Colunas" in report
        assert "Amostra dos Dados" in report
    
    def test_generate_cleaning_report(self) -> None:
        """Testa geração de relatório de limpeza."""
        # Criar datasets de teste
        original_data = pd.DataFrame({
            'name': ['Alice', 'Bob', 'Charlie', 'Alice'],
            'age': [25, 30, 35, 25],
            'salary': [50000, 60000, 70000, 50000]
        })
        
        cleaned_data = pd.DataFrame({
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [25, 30, 35],
            'salary': [50000, 60000, 70000]
        })
        
        # Gerar relatório
        report = self.analyzer._generate_cleaning_report(original_data, cleaned_data)
        
        assert isinstance(report, str)
        assert len(report) > 0
        assert "# Relatório de Limpeza de Dados" in report
        assert "Alterações Realizadas" in report
    
    def test_generate_visualizations(self) -> None:
        """Testa geração de visualizações."""
        # Criar dataset de teste
        test_data = pd.DataFrame({
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [25, 30, 35],
            'salary': [50000, 60000, 70000]
        })
        
        # Gerar visualizações
        viz_paths = self.analyzer._generate_visualizations(test_data, Path("test.csv"))
        
        # Deve retornar lista de caminhos (pode estar vazia se matplotlib não estiver disponível)
        assert isinstance(viz_paths, list)
    
    def test_calculate_data_metrics(self) -> None:
        """Testa cálculo de métricas de dados."""
        # Criar dataset de teste
        test_data = pd.DataFrame({
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [25, 30, 35],
            'salary': [50000, 60000, 70000]
        })
        
        # Calcular métricas
        metrics = self.analyzer._calculate_data_metrics(test_data)
        
        assert isinstance(metrics, dict)
        assert 'data_rows' in metrics
        assert 'data_columns' in metrics
        assert 'data_completeness' in metrics
        assert 'data_type_diversity' in metrics
        assert 'data_numeric_columns' in metrics
        assert 'data_categorical_columns' in metrics
        
        # Verificar valores
        assert metrics['data_rows'] == 3.0
        assert metrics['data_columns'] == 3.0
        assert metrics['data_completeness'] == 1.0  # Sem valores faltantes
        assert metrics['data_numeric_columns'] == 2.0  # age e salary
        assert metrics['data_categorical_columns'] == 1.0  # name