"""Análise de dados autônoma para o SeedAI."""

from __future__ import annotations

import ast
import csv
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict

from .actions import AgentAction, ActionType, Observation
from .config import AgentConfig


@dataclass
class DataAnalysisResult:
    """Resultado da análise de dados."""
    
    success: bool
    output: str = ""
    error: Optional[str] = None
    type: str = "data_analysis"
    metrics: Optional[Dict[str, float]] = None
    visualization_path: Optional[str] = None


class DataAnalyzer:
    """Analisador de dados autônomo."""
    
    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.workspace_root = Path(config.workspace.root).resolve()
        self.data_dir = self.workspace_root / "data"
        self.reports_dir = self.workspace_root / "reports" / "data"
        self.visualizations_dir = self.workspace_root / "visualizations"
        
        # Criar diretórios necessários
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.visualizations_dir.mkdir(parents=True, exist_ok=True)
    
    def analyze(self, action: AgentAction) -> DataAnalysisResult:
        """Executa análise de dados com base na ação solicitada."""
        try:
            if action.type == ActionType.ANALYZE_DATA:
                return self._handle_analyze_data(action)
            elif action.type == ActionType.VISUALIZE_DATA:
                return self._handle_visualize_data(action)
            elif action.type == ActionType.CLEAN_DATA:
                return self._handle_clean_data(action)
            elif action.type == ActionType.STATISTICS:
                return self._handle_statistics(action)
            else:
                return DataAnalysisResult(
                    success=False,
                    error=f"Ação de análise de dados não suportada: {action.type}",
                    type="data_analysis"
                )
        except Exception as e:
            return DataAnalysisResult(
                success=False,
                error=str(e),
                type="data_analysis"
            )
    
    def _handle_analyze_data(self, action: AgentAction) -> DataAnalysisResult:
        """Manipula ação de análise de dados completa."""
        if not action.path:
            return DataAnalysisResult(
                success=False,
                error="Caminho do dataset não informado",
                type="data_analysis"
            )
        
        try:
            # Carregar dados
            data_path = self._resolve_path(action.path)
            df = self._load_dataset(data_path)
            
            if df is None:
                return DataAnalysisResult(
                    success=False,
                    error=f"Não foi possível carregar o dataset: {data_path}",
                    type="data_analysis"
                )
            
            # Gerar relatório de análise
            report = self._generate_analysis_report(df, data_path)
            
            # Salvar relatório
            report_path = self.reports_dir / f"{data_path.stem}_analysis_report.md"
            report_path.write_text(report, encoding="utf-8")
            
            # Calcular métricas
            metrics = self._calculate_data_metrics(df)
            
            return DataAnalysisResult(
                success=True,
                output=f"Análise concluída. Relatório salvo em {report_path}",
                type="data_analysis",
                metrics=metrics,
                visualization_path=str(report_path)
            )
            
        except Exception as e:
            return DataAnalysisResult(
                success=False,
                error=f"Erro na análise de dados: {str(e)}",
                type="data_analysis"
            )
    
    def _handle_visualize_data(self, action: AgentAction) -> DataAnalysisResult:
        """Manipula ação de visualização de dados."""
        if not action.path:
            return DataAnalysisResult(
                success=False,
                error="Caminho do dataset não informado",
                type="data_analysis"
            )
        
        try:
            # Carregar dados
            data_path = self._resolve_path(action.path)
            df = self._load_dataset(data_path)
            
            if df is None:
                return DataAnalysisResult(
                    success=False,
                    error=f"Não foi possível carregar o dataset: {data_path}",
                    type="data_analysis"
                )
            
            # Gerar visualizações
            viz_paths = self._generate_visualizations(df, data_path)
            
            if not viz_paths:
                return DataAnalysisResult(
                    success=False,
                    error="Nenhuma visualização pôde ser gerada",
                    type="data_analysis"
                )
            
            return DataAnalysisResult(
                success=True,
                output=f"Visualizações geradas: {', '.join(viz_paths)}",
                type="data_analysis",
                visualization_path=viz_paths[0] if viz_paths else None
            )
            
        except Exception as e:
            return DataAnalysisResult(
                success=False,
                error=f"Erro na visualização de dados: {str(e)}",
                type="data_analysis"
            )
    
    def _handle_clean_data(self, action: AgentAction) -> DataAnalysisResult:
        """Manipula ação de limpeza de dados."""
        if not action.path:
            return DataAnalysisResult(
                success=False,
                error="Caminho do dataset não informado",
                type="data_analysis"
            )
        
        try:
            # Carregar dados
            data_path = self._resolve_path(action.path)
            df = self._load_dataset(data_path)
            
            if df is None:
                return DataAnalysisResult(
                    success=False,
                    error=f"Não foi possível carregar o dataset: {data_path}",
                    type="data_analysis"
                )
            
            # Limpar dados
            cleaned_df = self._clean_dataset(df)
            
            # Salvar dataset limpo
            clean_path = data_path.with_name(f"{data_path.stem}_cleaned{data_path.suffix}")
            self._save_dataset(cleaned_df, clean_path)
            
            # Gerar relatório de limpeza
            report = self._generate_cleaning_report(df, cleaned_df)
            report_path = self.reports_dir / f"{data_path.stem}_cleaning_report.md"
            report_path.write_text(report, encoding="utf-8")
            
            return DataAnalysisResult(
                success=True,
                output=f"Dados limpos e salvos em {clean_path}. Relatório em {report_path}",
                type="data_analysis",
                visualization_path=str(report_path)
            )
            
        except Exception as e:
            return DataAnalysisResult(
                success=False,
                error=f"Erro na limpeza de dados: {str(e)}",
                type="data_analysis"
            )
    
    def _handle_statistics(self, action: AgentAction) -> DataAnalysisResult:
        """Manipula ação de cálculo de estatísticas."""
        if not action.path:
            return DataAnalysisResult(
                success=False,
                error="Caminho do dataset não informado",
                type="data_analysis"
            )
        
        try:
            # Carregar dados
            data_path = self._resolve_path(action.path)
            df = self._load_dataset(data_path)
            
            if df is None:
                return DataAnalysisResult(
                    success=False,
                    error=f"Não foi possível carregar o dataset: {data_path}",
                    type="data_analysis"
                )
            
            # Calcular estatísticas
            stats = self._calculate_statistics(df)
            
            # Salvar estatísticas
            stats_path = self.reports_dir / f"{data_path.stem}_statistics.json"
            stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
            
            # Formatar saída
            output_lines = ["# Estatísticas do Dataset", ""]
            for column, column_stats in stats.items():
                output_lines.append(f"## {column}")
                for stat_name, stat_value in column_stats.items():
                    output_lines.append(f"- {stat_name}: {stat_value}")
                output_lines.append("")
            
            output = "\n".join(output_lines)
            
            return DataAnalysisResult(
                success=True,
                output=output,
                type="data_analysis",
                metrics=self._stats_to_metrics(stats),
                visualization_path=str(stats_path)
            )
            
        except Exception as e:
            return DataAnalysisResult(
                success=False,
                error=f"Erro no cálculo de estatísticas: {str(e)}",
                type="data_analysis"
            )
    
    def _resolve_path(self, path: str) -> Path:
        """Resolve caminho relativo para absoluto."""
        candidate = (
            (self.workspace_root / path).resolve()
            if not Path(path).is_absolute()
            else Path(path).resolve()
        )
        return candidate
    
    def _load_dataset(self, path: Path) -> Optional[pd.DataFrame]:
        """Carrega dataset de vários formatos."""
        try:
            if path.suffix.lower() == '.csv':
                return pd.read_csv(path)
            elif path.suffix.lower() == '.json':
                return pd.read_json(path)
            elif path.suffix.lower() == '.xlsx':
                return pd.read_excel(path)
            elif path.suffix.lower() == '.parquet':
                return pd.read_parquet(path)
            else:
                # Tentar carregar como CSV por padrão
                return pd.read_csv(path)
        except Exception:
            return None
    
    def _save_dataset(self, df: pd.DataFrame, path: Path) -> None:
        """Salva dataset em vários formatos."""
        if path.suffix.lower() == '.csv':
            df.to_csv(path, index=False)
        elif path.suffix.lower() == '.json':
            df.to_json(path, orient='records', indent=2)
        elif path.suffix.lower() == '.xlsx':
            df.to_excel(path, index=False)
        elif path.suffix.lower() == '.parquet':
            df.to_parquet(path, index=False)
        else:
            # Salvar como CSV por padrão
            df.to_csv(path, index=False)
    
    def _clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpa dataset removendo dados faltantes e inconsistências."""
        # Remover linhas com todos os valores NaN
        df_cleaned = df.dropna(how='all')
        
        # Remover colunas com todos os valores NaN
        df_cleaned = df_cleaned.dropna(axis=1, how='all')
        
        # Remover duplicatas
        df_cleaned = df_cleaned.drop_duplicates()
        
        # Converter tipos de dados automaticamente
        for column in df_cleaned.columns:
            if df_cleaned[column].dtype == 'object':
                # Tentar converter para numérico
                numeric_series = pd.to_numeric(df_cleaned[column], errors='ignore')
                if not numeric_series.equals(df_cleaned[column]):
                    df_cleaned[column] = numeric_series
        
        return df_cleaned
    
    def _calculate_statistics(self, df: pd.DataFrame) -> Dict[str, Dict[str, Union[float, int, str]]]:
        """Calcula estatísticas descritivas para cada coluna."""
        stats = {}
        
        for column in df.columns:
            series = df[column]
            column_stats = {}
            
            # Estatísticas básicas
            column_stats['count'] = int(series.count())
            column_stats['missing'] = int(series.isnull().sum())
            column_stats['missing_percentage'] = float((series.isnull().sum() / len(series)) * 100)
            
            # Estatísticas para colunas numéricas
            if pd.api.types.is_numeric_dtype(series):
                column_stats['mean'] = float(series.mean()) if not series.isnull().all() else 0.0
                column_stats['std'] = float(series.std()) if not series.isnull().all() else 0.0
                column_stats['min'] = float(series.min()) if not series.isnull().all() else 0.0
                column_stats['max'] = float(series.max()) if not series.isnull().all() else 0.0
                column_stats['median'] = float(series.median()) if not series.isnull().all() else 0.0
                
                # Quartis
                try:
                    q1 = series.quantile(0.25)
                    q3 = series.quantile(0.75)
                    column_stats['q1'] = float(q1) if not pd.isna(q1) else 0.0
                    column_stats['q3'] = float(q3) if not pd.isna(q3) else 0.0
                except Exception:
                    column_stats['q1'] = 0.0
                    column_stats['q3'] = 0.0
            else:
                # Estatísticas para colunas categóricas
                value_counts = series.value_counts()
                column_stats['unique'] = int(series.nunique())
                if not value_counts.empty:
                    column_stats['most_frequent'] = str(value_counts.index[0])
                    column_stats['most_frequent_count'] = int(value_counts.iloc[0])
        
            stats[str(column)] = column_stats
        
        return stats
    
    def _stats_to_metrics(self, stats: Dict[str, Dict[str, Union[float, int, str]]]) -> Dict[str, float]:
        """Converte estatísticas em métricas para autoavaliação."""
        metrics = {}
        
        # Métricas gerais
        total_columns = len(stats)
        metrics['data_columns'] = float(total_columns)
        
        # Métricas de qualidade
        total_missing = sum(stat.get('missing', 0) for stat in stats.values())
        metrics['data_missing_values'] = float(total_missing)
        
        missing_percentage = sum(stat.get('missing_percentage', 0) for stat in stats.values()) / total_columns if total_columns > 0 else 0
        metrics['data_missing_percentage'] = float(missing_percentage)
        
        # Métricas de variedade
        numeric_columns = sum(1 for stat in stats.values() if 'mean' in stat)
        categorical_columns = total_columns - numeric_columns
        metrics['data_numeric_columns'] = float(numeric_columns)
        metrics['data_categorical_columns'] = float(categorical_columns)
        
        return metrics
    
    def _generate_analysis_report(self, df: pd.DataFrame, path: Path) -> str:
        """Gera relatório de análise do dataset."""
        lines = [
            f"# Relatório de Análise de Dados - {path.name}",
            "",
            "## Resumo Geral",
            f"- Número de linhas: {len(df)}",
            f"- Número de colunas: {len(df.columns)}",
            f"- Tipos de dados: {df.dtypes.nunique()}",
            "",
            "## Colunas",
        ]
        
        # Informações sobre colunas
        for column in df.columns:
            lines.append(f"- **{column}** ({df[column].dtype})")
            lines.append(f"  - Valores não nulos: {df[column].count()}")
            lines.append(f"  - Valores nulos: {df[column].isnull().sum()}")
            if df[column].dtype in ['int64', 'float64']:
                lines.append(f"  - Média: {df[column].mean():.2f}")
                lines.append(f"  - Desvio padrão: {df[column].std():.2f}")
            lines.append("")
        
        # Amostra dos dados
        lines.append("## Amostra dos Dados")
        lines.append("")
        sample_df = df.head(10)
        lines.append(sample_df.to_string(index=False))
        lines.append("")
        
        return "\n".join(lines)
    
    def _generate_cleaning_report(self, original_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> str:
        """Gera relatório de limpeza de dados."""
        lines = [
            "# Relatório de Limpeza de Dados",
            "",
            "## Alterações Realizadas",
            f"- Linhas originais: {len(original_df)}",
            f"- Linhas após limpeza: {len(cleaned_df)}",
            f"- Linhas removidas: {len(original_df) - len(cleaned_df)}",
            f"- Duplicatas removidas: {len(original_df) - len(original_df.drop_duplicates())}",
            "",
        ]
        
        # Comparar colunas
        original_cols = set(original_df.columns)
        cleaned_cols = set(cleaned_df.columns)
        removed_cols = original_cols - cleaned_cols
        
        if removed_cols:
            lines.append("## Colunas Removidas")
            for col in removed_cols:
                lines.append(f"- {col}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _generate_visualizations(self, df: pd.DataFrame, path: Path) -> List[str]:
        """Gera visualizações dos dados."""
        viz_paths = []
        
        try:
            # Histograma para colunas numéricas
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for column in numeric_columns[:3]:  # Limitar a 3 colunas
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(df[column].dropna(), bins=30, alpha=0.7)
                ax.set_title(f'Distribuição de {column}')
                ax.set_xlabel(column)
                ax.set_ylabel('Frequência')
                
                viz_path = self.visualizations_dir / f"{path.stem}_{column}_histogram.png"
                fig.savefig(viz_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                viz_paths.append(str(viz_path))
            
            # Gráfico de barras para colunas categóricas
            categorical_columns = df.select_dtypes(include=['object']).columns
            for column in categorical_columns[:2]:  # Limitar a 2 colunas
                value_counts = df[column].value_counts().head(10)  # Top 10 categorias
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.bar(range(len(value_counts)), value_counts.values)
                ax.set_title(f'Distribuição de {column}')
                ax.set_xlabel(column)
                ax.set_ylabel('Contagem')
                ax.set_xticks(range(len(value_counts)))
                ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
                
                viz_path = self.visualizations_dir / f"{path.stem}_{column}_bar_chart.png"
                fig.savefig(viz_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                viz_paths.append(str(viz_path))
                
        except Exception as e:
            # Silently continue if visualization fails
            pass
        
        return viz_paths
    
    def _calculate_data_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calcula métricas de qualidade dos dados."""
        metrics = {}
        
        # Dimensões do dataset
        metrics['data_rows'] = float(len(df))
        metrics['data_columns'] = float(len(df.columns))
        
        # Qualidade dos dados
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isnull().sum().sum()
        metrics['data_completeness'] = float(1.0 - (missing_cells / total_cells)) if total_cells > 0 else 0.0
        
        # Variedade de tipos de dados
        metrics['data_type_diversity'] = float(df.dtypes.nunique())
        
        # Distribuição de dados
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        metrics['data_numeric_columns'] = float(len(numeric_columns))
        
        categorical_columns = df.select_dtypes(include=['object']).columns
        metrics['data_categorical_columns'] = float(len(categorical_columns))
        
        # Cardinalidade média para colunas categóricas
        if len(categorical_columns) > 0:
            avg_cardinality = np.mean([df[col].nunique() for col in categorical_columns])
            metrics['data_avg_categorical_cardinality'] = float(avg_cardinality)
        
        return metrics


# Ações de análise de dados disponíveis
DATA_ANALYSIS_ACTIONS = [
    ActionType.ANALYZE_DATA,
    ActionType.VISUALIZE_DATA,
    ActionType.CLEAN_DATA,
    ActionType.STATISTICS,
]


__all__ = [
    "DataAnalyzer",
    "DataAnalysisResult",
    "DATA_ANALYSIS_ACTIONS",
]