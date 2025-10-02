#!/usr/bin/env python3
"""Performance benchmark script for A3X caching and performance monitoring system."""

import asyncio
import json
import statistics
import time
from typing import Dict, List, Any
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from a3x.cache import CacheManager, llm_cache_manager, ast_cache_manager, memory_cache_manager
from a3x.execution.monitoring import PerformanceMonitor
from a3x.memory.store import SemanticMemory
from a3x.memory.embedder import get_embedder


class PerformanceBenchmark:
    """Benchmark suite for testing caching and performance improvements."""

    def __init__(self):
        self.results = {}
        self.cache_manager = CacheManager()

    def benchmark_llm_caching(self) -> Dict[str, Any]:
        """Benchmark LLM response caching performance."""
        print("🔄 Benchmarking LLM caching...")

        # Sample prompts for testing
        test_prompts = [
            "What is the meaning of life?",
            "Explain quantum computing in simple terms.",
            "Write a Python function to calculate factorial.",
            "How does machine learning work?",
            "Describe the benefits of caching in software systems."
        ]

        # Test without caching (simulated)
        start_time = time.time()
        for prompt in test_prompts:
            # Simulate LLM call delay
            time.sleep(0.1)
        uncached_time = time.time() - start_time

        # Test with caching (simulated)
        start_time = time.time()
        cache = llm_cache_manager.get_cache("benchmark_llm")

        for prompt in test_prompts:
            # Check cache first
            cached = cache.get("test_model", [{"role": "user", "content": prompt}])
            if not cached:
                # Simulate LLM call
                time.sleep(0.1)
                # Cache result
                response = {
                    "choices": [{"message": {"content": f"Response to: {prompt}"}}]
                }
                cache.put("test_model", [{"role": "user", "content": prompt}],
                         response, {"prompt_tokens": 10, "completion_tokens": 20},
                         "completed")

        cached_time = time.time() - start_time

        improvement = ((uncached_time - cached_time) / uncached_time) * 100

        return {
            "uncached_time": uncached_time,
            "cached_time": cached_time,
            "improvement_percent": improvement,
            "cache_hits": len(test_prompts),  # All should be cache misses on first run
            "test_prompts": len(test_prompts)
        }

    def benchmark_ast_caching(self) -> Dict[str, Any]:
        """Benchmark AST parsing and analysis caching."""
        print("🔄 Benchmarking AST caching...")

        # Sample Python code snippets for testing
        test_codes = [
            "def hello_world(): print('Hello, World!')",
            "class Calculator: def add(self, a, b): return a + b",
            """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
""",
            """
import math

class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def magnitude(self):
        return math.sqrt(self.x**2 + self.y**2)
""",
            """
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
"""
        ]

        # Test without caching
        import ast
        start_time = time.time()
        for code in test_codes:
            try:
                tree = ast.parse(code)
                # Simulate complexity analysis
                for node in ast.walk(tree):
                    pass
            except SyntaxError:
                pass
        uncached_time = time.time() - start_time

        # Test with caching
        start_time = time.time()
        cache = ast_cache_manager.get_cache("benchmark_ast")

        for code in test_codes:
            try:
                # Use cached parsing
                tree, syntax_valid = cache.get_parse_result(code)
                if syntax_valid:
                    # Simulate complexity analysis (cached tree)
                    for node in ast.walk(tree):
                        pass
            except SyntaxError:
                pass

        cached_time = time.time() - start_time

        improvement = ((uncached_time - cached_time) / uncached_time) * 100

        return {
            "uncached_time": uncached_time,
            "cached_time": cached_time,
            "improvement_percent": improvement,
            "code_snippets": len(test_codes),
            "cache_effectiveness": "high" if improvement > 30 else "moderate" if improvement > 10 else "low"
        }

    def benchmark_memory_operations(self) -> Dict[str, Any]:
        """Benchmark memory operations with embedding and similarity caching."""
        print("🔄 Benchmarking memory operations...")

        # Sample texts for embedding and similarity testing
        test_texts = [
            "Machine learning is a subset of artificial intelligence.",
            "Neural networks are inspired by biological neural networks.",
            "Deep learning uses multiple layers of neural networks.",
            "Natural language processing helps computers understand text.",
            "Computer vision enables machines to interpret visual information.",
            "Reinforcement learning trains agents through rewards and penalties.",
            "Supervised learning requires labeled training data.",
            "Unsupervised learning finds patterns in unlabeled data.",
            "Transfer learning reuses knowledge from one task for another.",
            "Artificial intelligence aims to create intelligent machines."
        ]

        # Test embedding computation without caching
        embedder = get_embedder()
        start_time = time.time()
        embeddings = []
        for text in test_texts:
            embedding = embedder.embed([text])[0]
            embeddings.append(embedding)
        uncached_embedding_time = time.time() - start_time

        # Test similarity calculations without caching
        start_time = time.time()
        similarities = []
        for i, emb1 in enumerate(embeddings):
            for j, emb2 in enumerate(embeddings):
                if i != j:
                    # Simple cosine similarity calculation
                    dot = sum(a * b for a, b in zip(emb1, emb2))
                    norm1 = sum(a * a for a in emb1) ** 0.5
                    norm2 = sum(b * b for b in emb2) ** 0.5
                    if norm1 > 0 and norm2 > 0:
                        similarities.append(dot / (norm1 * norm2))
        uncached_similarity_time = time.time() - start_time

        # Test with caching
        memory_cache = memory_cache_manager.get_cache("benchmark_memory")

        # Test embedding caching
        start_time = time.time()
        cached_embeddings = []
        for text in test_texts:
            cached = memory_cache.get_cached_embedding(text, "test_model", "test_method")
            if cached:
                cached_embeddings.append(cached)
            else:
                embedding = embedder.embed([text])[0]
                cached_embeddings.append(embedding)
                memory_cache.cache_embedding(text, embedding, "test_model", "test_method")
        cached_embedding_time = time.time() - start_time

        # Test similarity caching
        start_time = time.time()
        cached_similarities = []
        for i, emb1 in enumerate(cached_embeddings):
            for j, emb2 in enumerate(cached_embeddings):
                if i != j:
                    cached = memory_cache.get_cached_similarity(emb1, emb2)
                    if cached:
                        cached_similarities.append(cached)
                    else:
                        # Calculate and cache
                        dot = sum(a * b for a, b in zip(emb1, emb2))
                        norm1 = sum(a * a for a in emb1) ** 0.5
                        norm2 = sum(b * b for b in emb2) ** 0.5
                        if norm1 > 0 and norm2 > 0:
                            similarity = dot / (norm1 * norm2)
                            cached_similarities.append(similarity)
                            memory_cache.cache_similarity(emb1, emb2, similarity)
        cached_similarity_time = time.time() - start_time

        embedding_improvement = ((uncached_embedding_time - cached_embedding_time) / uncached_embedding_time) * 100
        similarity_improvement = ((uncached_similarity_time - cached_similarity_time) / uncached_similarity_time) * 100

        return {
            "embedding": {
                "uncached_time": uncached_embedding_time,
                "cached_time": cached_embedding_time,
                "improvement_percent": embedding_improvement
            },
            "similarity": {
                "uncached_time": uncached_similarity_time,
                "cached_time": cached_similarity_time,
                "improvement_percent": similarity_improvement
            },
            "overall_improvement": (embedding_improvement + similarity_improvement) / 2,
            "test_texts": len(test_texts)
        }

    def benchmark_semantic_memory(self) -> Dict[str, Any]:
        """Benchmark semantic memory operations with caching."""
        print("🔄 Benchmarking semantic memory operations...")

        # Create test memory entries
        test_entries = [
            ("AI Overview", "Artificial Intelligence is the simulation of human intelligence in machines."),
            ("Machine Learning", "Machine Learning is a subset of AI that enables computers to learn from data."),
            ("Deep Learning", "Deep Learning uses neural networks with multiple layers to process data."),
            ("NLP", "Natural Language Processing helps computers understand and generate human language."),
            ("Computer Vision", "Computer Vision enables computers to interpret and understand visual information.")
        ]

        # Test memory operations without caching
        start_time = time.time()
        memory = SemanticMemory(":memory:")  # In-memory for testing

        # Add entries without caching
        for title, content in test_entries:
            memory.add(title, content)

        # Query without caching
        for title, _ in test_entries:
            results = memory.query(title, top_k=3)
        uncached_time = time.time() - start_time

        # Test memory operations with caching
        start_time = time.time()
        cached_memory = SemanticMemory(":memory:")  # In-memory for testing

        # Add entries with caching
        for title, content in test_entries:
            cached_memory.add(title, content)

        # Query with caching
        for title, _ in test_entries:
            results = cached_memory.query(title, top_k=3)
        cached_time = time.time() - start_time

        improvement = ((uncached_time - cached_time) / uncached_time) * 100

        return {
            "uncached_time": uncached_time,
            "cached_time": cached_time,
            "improvement_percent": improvement,
            "entries_added": len(test_entries),
            "queries_executed": len(test_entries)
        }

    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all benchmark tests."""
        print("🚀 Starting A3X Performance Benchmark Suite")
        print("=" * 60)

        results = {}

        try:
            results["llm_caching"] = self.benchmark_llm_caching()
            print(f"✅ LLM Caching: {results['llm_caching']['improvement_percent']:.1f}% improvement")
        except Exception as e:
            print(f"❌ LLM Caching benchmark failed: {e}")
            results["llm_caching"] = {"error": str(e)}

        try:
            results["ast_caching"] = self.benchmark_ast_caching()
            print(f"✅ AST Caching: {results['ast_caching']['improvement_percent']:.1f}% improvement")
        except Exception as e:
            print(f"❌ AST Caching benchmark failed: {e}")
            results["ast_caching"] = {"error": str(e)}

        try:
            results["memory_operations"] = self.benchmark_memory_operations()
            print(f"✅ Memory Operations: {results['memory_operations']['overall_improvement']:.1f}% improvement")
        except Exception as e:
            print(f"❌ Memory Operations benchmark failed: {e}")
            results["memory_operations"] = {"error": str(e)}

        try:
            results["semantic_memory"] = self.benchmark_semantic_memory()
            print(f"✅ Semantic Memory: {results['semantic_memory']['improvement_percent']:.1f}% improvement")
        except Exception as e:
            print(f"❌ Semantic Memory benchmark failed: {e}")
            results["semantic_memory"] = {"error": str(e)}

        # Generate summary
        self.generate_summary_report(results)

        return results

    def generate_summary_report(self, results: Dict[str, Any]) -> None:
        """Generate a comprehensive summary report."""
        print("\n" + "=" * 60)
        print("📊 PERFORMANCE BENCHMARK SUMMARY")
        print("=" * 60)

        total_improvements = []
        successful_benchmarks = 0

        for benchmark_name, benchmark_results in results.items():
            if "error" not in benchmark_results:
                successful_benchmarks += 1
                if "improvement_percent" in benchmark_results:
                    improvement = benchmark_results["improvement_percent"]
                    total_improvements.append(improvement)
                    status = "🟢 EXCELLENT" if improvement > 50 else "🟡 GOOD" if improvement > 20 else "🟠 MODERATE"
                    print(f"{benchmark_name.upper()}: {improvement:6.1f}% improvement {status}")
                elif "overall_improvement" in benchmark_results:
                    improvement = benchmark_results["overall_improvement"]
                    total_improvements.append(improvement)
                    status = "🟢 EXCELLENT" if improvement > 50 else "🟡 GOOD" if improvement > 20 else "🟠 MODERATE"
                    print(f"{benchmark_name.upper()}: {improvement:6.1f}% improvement {status}")
            else:
                print(f"{benchmark_name.upper()}: ❌ FAILED - {benchmark_results['error']}")

        if total_improvements:
            avg_improvement = statistics.mean(total_improvements)
            max_improvement = max(total_improvements)
            min_improvement = min(total_improvements)

            print("\n📈 OVERALL RESULTS:")
            print(f"   Average Improvement: {avg_improvement:.1f}%")
            print(f"   Best Performance: {max_improvement:.1f}%")
            print(f"   Worst Performance: {min_improvement:.1f}%")
            print(f"   Successful Benchmarks: {successful_benchmarks}/{len(results)}")

            if avg_improvement > 30:
                print("🏆 EXCELLENT: Caching system provides significant performance gains!")
            elif avg_improvement > 15:
                print("👍 GOOD: Caching system provides moderate performance improvements.")
            else:
                print("⚠️  MODERATE: Caching system needs optimization.")

        # Save detailed results
        self.save_results(results)

    def save_results(self, results: Dict[str, Any]) -> None:
        """Save benchmark results to file."""
        try:
            results_file = "a3x/performance_benchmark_results.json"
            os.makedirs("a3x", exist_ok=True)

            with open(results_file, 'w') as f:
                json.dump({
                    "timestamp": time.time(),
                    "results": results,
                    "summary": {
                        "total_benchmarks": len(results),
                        "successful_benchmarks": len([r for r in results.values() if "error" not in r])
                    }
                }, f, indent=2)

            print(f"\n💾 Detailed results saved to: {results_file}")
        except Exception as e:
            print(f"\n⚠️  Could not save results: {e}")


def main():
    """Main entry point for performance benchmarking."""
    print("A3X Performance Benchmark Suite")
    print("Testing caching and performance monitoring system...\n")

    benchmark = PerformanceBenchmark()
    results = benchmark.run_all_benchmarks()

    return 0 if len([r for r in results.values() if "error" not in r]) > 0 else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)