"""
RAG Accuracy Evaluation Script - E5-Base-V2

Evaluates the accuracy of RAG retrieval using e5-base-v2 embeddings with:
1. L2 Distance with semantic similarity matching
2. Cosine Similarity with semantic similarity matching

Tests on the test dataset and generates comparison bar graphs.
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

# File paths
TEST_FILE = "../test.jsonl"
L2_INDEX_FILE = "bash_commands_l2_e5.bin"
L2_META_FILE = "metadata_l2_e5.npz"
COSINE_INDEX_FILE = "bash_commands_cosine_e5.bin"
COSINE_META_FILE = "metadata_cosine_e5.npz"

# Cosine similarity threshold for matching retrieved commands with expected commands
COSINE_SIMILARITY_THRESHOLD = 0.85

class RAGAccuracyEvaluator:
    def __init__(self):
        console.print("[cyan]Loading e5-base-v2 model...[/cyan]")
        self.embedder = SentenceTransformer("intfloat/e5-base-v2")
        console.print("[green]✓ Model loaded[/green]")
        self.l2_index = None
        self.cosine_index = None
        self.l2_data = None
        self.cosine_data = None
        
    def load_indexes(self):
        """Load both L2 and Cosine similarity indexes"""
        console.print("[cyan]Loading indexes...[/cyan]")
        
        # Load L2 index
        try:
            self.l2_index = faiss.read_index(L2_INDEX_FILE)
            meta_data = np.load(L2_META_FILE, allow_pickle=True)
            self.l2_data = meta_data['data'].tolist()
            console.print(f"[green]✓ Loaded L2 index with {len(self.l2_data)} entries (dimension: {self.l2_index.d})[/green]")
        except Exception as e:
            console.print(f"[red]✗ Failed to load L2 index: {e}[/red]")
            
        # Load Cosine index
        try:
            self.cosine_index = faiss.read_index(COSINE_INDEX_FILE)
            meta_data = np.load(COSINE_META_FILE, allow_pickle=True)
            self.cosine_data = meta_data['data'].tolist()
            console.print(f"[green]✓ Loaded Cosine index with {len(self.cosine_data)} entries (dimension: {self.cosine_index.d})[/green]")
        except Exception as e:
            console.print(f"[red]✗ Failed to load Cosine index: {e}[/red]")
    
    def load_test_data(self) -> List[Dict]:
        """Load test dataset"""
        console.print(f"[cyan]Loading test data from {TEST_FILE}...[/cyan]")
        test_data = []
        
        with open(TEST_FILE, "r", encoding="utf-8") as f:
            for line in f:
                test_data.append(json.loads(line))
        
        console.print(f"[green]✓ Loaded {len(test_data)} test cases[/green]")
        return test_data
    
    def retrieve_l2(self, query: str, k: int = 5) -> Tuple[List[Dict], np.ndarray]:
        """Retrieve using L2 distance"""
        query_vec = self.embedder.encode([f"query: {query}"], convert_to_numpy=True)
        D, I = self.l2_index.search(query_vec, k)
        results = [self.l2_data[idx] for idx in I[0]]
        return results, D[0]
    
    def retrieve_cosine(self, query: str, k: int = 5) -> Tuple[List[Dict], np.ndarray]:
        """Retrieve using Cosine similarity"""
        query_vec = self.embedder.encode([f"query: {query}"], convert_to_numpy=True)
        faiss.normalize_L2(query_vec)
        D, I = self.cosine_index.search(query_vec, k)
        results = [self.cosine_data[idx] for idx in I[0]]
        return results, D[0]
    
    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def check_command_match(self, retrieved_commands: List[str], expected_commands: List[str], threshold: float = 0.85) -> Tuple[bool, float]:
        """
        Check if any retrieved command matches any expected command using cosine similarity.
        Returns (match_found, best_similarity_score)
        """
        best_sim = 0.0
        match_found = False
        
        for retrieved in retrieved_commands:
            for expected in expected_commands:
                # First try exact match (faster)
                if retrieved.strip().lower() == expected.strip().lower():
                    return True, 1.0
                
                # Then compute cosine similarity
                emb1 = self.embedder.encode(f"passage: {retrieved}", convert_to_numpy=True)
                emb2 = self.embedder.encode(f"passage: {expected}", convert_to_numpy=True)
                sim = self.cosine_similarity(emb1, emb2)
                
                if sim > best_sim:
                    best_sim = sim
                if sim >= threshold:
                    match_found = True
                    
        return match_found, best_sim
    
    def evaluate_method(self, test_data: List[Dict], method: str, k: int = 5, diagnostic: bool = False) -> Dict:
        """Evaluate a specific retrieval method using cosine similarity for matching"""
        console.print(f"\n[bold cyan]Evaluating {method} with e5-base-v2 (cosine threshold={COSINE_SIMILARITY_THRESHOLD})...[/bold cyan]")
        
        total_tests = len(test_data)
        correct_retrievals = 0
        correct_by_difficulty = {0: 0, 1: 0, 2: 0}
        total_by_difficulty = {0: 0, 1: 0, 2: 0}
        
        # Diagnostic data
        all_distances = []
        top1_distances = []
        correct_distances = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"[cyan]Processing {total_tests} tests...", total=total_tests)
            
            for test_case in test_data:
                query = test_case['instruction']
                expected_commands = test_case['responses']
                difficulty = test_case.get('difficulty', 0)
                
                total_by_difficulty[difficulty] += 1
                
                # Retrieve based on method
                if method == "L2 Distance":
                    results, distances = self.retrieve_l2(query, k=k)
                    all_distances.extend(distances)
                    top1_distances.append(distances[0])
                else:  # Cosine Similarity
                    results, similarities = self.retrieve_cosine(query, k=k)
                    all_distances.extend(similarities)
                    top1_distances.append(similarities[0])
                
                # Check if any retrieved command matches expected using cosine similarity
                retrieved_commands = [r['response'] for r in results]
                match_found, best_sim = self.check_command_match(retrieved_commands, expected_commands, threshold=COSINE_SIMILARITY_THRESHOLD)
                
                if match_found:
                    correct_retrievals += 1
                    correct_by_difficulty[difficulty] += 1
                    
                    # Track the distance/similarity of the matching result
                    if method == "L2 Distance":
                        for i, r in enumerate(results):
                            r_match, _ = self.check_command_match([r['response']], expected_commands, threshold=COSINE_SIMILARITY_THRESHOLD)
                            if r_match:
                                correct_distances.append(distances[i])
                                break
                    else:
                        for i, r in enumerate(results):
                            r_match, _ = self.check_command_match([r['response']], expected_commands, threshold=COSINE_SIMILARITY_THRESHOLD)
                            if r_match:
                                correct_distances.append(similarities[i])
                                break
                
                progress.advance(task)
        
        accuracy = (correct_retrievals / total_tests) * 100 if total_tests > 0 else 0
        
        # Calculate accuracy by difficulty
        accuracy_by_difficulty = {}
        for diff in [0, 1, 2]:
            if total_by_difficulty[diff] > 0:
                accuracy_by_difficulty[diff] = (correct_by_difficulty[diff] / total_by_difficulty[diff]) * 100
            else:
                accuracy_by_difficulty[diff] = 0
        
        console.print(f"[green]✓ {method} Accuracy: {accuracy:.2f}% ({correct_retrievals}/{total_tests})[/green]")
        console.print(f"[dim]  Difficulty 0: {accuracy_by_difficulty[0]:.2f}% ({correct_by_difficulty[0]}/{total_by_difficulty[0]})[/dim]")
        console.print(f"[dim]  Difficulty 1: {accuracy_by_difficulty[1]:.2f}% ({correct_by_difficulty[1]}/{total_by_difficulty[1]})[/dim]")
        console.print(f"[dim]  Difficulty 2: {accuracy_by_difficulty[2]:.2f}% ({correct_by_difficulty[2]}/{total_by_difficulty[2]})[/dim]")
        
        # Print diagnostic statistics
        if diagnostic and all_distances:
            console.print(f"\n[yellow]📊 Diagnostic Statistics for {method}:[/yellow]")
            if method == "L2 Distance":
                console.print(f"[dim]  All distances - Min: {np.min(all_distances):.4f}, Max: {np.max(all_distances):.4f}, Mean: {np.mean(all_distances):.4f}, Median: {np.median(all_distances):.4f}[/dim]")
                console.print(f"[dim]  Top-1 distances - Min: {np.min(top1_distances):.4f}, Max: {np.max(top1_distances):.4f}, Mean: {np.mean(top1_distances):.4f}, Median: {np.median(top1_distances):.4f}[/dim]")
                if correct_distances:
                    console.print(f"[dim]  Correct match distances - Min: {np.min(correct_distances):.4f}, Max: {np.max(correct_distances):.4f}, Mean: {np.mean(correct_distances):.4f}, Median: {np.median(correct_distances):.4f}[/dim]")
                    console.print(f"[green]  → Suggested threshold (90th percentile of correct): {np.percentile(correct_distances, 90):.4f}[/green]")
            else:
                console.print(f"[dim]  All similarities - Min: {np.min(all_distances):.4f}, Max: {np.max(all_distances):.4f}, Mean: {np.mean(all_distances):.4f}, Median: {np.median(all_distances):.4f}[/dim]")
                console.print(f"[dim]  Top-1 similarities - Min: {np.min(top1_distances):.4f}, Max: {np.max(top1_distances):.4f}, Mean: {np.mean(top1_distances):.4f}, Median: {np.median(top1_distances):.4f}[/dim]")
                if correct_distances:
                    console.print(f"[dim]  Correct match similarities - Min: {np.min(correct_distances):.4f}, Max: {np.max(correct_distances):.4f}, Mean: {np.mean(correct_distances):.4f}, Median: {np.median(correct_distances):.4f}[/dim]")
                    console.print(f"[green]  → Suggested threshold (10th percentile of correct): {np.percentile(correct_distances, 10):.4f}[/green]")
        
        return {
            'method': method,
            'cosine_threshold': COSINE_SIMILARITY_THRESHOLD,
            'accuracy': accuracy,
            'correct': correct_retrievals,
            'total': total_tests,
            'accuracy_by_difficulty': accuracy_by_difficulty,
            'correct_by_difficulty': correct_by_difficulty,
            'total_by_difficulty': total_by_difficulty,
            'diagnostic_data': {
                'all_distances': all_distances,
                'top1_distances': top1_distances,
                'correct_distances': correct_distances
            }
        }
    
    def generate_comparison_graph(self, l2_results: Dict, cosine_results: Dict):
        """Generate bar graphs comparing the two methods"""
        console.print("\n[cyan]Generating comparison graphs...[/cyan]")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('RAG Retrieval Accuracy Comparison (E5-Base-V2)', fontsize=16, fontweight='bold')
        
        # 1. Overall Accuracy Comparison
        ax1 = axes[0, 0]
        methods = [l2_results['method'], cosine_results['method']]
        accuracies = [l2_results['accuracy'], cosine_results['accuracy']]
        colors = ['#3498db', '#e74c3c']
        
        bars1 = ax1.bar(methods, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Overall Accuracy', fontsize=13, fontweight='bold')
        ax1.set_ylim(0, 100)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for bar, acc in zip(bars1, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.2f}%',
                    ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        # Add cosine similarity threshold info
        ax1.text(0.5, -0.15, f'Model: e5-base-v2 | Cosine Similarity Threshold: {COSINE_SIMILARITY_THRESHOLD}',
                ha='center', transform=ax1.transAxes, fontsize=9, style='italic')
        
        # 2. Accuracy by Difficulty Level
        ax2 = axes[0, 1]
        difficulties = [0, 1, 2]
        l2_accs = [l2_results['accuracy_by_difficulty'][d] for d in difficulties]
        cosine_accs = [cosine_results['accuracy_by_difficulty'][d] for d in difficulties]
        
        x = np.arange(len(difficulties))
        width = 0.35
        
        bars2_1 = ax2.bar(x - width/2, l2_accs, width, label=l2_results['method'], 
                         color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
        bars2_2 = ax2.bar(x + width/2, cosine_accs, width, label=cosine_results['method'],
                         color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)
        
        ax2.set_xlabel('Difficulty Level', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Accuracy by Difficulty Level', fontsize=13, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(['Easy (0)', 'Medium (1)', 'Hard (2)'])
        ax2.set_ylim(0, 100)
        ax2.legend(fontsize=10)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels
        for bars in [bars2_1, bars2_2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax2.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.1f}%',
                            ha='center', va='bottom', fontsize=9)
        
        # 3. Correct Retrievals Count
        ax3 = axes[1, 0]
        correct_counts = [l2_results['correct'], cosine_results['correct']]
        
        bars3 = ax3.bar(methods, correct_counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax3.set_ylabel('Number of Correct Retrievals', fontsize=12, fontweight='bold')
        ax3.set_title('Correct Retrievals Count', fontsize=13, fontweight='bold')
        ax3.set_ylim(0, max(correct_counts) * 1.2)
        ax3.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels
        for bar, count in zip(bars3, correct_counts):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(count)}/{l2_results["total"]}',
                    ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        # 4. Comparison Table
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        table_data = [
            ['Metric', l2_results['method'], cosine_results['method']],
            ['Overall Accuracy', f"{l2_results['accuracy']:.2f}%", f"{cosine_results['accuracy']:.2f}%"],
            ['Correct/Total', f"{l2_results['correct']}/{l2_results['total']}", 
             f"{cosine_results['correct']}/{cosine_results['total']}"],
            ['Model', 'e5-base-v2', 'e5-base-v2'],
            ['Cosine Threshold', str(COSINE_SIMILARITY_THRESHOLD), str(COSINE_SIMILARITY_THRESHOLD)],
            ['Diff 0 Accuracy', f"{l2_results['accuracy_by_difficulty'][0]:.2f}%",
             f"{cosine_results['accuracy_by_difficulty'][0]:.2f}%"],
            ['Diff 1 Accuracy', f"{l2_results['accuracy_by_difficulty'][1]:.2f}%",
             f"{cosine_results['accuracy_by_difficulty'][1]:.2f}%"],
            ['Diff 2 Accuracy', f"{l2_results['accuracy_by_difficulty'][2]:.2f}%",
             f"{cosine_results['accuracy_by_difficulty'][2]:.2f}%"],
        ]
        
        table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                         colWidths=[0.35, 0.325, 0.325])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header row
        for i in range(3):
            table[(0, i)].set_facecolor('#34495e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(table_data)):
            for j in range(3):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#ecf0f1')
        
        ax4.set_title('Detailed Comparison', fontsize=13, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # Save figure
        output_file = 'rag_accuracy_comparison_e5.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        console.print(f"[green]✓ Saved comparison graph to {output_file}[/green]")
        plt.close(fig)

def main():
    console.print("[bold cyan]RAG Accuracy Evaluation - E5-Base-V2[/bold cyan]")
    console.print("=" * 60)
    
    evaluator = RAGAccuracyEvaluator()
    
    # Load indexes
    evaluator.load_indexes()
    
    if evaluator.l2_index is None or evaluator.cosine_index is None:
        console.print("\n[red]Error: Both indexes must be available for evaluation.[/red]")
        console.print("Please run:")
        console.print("  1. python index_build/build_faiss_l2_e5.py")
        console.print("  2. python index_build/build_faiss_cosine_e5.py")
        return
    
    # Load test data
    test_data = evaluator.load_test_data()
    
    # Evaluate L2 Distance with diagnostics
    l2_results = evaluator.evaluate_method(test_data, "L2 Distance", k=5, diagnostic=True)
    
    # Evaluate Cosine Similarity with diagnostics
    cosine_results = evaluator.evaluate_method(test_data, "Cosine Similarity", k=5, diagnostic=True)
    
    # Generate comparison graph
    evaluator.generate_comparison_graph(l2_results, cosine_results)
    
    # Summary
    console.print("\n[bold green]Evaluation Complete![/bold green]")
    console.print("=" * 60)
    console.print(f"[cyan]Model: e5-base-v2[/cyan]")
    console.print(f"[cyan]Matching Method: Cosine Similarity (threshold={COSINE_SIMILARITY_THRESHOLD})[/cyan]")
    console.print(f"[cyan]L2 Distance Retrieval:[/cyan] {l2_results['accuracy']:.2f}%")
    console.print(f"[cyan]Cosine Similarity Retrieval:[/cyan] {cosine_results['accuracy']:.2f}%")
    
    if cosine_results['accuracy'] > l2_results['accuracy']:
        improvement = cosine_results['accuracy'] - l2_results['accuracy']
        console.print(f"\n[green]✓ Cosine Similarity performs better by {improvement:.2f}%[/green]")
    elif l2_results['accuracy'] > cosine_results['accuracy']:
        improvement = l2_results['accuracy'] - cosine_results['accuracy']
        console.print(f"\n[green]✓ L2 Distance performs better by {improvement:.2f}%[/green]")
    else:
        console.print(f"\n[yellow]Both methods perform equally[/yellow]")

if __name__ == "__main__":
    main()
