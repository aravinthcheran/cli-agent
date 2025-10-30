import json
import pickle
import time
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

# File paths
DATA_FILE = "NL2SH-ALFA_train_simple.json"
L2_INDEX_FILE = "bash_commands_l2.bin"
L2_META_FILE = "metadata_l2.npz"
COSINE_INDEX_FILE = "bash_commands_cosine.bin"
COSINE_META_FILE = "metadata_cosine.npz"

class RetrievalComparison:
    def __init__(self):
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.l2_index = None
        self.cosine_index = None
        self.l2_data = None
        self.cosine_data = None
        
    def load_indexes(self):
        """Load both L2 and Cosine similarity indexes"""
        print("Loading indexes...")
        
        # Load L2 index
        if os.path.exists(L2_INDEX_FILE) and os.path.exists(L2_META_FILE):
            self.l2_index = faiss.read_index(L2_INDEX_FILE)
            meta_data = np.load(L2_META_FILE, allow_pickle=True)
            self.l2_data = meta_data['data'].tolist()
            print(f"✓ Loaded L2 index with {len(self.l2_data)} entries")
        else:
            print("✗ L2 index not found. Please run build_faiss.py first.")
            
        # Load Cosine index
        if os.path.exists(COSINE_INDEX_FILE) and os.path.exists(COSINE_META_FILE):
            self.cosine_index = faiss.read_index(COSINE_INDEX_FILE)
            meta_data = np.load(COSINE_META_FILE, allow_pickle=True)
            self.cosine_data = meta_data['data'].tolist()
            print(f"✓ Loaded Cosine index with {len(self.cosine_data)} entries")
        else:
            print("✗ Cosine index not found. Please run build_faiss_cosine.py first.")
    
    def retrieve_l2(self, query: str, k: int = 3) -> Tuple[List[Dict], float, np.ndarray]:
        """Retrieve using L2 distance"""
        start_time = time.time()
        query_vec = self.embedder.encode([query], convert_to_numpy=True)
        D, I = self.l2_index.search(query_vec, k)
        query_time = time.time() - start_time
        results = [self.l2_data[idx] for idx in I[0]]
        return results, query_time, D[0]
    
    def retrieve_cosine(self, query: str, k: int = 3) -> Tuple[List[Dict], float, np.ndarray]:
        """Retrieve using Cosine similarity"""
        start_time = time.time()
        query_vec = self.embedder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_vec)  # Normalize for cosine similarity
        D, I = self.cosine_index.search(query_vec, k)
        query_time = time.time() - start_time
        results = [self.cosine_data[idx] for idx in I[0]]
        return results, query_time, D[0]
    
    def run_comparison(self, test_queries: List[str], k: int = 5):
        """Run comparison on multiple queries"""
        l2_times = []
        cosine_times = []
        l2_distances = []
        cosine_similarities = []
        
        print("\n" + "="*80)
        print("RETRIEVAL COMPARISON: L2 Distance vs Cosine Similarity")
        print("="*80 + "\n")
        
        for i, query in enumerate(test_queries, 1):
            print(f"\nQuery {i}: '{query}'")
            print("-" * 80)
            
            # L2 Distance retrieval
            l2_results, l2_time, l2_dists = self.retrieve_l2(query, k)
            l2_times.append(l2_time)
            l2_distances.append(l2_dists)
            
            print(f"\n[L2 Distance] Query time: {l2_time*1000:.4f} ms | Avg Distance: {np.mean(l2_dists):.6f}")
            print("Top results:")
            for j, (result, dist) in enumerate(zip(l2_results, l2_dists), 1):
                print(f"  {j}. Distance: {dist:.6f} | NL: {result['nl'][:60]}... | CMD: {result['bash']}")
            
            # Cosine Similarity retrieval
            cosine_results, cosine_time, cosine_sims = self.retrieve_cosine(query, k)
            cosine_times.append(cosine_time)
            cosine_similarities.append(cosine_sims)
            
            print(f"\n[Cosine Similarity] Query time: {cosine_time*1000:.4f} ms | Avg Similarity: {np.mean(cosine_sims):.6f}")
            print("Top results:")
            for j, (result, score) in enumerate(zip(cosine_results, cosine_sims), 1):
                print(f"  {j}. Similarity: {score:.6f} | NL: {result['nl'][:60]}... | CMD: {result['bash']}")
            
            print("\n" + "-" * 80)
        
        # Summary statistics
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        print(f"\nL2 Distance:")
        print(f"  Average query time: {np.mean(l2_times)*1000:.4f} ms")
        print(f"  Min query time: {np.min(l2_times)*1000:.4f} ms")
        print(f"  Max query time: {np.max(l2_times)*1000:.4f} ms")
        print(f"  Std dev: {np.std(l2_times)*1000:.4f} ms")
        print(f"  Average distance: {np.mean([np.mean(d) for d in l2_distances]):.6f}")
        
        print(f"\nCosine Similarity:")
        print(f"  Average query time: {np.mean(cosine_times)*1000:.4f} ms")
        print(f"  Min query time: {np.min(cosine_times)*1000:.4f} ms")
        print(f"  Max query time: {np.max(cosine_times)*1000:.4f} ms")
        print(f"  Std dev: {np.std(cosine_times)*1000:.4f} ms")
        print(f"  Average similarity: {np.mean([np.mean(s) for s in cosine_similarities]):.6f}")
        
        speedup = np.mean(l2_times) / np.mean(cosine_times)
        print(f"\nSpeedup factor: {speedup:.2f}x")
        print("="*80 + "\n")
        
        return l2_times, cosine_times, test_queries, l2_distances, cosine_similarities
    
    def plot_comparison(self, l2_times: List[float], cosine_times: List[float], queries: List[str], 
                        l2_distances: List[np.ndarray], cosine_similarities: List[np.ndarray]):
        """Generate comparison graphs"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Retrieval Method Comparison: L2 Distance vs Cosine Similarity', 
                     fontsize=16, fontweight='bold')
        
        # Convert to milliseconds
        l2_times_ms = [t * 1000 for t in l2_times]
        cosine_times_ms = [t * 1000 for t in cosine_times]
        
        # Calculate average distances and similarities per query
        avg_l2_distances = [np.mean(d) for d in l2_distances]
        avg_cosine_similarities = [np.mean(s) for s in cosine_similarities]
        
        # Colors: Pink and Violet
        color_l2 = '#FF1493'  # Deep Pink
        color_cosine = '#8B00FF'  # Violet
        
        # Plot 1: Query times comparison
        x = np.arange(len(queries))
        width = 0.35
        axes[0, 0].bar(x - width/2, l2_times_ms, width, label='L2 Distance', alpha=0.85, color=color_l2)
        axes[0, 0].bar(x + width/2, cosine_times_ms, width, label='Cosine Similarity', alpha=0.85, color=color_cosine)
        axes[0, 0].set_xlabel('Query Number', fontweight='bold')
        axes[0, 0].set_ylabel('Query Time (ms)', fontweight='bold')
        axes[0, 0].set_title('Query Time Comparison', fontweight='bold')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels([f'Q{i+1}' for i in range(len(queries))])
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Plot 2: Violin plot for distribution (better visualization)
        parts = axes[0, 1].violinplot([l2_times_ms, cosine_times_ms], 
                                      positions=[1, 2], 
                                      showmeans=True, 
                                      showmedians=True)
        
        # Color the violin plots
        for i, pc in enumerate(parts['bodies']):
            if i == 0:
                pc.set_facecolor(color_l2)
            else:
                pc.set_facecolor(color_cosine)
            pc.set_alpha(0.7)
        
        axes[0, 1].set_ylabel('Query Time (ms)', fontweight='bold')
        axes[0, 1].set_title('Query Time Distribution (Violin Plot)', fontweight='bold')
        axes[0, 1].set_xticks([1, 2])
        axes[0, 1].set_xticklabels(['L2 Distance', 'Cosine Similarity'])
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # Plot 3: Average comparison bar chart for query times
        methods = ['L2 Distance', 'Cosine Similarity']
        avg_times = [np.mean(l2_times_ms), np.mean(cosine_times_ms)]
        colors = [color_l2, color_cosine]
        bars = axes[0, 2].bar(methods, avg_times, color=colors, alpha=0.85)
        axes[0, 2].set_ylabel('Average Query Time (ms)', fontweight='bold')
        axes[0, 2].set_title('Average Query Time', fontweight='bold')
        axes[0, 2].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, time in zip(bars, avg_times):
            height = bar.get_height()
            axes[0, 2].text(bar.get_x() + bar.get_width()/2., height,
                           f'{time:.4f} ms',
                           ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # Plot 4: Cumulative time comparison
        cumulative_l2 = np.cumsum(l2_times_ms)
        cumulative_cosine = np.cumsum(cosine_times_ms)
        axes[1, 0].plot(range(1, len(queries)+1), cumulative_l2, 
                       marker='o', label='L2 Distance', linewidth=2.5, color=color_l2, markersize=8)
        axes[1, 0].plot(range(1, len(queries)+1), cumulative_cosine, 
                       marker='s', label='Cosine Similarity', linewidth=2.5, color=color_cosine, markersize=8)
        axes[1, 0].set_xlabel('Number of Queries', fontweight='bold')
        axes[1, 0].set_ylabel('Cumulative Time (ms)', fontweight='bold')
        axes[1, 0].set_title('Cumulative Query Time', fontweight='bold')
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(alpha=0.3)
        
        # Plot 5: Average Distance vs Average Similarity per query
        x = np.arange(len(queries))
        width = 0.35
        axes[1, 1].bar(x - width/2, avg_l2_distances, width, label='L2 Distance', alpha=0.85, color=color_l2)
        axes[1, 1].bar(x + width/2, avg_cosine_similarities, width, label='Cosine Similarity', alpha=0.85, color=color_cosine)
        axes[1, 1].set_xlabel('Query Number', fontweight='bold')
        axes[1, 1].set_ylabel('Score', fontweight='bold')
        axes[1, 1].set_title('Average Distance vs Similarity per Query', fontweight='bold')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels([f'Q{i+1}' for i in range(len(queries))])
        axes[1, 1].legend(fontsize=10)
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        # Plot 6: Overall average distance and similarity
        avg_dist = np.mean(avg_l2_distances)
        avg_sim = np.mean(avg_cosine_similarities)
        metrics = ['L2 Distance', 'Cosine Similarity']
        scores = [avg_dist, avg_sim]
        bars = axes[1, 2].bar(metrics, scores, color=[color_l2, color_cosine], alpha=0.85)
        axes[1, 2].set_ylabel('Average Score', fontweight='bold')
        axes[1, 2].set_title('Overall Average Distance/Similarity', fontweight='bold')
        axes[1, 2].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            axes[1, 2].text(bar.get_x() + bar.get_width()/2., height,
                           f'{score:.6f}',
                           ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        
        # Save the plot
        output_file = 'retrieval_comparison.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n✓ Graph saved as '{output_file}'")

def main():
    # Test queries
    test_queries = [
        "show free space on all filesystems",
        "list all running processes",
        "find files modified in the last 7 days",
        "display network connections",
        "check disk usage",
        "show current directory contents",
        "find large files over 100MB",
        "display system memory usage",
        "list all users",
        "show current logged in users"
    ]
    
    # Initialize comparison
    comparator = RetrievalComparison()
    comparator.load_indexes()
    
    if comparator.l2_index is None or comparator.cosine_index is None:
        print("\nError: Both indexes must be available for comparison.")
        print("Please run:")
        print("  1. python build_faiss.py")
        print("  2. python build_faiss_cosine.py")
        return
    
    # Run comparison
    l2_times, cosine_times, queries, l2_distances, cosine_similarities = comparator.run_comparison(test_queries, k=5)
    
    # Generate plots
    comparator.plot_comparison(l2_times, cosine_times, queries, l2_distances, cosine_similarities)

if __name__ == "__main__":
    main()
