import json
import time
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import faiss
import os

DATA_FILE = "NL2SH-ALFA_train_simple.json"

# L2 Distance files
L2_INDEX_FILE = "bash_commands_l2.bin"
L2_META_FILE = "metadata_l2.npz"

# Cosine Similarity files
COSINE_INDEX_FILE = "bash_commands_cosine.bin"
COSINE_META_FILE = "metadata_cosine.npz"

class IndexBuildTimer:
    def __init__(self):
        self.build_times = {
            'L2 Distance': None,
            'Cosine Similarity': None
        }
        self.load_times = {
            'L2 Distance': None,
            'Cosine Similarity': None
        }
        self.file_sizes = {
            'L2 Distance Index': None,
            'L2 Distance Metadata': None,
            'Cosine Similarity Index': None,
            'Cosine Similarity Metadata': None
        }
        self.breakdown_times = {
            'L2 Distance': {'encode': None, 'index_build': None, 'save': None},
            'Cosine Similarity': {'encode': None, 'normalize': None, 'index_build': None, 'save': None}
        }
        
    def build_l2_index(self):
        """Build L2 Distance FAISS index and record time"""
        print("\n" + "="*80)
        print("BUILDING L2 DISTANCE INDEX")
        print("="*80)
        
        # Check if already exists
        if os.path.exists(L2_INDEX_FILE) and os.path.exists(L2_META_FILE):
            print("Loading existing L2 index from disk...")
            load_start = time.time()
            index = faiss.read_index(L2_INDEX_FILE)
            meta_data = np.load(L2_META_FILE, allow_pickle=True)
            data = meta_data['data'].tolist()
            load_time = time.time() - load_start
            self.load_times['L2 Distance'] = load_time
            print(f"✓ Loaded in {load_time:.4f} seconds")
            
            # Record file sizes
            self.file_sizes['L2 Distance Index'] = os.path.getsize(L2_INDEX_FILE) / (1024 * 1024)  # MB
            self.file_sizes['L2 Distance Metadata'] = os.path.getsize(L2_META_FILE) / (1024 * 1024)  # MB
            print(f"✓ Index file size: {self.file_sizes['L2 Distance Index']:.2f} MB")
            print(f"✓ Metadata file size: {self.file_sizes['L2 Distance Metadata']:.2f} MB")
            return index, data
        
        # Build from scratch
        total_build_start = time.time()
        
        print("Loading dataset...")
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        texts = [item["nl"] for item in data]
        print(f"Loaded {len(texts)} texts")
        
        # Encode texts
        print("Encoding texts with SentenceTransformer...")
        encode_start = time.time()
        embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        embeddings = embedder.encode(texts, convert_to_numpy=True)
        encode_time = time.time() - encode_start
        print(f"✓ Encoded {len(embeddings)} embeddings in {encode_time:.4f} seconds")
        
        # Build index
        print("Building FAISS index with L2 distance...")
        index_build_start = time.time()
        d = embeddings.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(embeddings)
        index_build_time = time.time() - index_build_start
        print(f"✓ Index building time: {index_build_time:.4f} seconds")
        
        # Save files
        print("Saving index and metadata...")
        save_start = time.time()
        faiss.write_index(index, L2_INDEX_FILE)
        np.savez_compressed(L2_META_FILE, data=np.array(data, dtype=object))
        save_time = time.time() - save_start
        print(f"✓ Saving time: {save_time:.4f} seconds")
        
        # Total time
        total_build_time = time.time() - total_build_start
        self.build_times['L2 Distance'] = total_build_time
        self.breakdown_times['L2 Distance']['encode'] = encode_time
        self.breakdown_times['L2 Distance']['index_build'] = index_build_time
        self.breakdown_times['L2 Distance']['save'] = save_time
        print(f"\n✓✓✓ TOTAL L2 DISTANCE BUILD TIME: {total_build_time:.4f} seconds ({total_build_time/60:.2f} minutes) ✓✓✓")
        
        # Record file sizes
        self.file_sizes['L2 Distance Index'] = os.path.getsize(L2_INDEX_FILE) / (1024 * 1024)  # MB
        self.file_sizes['L2 Distance Metadata'] = os.path.getsize(L2_META_FILE) / (1024 * 1024)  # MB
        print(f"✓ Index file size: {self.file_sizes['L2 Distance Index']:.2f} MB")
        print(f"✓ Metadata file size: {self.file_sizes['L2 Distance Metadata']:.2f} MB")
        
        # Now load it back to record loading time
        print("\nMeasuring loading time...")
        load_start = time.time()
        index = faiss.read_index(L2_INDEX_FILE)
        meta_data = np.load(L2_META_FILE, allow_pickle=True)
        data = meta_data['data'].tolist()
        load_time = time.time() - load_start
        self.load_times['L2 Distance'] = load_time
        print(f"✓ Loading time: {load_time:.4f} seconds")
        
        return index, data
    
    def build_cosine_index(self):
        """Build Cosine Similarity FAISS index and record time"""
        print("\n" + "="*80)
        print("BUILDING COSINE SIMILARITY INDEX")
        print("="*80)
        
        # Check if already exists
        if os.path.exists(COSINE_INDEX_FILE) and os.path.exists(COSINE_META_FILE):
            print("Loading existing Cosine index from disk...")
            load_start = time.time()
            index = faiss.read_index(COSINE_INDEX_FILE)
            meta_data = np.load(COSINE_META_FILE, allow_pickle=True)
            data = meta_data['data'].tolist()
            load_time = time.time() - load_start
            self.load_times['Cosine Similarity'] = load_time
            print(f"✓ Loaded in {load_time:.4f} seconds")
            
            # Record file sizes
            self.file_sizes['Cosine Similarity Index'] = os.path.getsize(COSINE_INDEX_FILE) / (1024 * 1024)  # MB
            self.file_sizes['Cosine Similarity Metadata'] = os.path.getsize(COSINE_META_FILE) / (1024 * 1024)  # MB
            print(f"✓ Index file size: {self.file_sizes['Cosine Similarity Index']:.2f} MB")
            print(f"✓ Metadata file size: {self.file_sizes['Cosine Similarity Metadata']:.2f} MB")
            return index, data
        
        # Build from scratch - record TOTAL time
        total_build_start = time.time()
        
        print("Loading dataset...")
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        texts = [item["nl"] for item in data]
        print(f"Loaded {len(texts)} texts")
        
        # Encode texts
        print("Encoding texts with SentenceTransformer...")
        encode_start = time.time()
        embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        embeddings = embedder.encode(texts, convert_to_numpy=True)
        encode_time = time.time() - encode_start
        print(f"✓ Encoded {len(embeddings)} embeddings in {encode_time:.4f} seconds")
        
        # Normalize for cosine similarity
        print("Normalizing embeddings...")
        norm_start = time.time()
        faiss.normalize_L2(embeddings)
        norm_time = time.time() - norm_start
        print(f"✓ Normalization time: {norm_time:.4f} seconds")
        
        # Build index
        print("Building FAISS index with Cosine similarity...")
        index_build_start = time.time()
        d = embeddings.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(embeddings)
        index_build_time = time.time() - index_build_start
        print(f"✓ Index building time: {index_build_time:.4f} seconds")
        
        # Save files
        print("Saving index and metadata...")
        save_start = time.time()
        faiss.write_index(index, COSINE_INDEX_FILE)
        np.savez_compressed(COSINE_META_FILE, data=np.array(data, dtype=object))
        save_time = time.time() - save_start
        print(f"✓ Saving time: {save_time:.4f} seconds")
        
        # Total time
        total_build_time = time.time() - total_build_start
        self.build_times['Cosine Similarity'] = total_build_time
        self.breakdown_times['Cosine Similarity']['encode'] = encode_time
        self.breakdown_times['Cosine Similarity']['normalize'] = norm_time
        self.breakdown_times['Cosine Similarity']['index_build'] = index_build_time
        self.breakdown_times['Cosine Similarity']['save'] = save_time
        print(f"\n✓✓✓ TOTAL COSINE SIMILARITY BUILD TIME: {total_build_time:.4f} seconds ({total_build_time/60:.2f} minutes) ✓✓✓")
        
        # Record file sizes
        self.file_sizes['Cosine Similarity Index'] = os.path.getsize(COSINE_INDEX_FILE) / (1024 * 1024)  # MB
        self.file_sizes['Cosine Similarity Metadata'] = os.path.getsize(COSINE_META_FILE) / (1024 * 1024)  # MB
        print(f"✓ Index file size: {self.file_sizes['Cosine Similarity Index']:.2f} MB")
        print(f"✓ Metadata file size: {self.file_sizes['Cosine Similarity Metadata']:.2f} MB")
        
        # Now load it back to record loading time
        print("\nMeasuring loading time...")
        load_start = time.time()
        index = faiss.read_index(COSINE_INDEX_FILE)
        meta_data = np.load(COSINE_META_FILE, allow_pickle=True)
        data = meta_data['data'].tolist()
        load_time = time.time() - load_start
        self.load_times['Cosine Similarity'] = load_time
        print(f"✓ Loading time: {load_time:.4f} seconds")
        
        return index, data
    
    def print_summary(self):
        """Print timing summary"""
        print("\n" + "="*80)
        print("BUILD TIME SUMMARY")
        print("="*80)
        
        print("\nTOTAL Index Building Times:")
        for method, build_time in self.build_times.items():
            if build_time is not None:
                print(f"  {method}: {build_time:.4f} seconds ({build_time/60:.2f} minutes)")
        
        print("\nIndex Loading Times:")
        for method, load_time in self.load_times.items():
            if load_time is not None:
                print(f"  {method}: {load_time:.4f} seconds")
        
        print("\nBuild Time Breakdown:")
        for method, times in self.breakdown_times.items():
            if any(t is not None for t in times.values()):
                print(f"  {method}:")
                for operation, op_time in times.items():
                    if op_time is not None:
                        print(f"    - {operation.replace('_', ' ').title()}: {op_time:.4f}s")
        
        print("\nFile Sizes:")
        for file_name, size in self.file_sizes.items():
            if size is not None:
                print(f"  {file_name}: {size:.2f} MB")
        
        print("="*80 + "\n")
    
    def plot_build_times(self):
        """Generate comparison graphs for build times"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Index Building & Loading Performance Comparison', fontsize=16, fontweight='bold')
        
        # Colors
        color_l2 = '#FF1493'  # Deep Pink
        color_cosine = '#8B00FF'  # Violet
        
        # Plot 1: Build times comparison (in seconds and minutes)
        methods = [m for m, t in self.build_times.items() if t is not None]
        build_times_sec = [t for t in self.build_times.values() if t is not None]
        build_times_min = [t/60 for t in build_times_sec]
        
        if build_times_sec:
            colors = [color_l2, color_cosine][:len(methods)]
            bars = axes[0, 0].bar(methods, build_times_sec, color=colors, alpha=0.85)
            axes[0, 0].set_ylabel('Time (seconds)', fontweight='bold')
            axes[0, 0].set_title('Total Index Building Time', fontweight='bold')
            axes[0, 0].grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bar, time_sec, time_min in zip(bars, build_times_sec, build_times_min):
                height = bar.get_height()
                label_text = f'{time_min:.2f}m\n({time_sec:.0f}s)'
                axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                               label_text,
                               ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        # Plot 2: Load times comparison
        methods_load = [m for m, t in self.load_times.items() if t is not None]
        load_times = [t for t in self.load_times.values() if t is not None]
        
        if load_times:
            colors = [color_l2, color_cosine][:len(methods_load)]
            bars = axes[0, 1].bar(methods_load, load_times, color=colors, alpha=0.85)
            axes[0, 1].set_ylabel('Time (seconds)', fontweight='bold')
            axes[0, 1].set_title('Index Loading Time', fontweight='bold')
            axes[0, 1].grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bar, time in zip(bars, load_times):
                height = bar.get_height()
                axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                               f'{time:.4f}s',
                               ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # Plot 4: File sizes comparison
        file_names = list(self.file_sizes.keys())
        file_sizes = [s for s in self.file_sizes.values() if s is not None]
        
        if file_sizes:
            colors_files = [color_l2, color_l2, color_cosine, color_cosine][:len(file_sizes)]
            bars = axes[1, 0].barh(file_names, file_sizes, color=colors_files, alpha=0.85)
            axes[1, 0].set_xlabel('File Size (MB)', fontweight='bold')
            axes[1, 0].set_title('Index and Metadata File Sizes', fontweight='bold')
            axes[1, 0].grid(axis='x', alpha=0.3)
            
            # Add value labels
            for bar, size in zip(bars, file_sizes):
                width = bar.get_width()
                axes[1, 0].text(width, bar.get_y() + bar.get_height()/2.,
                               f'{size:.2f} MB',
                               ha='left', va='center', fontweight='bold', fontsize=10)
        
        # Plot 5: Summary metrics
        axes[1, 1].axis('off')
        summary_text = "PERFORMANCE SUMMARY\n"
        summary_text += "="*45 + "\n\n"
        
        if self.build_times['L2 Distance'] is not None:
            bt = self.build_times['L2 Distance']
            lt = self.load_times['L2 Distance'] if self.load_times['L2 Distance'] is not None else 0
            summary_text += f"L2 Distance:\n"
            summary_text += f"  Build Time: {bt/60:.2f}m ({bt:.0f}s)\n"
            summary_text += f"  Load Time: {lt:.4f}s\n"
            total_size = self.file_sizes['L2 Distance Index'] + self.file_sizes['L2 Distance Metadata']
            summary_text += f"  Total Size: {total_size:.2f} MB\n\n"
        
        if self.build_times['Cosine Similarity'] is not None:
            bt = self.build_times['Cosine Similarity']
            lt = self.load_times['Cosine Similarity'] if self.load_times['Cosine Similarity'] is not None else 0
            summary_text += f"Cosine Similarity:\n"
            summary_text += f"  Build Time: {bt/60:.2f}m ({bt:.0f}s)\n"
            summary_text += f"  Load Time: {lt:.4f}s\n"
            total_size = self.file_sizes['Cosine Similarity Index'] + self.file_sizes['Cosine Similarity Metadata']
            summary_text += f"  Total Size: {total_size:.2f} MB\n\n"
        
        if self.breakdown_times['L2 Distance']['encode'] is not None:
            summary_text += "Build Time Breakdown:\n"
            summary_text += "L2 Distance:\n"
            for op, time in self.breakdown_times['L2 Distance'].items():
                if time is not None:
                    pct = (time / self.build_times['L2 Distance']) * 100
                    summary_text += f"  {op.replace('_', ' ').title()}: {time:.2f}s ({pct:.1f}%)\n"
            summary_text += "\nCosine Similarity:\n"
            for op, time in self.breakdown_times['Cosine Similarity'].items():
                if time is not None:
                    pct = (time / self.build_times['Cosine Similarity']) * 100
                    summary_text += f"  {op.replace('_', ' ').title()}: {time:.2f}s ({pct:.1f}%)\n"
        
        axes[1, 1].text(0.05, 0.95, summary_text, fontsize=10, family='monospace',
                       verticalalignment='top', bbox=dict(boxstyle='round', 
                       facecolor='lightyellow', alpha=0.7))
        
        plt.tight_layout()
        
        # Save the plot
        output_file = 'index_building_times.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n✓ Graph saved as '{output_file}'")
        plt.show()

def main():
    timer = IndexBuildTimer()
    
    # Build both indexes
    l2_index, l2_data = timer.build_l2_index()
    cosine_index, cosine_data = timer.build_cosine_index()
    
    # Print summary
    timer.print_summary()
    
    # Generate graphs
    timer.plot_build_times()

if __name__ == "__main__":
    main()
