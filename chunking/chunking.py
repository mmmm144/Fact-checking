#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Semantic Chunking Module for Vietnamese Fact-Checking Dataset Generation
"""

import os
import re
import sys
import json
import argparse
import numpy as np
from typing import List, Dict, Any, Tuple

# Reconfigure stdout and stderr to UTF-8 to prevent encoding errors on Windows terminals
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass
if sys.stderr.encoding != 'utf-8':
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass
from underthesea import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Constants
MIN_SENTENCES = 3
MAX_SENTENCES = 8
DEFAULT_SIMILARITY_THRESHOLD = 0.65
DYNAMIC_THRESHOLD_MIN = 0.55
OUTLIER_THRESHOLD = 0.50
MERGE_CENTROID_THRESHOLD = 0.85
MIN_WORD_COUNT = 30

class SemanticChunker:
    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = None):
        print(f"Loading SentenceTransformer model: {model_name}...")
        self.model = SentenceTransformer(model_name, device=device)
        print("Model loaded successfully.")

    def segment_sentences(self, text: str) -> List[str]:
        """
        Step 1: Sentence Segmentation
        Use underthesea.sent_tokenize and filter out empty sentences or sentences < 10 characters.
        """
        if not text or not text.strip():
            return []
        
        try:
            raw_sentences = sent_tokenize(text)
        except Exception as e:
            # Fallback segmenter using regex
            raw_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
            
        valid_sentences = []
        for s in raw_sentences:
            s_clean = s.strip()
            # Filter empty and length < 10 characters
            if s_clean and len(s_clean) >= 10:
                valid_sentences.append(s_clean)
                
        return valid_sentences

    def get_embeddings(self, sentences: List[str]) -> np.ndarray:
        """
        Step 2: Sentence Embedding
        Generate normalized sentence embeddings using BAAI/bge-m3.
        """
        if not sentences:
            return np.empty((0, 1024))
        # normalize_embeddings=True ensures unit vectors for cosine similarity calculation
        embeddings = self.model.encode(sentences, normalize_embeddings=True, show_progress_bar=False)
        return np.array(embeddings)

    def calculate_adjacent_similarities(self, embeddings: np.ndarray) -> List[float]:
        """
        Step 3: Adjacent Similarity Calculation
        Compute cosine similarity between consecutive sentence embeddings.
        """
        similarities = []
        n = len(embeddings)
        for i in range(n - 1):
            # Calculate cosine similarity using sklearn
            sim = cosine_similarity(
                embeddings[i].reshape(1, -1),
                embeddings[i+1].reshape(1, -1)
            )[0][0]
            similarities.append(float(sim))
        return similarities

    def detect_boundaries(self, similarities: List[float]) -> Tuple[List[int], float]:
        """
        Step 4 & 5: Boundary Detection & Dynamic Threshold
        Calculate dynamic threshold and find indices where adjacent similarity < threshold.
        """
        if not similarities:
            return [], DEFAULT_SIMILARITY_THRESHOLD
            
        mean_sim = np.mean(similarities)
        std_sim = np.std(similarities)
        
        # dynamic_threshold = mean_similarity - 0.5 * std_similarity
        dynamic_threshold = mean_sim - 0.5 * std_sim
        threshold = max(DYNAMIC_THRESHOLD_MIN, dynamic_threshold)
        
        boundaries = []
        for i, sim in enumerate(similarities):
            if sim < threshold:
                boundaries.append(i + 1) # Boundary is after index i (so index i + 1 starts new chunk)
                
        return boundaries, threshold

    def _get_centroid(self, indices: List[int], embeddings: np.ndarray) -> np.ndarray:
        """Helper to get normalized centroid for a list of sentence indices."""
        chunk_embs = embeddings[indices]
        centroid = np.mean(chunk_embs, axis=0)
        # Normalize centroid
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        return centroid

    def _split_large_chunk(self, chunk_indices: List[int], similarities: List[float]) -> List[List[int]]:
        """
        Helper to split chunks > MAX_SENTENCES while respecting MIN_SENTENCES constraint if possible.
        """
        n = len(chunk_indices)
        if n <= MAX_SENTENCES:
            return [chunk_indices]
            
        # We need to find a split point index `split_idx` (relative to the chunk)
        # valid split points must keep at least MIN_SENTENCES on each side
        # So split_idx must be in range [MIN_SENTENCES, n - MIN_SENTENCES]
        start_idx = MIN_SENTENCES
        end_idx = n - MIN_SENTENCES
        
        if start_idx > end_idx:
            # If S = 5 and we want max=4, min=3, start=3, end=2 (not possible to satisfy both)
            # Just split in half
            split_point = n // 2
        else:
            # Find the minimum similarity point in the valid range
            min_sim = float('inf')
            split_point = n // 2
            for idx in range(start_idx, end_idx + 1):
                # The similarity between sentence (idx-1) and sentence (idx)
                global_sim_idx = chunk_indices[0] + idx - 1
                if global_sim_idx < len(similarities):
                    sim = similarities[global_sim_idx]
                    if sim < min_sim:
                        min_sim = sim
                        split_point = idx
                        
        left_chunk = chunk_indices[:split_point]
        right_chunk = chunk_indices[split_point:]
        
        # Recursively split if they are still larger than MAX_SENTENCES
        result = []
        for sub_chunk in [left_chunk, right_chunk]:
            if len(sub_chunk) > MAX_SENTENCES:
                result.extend(self._split_large_chunk(sub_chunk, similarities))
            else:
                result.append(sub_chunk)
        return result

    def enforce_size_constraints(self, initial_chunks: List[List[int]], embeddings: np.ndarray, similarities: List[float]) -> List[List[int]]:
        """
        Step 6: Chunk Size Constraints
        Enforce MIN_SENTENCES and MAX_SENTENCES constraints.
        """
        if not initial_chunks:
            return []
            
        # If the whole article is extremely short, return as a single chunk
        total_sentences = len(embeddings)
        if total_sentences <= MIN_SENTENCES:
            return [list(range(total_sentences))]

        # Iteratively merge chunks that are < MIN_SENTENCES
        chunks = [list(c) for c in initial_chunks]
        
        changed = True
        while changed and len(chunks) > 1:
            changed = False
            for i in range(len(chunks)):
                if len(chunks[i]) < MIN_SENTENCES:
                    # Calculate similarity with neighbors
                    centroid_curr = self._get_centroid(chunks[i], embeddings)
                    
                    sim_prev = -1.0
                    if i > 0:
                        centroid_prev = self._get_centroid(chunks[i-1], embeddings)
                        sim_prev = float(cosine_similarity(centroid_curr.reshape(1, -1), centroid_prev.reshape(1, -1))[0][0])
                        
                    sim_next = -1.0
                    if i < len(chunks) - 1:
                        centroid_next = self._get_centroid(chunks[i+1], embeddings)
                        sim_next = float(cosine_similarity(centroid_curr.reshape(1, -1), centroid_next.reshape(1, -1))[0][0])
                        
                    # Merge with the neighbor with higher similarity
                    if sim_prev >= sim_next and i > 0:
                        chunks[i-1].extend(chunks[i])
                        chunks.pop(i)
                    elif sim_next > sim_prev and i < len(chunks) - 1:
                        chunks[i].extend(chunks[i+1])
                        chunks.pop(i+1)
                    else:
                        # Fallback (e.g. edge cases)
                        if i > 0:
                            chunks[i-1].extend(chunks[i])
                            chunks.pop(i)
                        else:
                            chunks[i].extend(chunks[i+1])
                            chunks.pop(i+1)
                            
                    changed = True
                    break
                    
        # Now split chunks that are > MAX_SENTENCES
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > MAX_SENTENCES:
                final_chunks.extend(self._split_large_chunk(chunk, similarities))
            else:
                final_chunks.append(chunk)
                
        return final_chunks

    def validate_topic_coherence(self, chunks: List[List[int]], embeddings: np.ndarray) -> List[List[int]]:
        """
        Step 7: Topic Coherence Validation
        Detect sentence outliers (score < 0.50) and reassign to adjacent chunks if more similar.
        """
        if len(chunks) <= 1:
            return chunks
            
        adjusted_chunks = [list(c) for c in chunks]
        
        # We inspect outliers at boundaries to maintain contiguity
        # Iterate over adjacent boundary pairs
        for i in range(len(adjusted_chunks) - 1):
            chunk_a = adjusted_chunks[i]
            chunk_b = adjusted_chunks[i+1]
            
            if not chunk_a or not chunk_b:
                continue
                
            centroid_a = self._get_centroid(chunk_a, embeddings)
            centroid_b = self._get_centroid(chunk_b, embeddings)
            
            # Check if last sentence of A is an outlier in A
            last_a_idx = chunk_a[-1]
            emb_last_a = embeddings[last_a_idx].reshape(1, -1)
            score_in_a = float(cosine_similarity(emb_last_a, centroid_a.reshape(1, -1))[0][0])
            
            if score_in_a < OUTLIER_THRESHOLD:
                # Compare similarity to B
                score_in_b = float(cosine_similarity(emb_last_a, centroid_b.reshape(1, -1))[0][0])
                if score_in_b > score_in_a:
                    # Move from A to B
                    chunk_a.pop()
                    chunk_b.insert(0, last_a_idx)
                    # Recompute centroids
                    centroid_a = self._get_centroid(chunk_a, embeddings) if chunk_a else centroid_a
                    centroid_b = self._get_centroid(chunk_b, embeddings) if chunk_b else centroid_b
                    
            # Check if first sentence of B is an outlier in B
            if not chunk_b:
                continue
            first_b_idx = chunk_b[0]
            emb_first_b = embeddings[first_b_idx].reshape(1, -1)
            score_in_b = float(cosine_similarity(emb_first_b, centroid_b.reshape(1, -1))[0][0])
            
            if score_in_b < OUTLIER_THRESHOLD:
                # Compare similarity to A
                score_in_a = float(cosine_similarity(emb_first_b, centroid_a.reshape(1, -1))[0][0])
                if score_in_a > score_in_b:
                    # Move from B to A
                    chunk_b.pop(0)
                    chunk_a.append(first_b_idx)
                    
        # Filter empty chunks that might result from reassignments
        return [c for c in adjusted_chunks if c]

    def merge_coherent_chunks(self, chunks: List[List[int]], embeddings: np.ndarray) -> List[List[int]]:
        """
        Step 8: Chunk Merge
        Merge consecutive chunks if their centroid similarity > 0.85.
        """
        if len(chunks) <= 1:
            return chunks
            
        merged_chunks = [list(c) for c in chunks]
        
        changed = True
        while changed and len(merged_chunks) > 1:
            changed = False
            for i in range(len(merged_chunks) - 1):
                centroid_a = self._get_centroid(merged_chunks[i], embeddings)
                centroid_b = self._get_centroid(merged_chunks[i+1], embeddings)
                
                sim = float(cosine_similarity(centroid_a.reshape(1, -1), centroid_b.reshape(1, -1))[0][0])
                if sim > MERGE_CENTROID_THRESHOLD:
                    # Merge them
                    merged_chunks[i].extend(merged_chunks[i+1])
                    merged_chunks.pop(i+1)
                    changed = True
                    break
                    
        return merged_chunks

    def filter_evidence_quality(self, chunks: List[List[int]], sentences: List[str]) -> List[List[int]]:
        """
        Step 9: Evidence Quality Filter
        Drop chunks if they:
        - have less than 30 words
        - only contain titles, sources, advertisements, or hyperlinks
        """
        filtered_chunks = []
        
        for chunk in chunks:
            chunk_sentences = [sentences[idx] for idx in chunk]
            chunk_text = " ".join(chunk_sentences)
            
            # 1. Word count filter
            words = chunk_text.split()
            if len(words) < MIN_WORD_COUNT:
                continue
                
            # 2. Check if chunk only contains source or metadata (e.g. "Nguồn: ...", "Ảnh: ...", "Liên hệ quảng cáo")
            is_noise = False
            
            # Patterns for source, ads, links
            source_pattern = re.compile(r'^(nguồn|theo|ảnh|tác giả|phóng viên|bài viết|credit):\s*.*$', re.IGNORECASE)
            ad_pattern = re.compile(r'(quảng cáo|liên hệ quảng cáo|đăng ký quảng cáo|hotline quảng cáo)', re.IGNORECASE)
            link_pattern = re.compile(r'^https?://[^\s]+$', re.IGNORECASE)
            
            # Check if all sentences in the chunk match noise patterns
            noise_sentence_count = 0
            for s in chunk_sentences:
                s_clean = s.strip()
                if (source_pattern.match(s_clean) or 
                    ad_pattern.search(s_clean) or 
                    link_pattern.match(s_clean) or
                    len(s_clean) < 15): # too short to contain a real fact
                    noise_sentence_count += 1
                    
            # If more than 75% of the chunk sentences are noise, reject the chunk
            if noise_sentence_count / len(chunk_sentences) >= 0.75:
                is_noise = True
                
            if not is_noise:
                filtered_chunks.append(chunk)
                
        return filtered_chunks

    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Executes the full 10-step semantic chunking pipeline.
        """
        # Step 1: Sentence Segmentation
        sentences = self.segment_sentences(text)
        if not sentences:
            return []
            
        # If very few sentences, return as a single chunk directly
        if len(sentences) <= MIN_SENTENCES:
            word_count = len(" ".join(sentences).split())
            if word_count >= MIN_WORD_COUNT:
                return [{
                    "chunk_id": 1,
                    "sentence_count": len(sentences),
                    "word_count": word_count,
                    "text": " ".join(sentences),
                    "sentences": sentences
                }]
            return []
            
        # Step 2: Sentence Embedding
        embeddings = self.get_embeddings(sentences)
        
        # Step 3: Adjacent Similarity Calculation
        similarities = self.calculate_adjacent_similarities(embeddings)
        
        # Step 4 & 5: Boundary Detection with Dynamic Threshold
        boundaries, threshold = self.detect_boundaries(similarities)
        
        # Create initial chunks from boundary indices
        initial_chunks = []
        start = 0
        for b in boundaries:
            initial_chunks.append(list(range(start, b)))
            start = b
        initial_chunks.append(list(range(start, len(sentences))))
        
        # Step 6: Chunk Size Constraints
        chunks = self.enforce_size_constraints(initial_chunks, embeddings, similarities)
        
        # Step 7: Topic Coherence Validation
        chunks = self.validate_topic_coherence(chunks, embeddings)
        
        # Step 8: Chunk Merge
        chunks = self.merge_coherent_chunks(chunks, embeddings)
        
        # Step 9: Evidence Quality Filter
        chunks = self.filter_evidence_quality(chunks, sentences)
        
        # Step 10: Final Output Formatting
        final_output = []
        for idx, chunk in enumerate(chunks):
            chunk_sentences = [sentences[i] for i in chunk]
            chunk_text = " ".join(chunk_sentences)
            final_output.append({
                "chunk_id": idx + 1,
                "sentence_count": len(chunk_sentences),
                "word_count": len(chunk_text.split()),
                "text": chunk_text,
                "sentences": chunk_sentences
            })
            
        return final_output

def run_test():
    """Unit test with sample Vietnamese article text."""
    test_text = (
        "Bộ Giáo dục công bố lịch thi tốt nghiệp THPT năm nay. "
        "Kỳ thi diễn ra vào cuối tháng 6 trên phạm vi cả nước. "
        "Có tổng cộng khoảng 1,1 triệu thí sinh đăng ký tham dự kỳ thi năm nay. "
        "Các hội đồng thi đang tích cực chuẩn bị cơ sở vật chất tốt nhất. "
        "Trong khi đó, thời tiết nắng nóng gay gắt kéo dài đang gây lo ngại lớn cho sức khỏe học sinh. "
        "Các chuyên gia y tế khuyên thí sinh cần uống đủ nước và ăn uống đầy đủ chất dinh dưỡng. "
        "Ngoài ra, các gia đình cần chuẩn bị sẵn sàng các phương án làm mát cho con em mình. "
        "Bộ Y tế cũng đã có công văn khẩn gửi các địa phương hỗ trợ y tế tại điểm thi. "
        "Nguồn: Báo Giáo dục và Thời đại."
    )
    print("\n--- Running Semantic Chunker Unit Test ---")
    chunker = SemanticChunker(model_name="BAAI/bge-m3")
    chunks = chunker.chunk_text(test_text)
    print(f"Generated {len(chunks)} chunks:")
    print(json.dumps(chunks, indent=2, ensure_ascii=False))

def main():
    parser = argparse.ArgumentParser(description="Semantic Chunking for Vietnamese Articles")
    parser.add_argument("--input", type=str, help="Path to input JSON dataset")
    parser.add_argument("--output", type=str, help="Path to save output JSON dataset")
    parser.add_argument("--test", action="store_true", help="Run self-test on sample text")
    parser.add_argument("--field", type=str, default="justification", help="Field inside JSON to chunk (justification/content/original_text)")
    args = parser.parse_args()

    if args.test or (not args.input and not args.output):
        run_test()
        return

    if not args.input or not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist.")
        return

    # Load input dataset
    with open(args.input, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if not isinstance(dataset, list):
        print("Error: Input JSON must be a list of articles.")
        return

    chunker = SemanticChunker(model_name="BAAI/bge-m3")
    
    print(f"Processing {len(dataset)} articles from '{args.input}'...")
    processed_dataset = []
    
    for idx, item in enumerate(dataset):
        # Determine content text
        text = item.get(args.field, "")
        if not text:
            # Fallback field
            text = item.get("original_text", "")
            
        print(f"[{idx+1}/{len(dataset)}] Chunking article ID: {item.get('id', 'N/A')[:8]}...")
        chunks = chunker.chunk_text(text)
        
        # Clone item and insert chunks
        chunked_item = item.copy()
        chunked_item["chunks"] = chunks
        processed_dataset.append(chunked_item)

    # Save output dataset
    output_path = args.output if args.output else "fact_checking_dataset_chunked.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed_dataset, f, indent=2, ensure_ascii=False)
        
    print(f"Successfully processed dataset and saved to '{output_path}'.")

if __name__ == "__main__":
    main()
