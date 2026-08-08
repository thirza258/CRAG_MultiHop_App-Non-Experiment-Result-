from typing import List, Dict
import numpy as np
from nltk.tokenize import sent_tokenize
import logging
import torch
from common.memory import _local_model_path
from common.nltk_setup import ensure_nltk_data
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from torch import Tensor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ensure_nltk_data()

class CRAGEvaluator:
    def __init__(self, config):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        MODEL_NAME = "intfloat/multilingual-e5-small"
        model_path = _local_model_path(MODEL_NAME)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(self.device)
        self.model.eval()
        
        self.upper_threshold = config.get("upper_threshold", 0.7)
        self.lower_threshold = config.get("lower_threshold", 0.3)
        self.strip_threshold = config.get("strip_threshold", 0.5)
        
    def average_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def _encode(self, texts: List[str], prefix: str) -> Tensor:
        input_texts = [f"{prefix}: {t}" for t in texts]
        batch = self.tokenizer(
            input_texts, max_length=512,
            padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**batch)
        embeddings = self.average_pool(outputs.last_hidden_state, batch["attention_mask"])
        return F.normalize(embeddings, p=2, dim=1)

    def _encode_query(self, query: str) -> Tensor:
        return self._encode([query], prefix="query")

    def _encode_passages(self, docs: List[str]) -> Tensor:
        return self._encode(docs, prefix="passage")

    def _similarity(self, query_emb: Tensor, passage_embs: Tensor) -> np.ndarray:
        scores = (query_emb @ passage_embs.T) * 100  
        return scores.squeeze().cpu().numpy().flatten()

    def _normalize(self, raw_scores: np.ndarray) -> np.ndarray:
        """Map [-100, +100] → [0, 1]"""
        return (raw_scores + 100) / 200.0

    def _score_texts(self, query: str, texts: List[str]) -> np.ndarray:
        """Returns normalized [0, 1] scores for a list of texts."""
        if not texts:
            return np.array([])
        query_emb    = self._encode_query(query)
        passage_embs = self._encode_passages(texts)
        raw          = self._similarity(query_emb, passage_embs)
        # ensure array even for single text
        if raw.ndim == 0:
            raw = raw.reshape(1)
        return self._normalize(raw)

    def score_docs(self, query: str, docs: List[str]) -> List[float]:
        """Public method — returns normalized [0, 1] scores."""
        return self._score_texts(query, docs).tolist()

    def evaluate(
        self, query: str, docs: List[str], metas: List[Dict] = None
    ) -> tuple[str, List[str], List[Dict], float]:

        if not docs:
            return "incorrect", [], [], float("-inf")

        if metas is None:
            metas = [{} for _ in docs]

        scores = self._score_texts(query, docs)

        doc_labels = []
        for score in scores:
            if score > self.upper_threshold:
                doc_labels.append("correct")
            elif score > self.lower_threshold:
                doc_labels.append("ambiguous")
            else:
                doc_labels.append("incorrect")

        best_score = float(scores.max())

        filtered_docs  = []
        filtered_metas = []
        final_labels   = [] 

        for doc, meta, label in zip(docs, metas, doc_labels):

            if label == "correct":
                filtered_docs.append(doc)
                filtered_metas.append(meta)
                final_labels.append("correct")

            elif label == "ambiguous":
                refined = self.knowledge_refinement(query, doc)
                if not refined:
                    filtered_docs.append(doc)
                    filtered_metas.append(meta)
                    final_labels.append("ambiguous")
                    continue
                
                refined_score = float(self._score_texts(query, [refined])[0])
                logger.info(f"[evaluate] ambiguous chunk refined: score {label} → {refined_score:.3f}")

                if refined_score > self.upper_threshold:
                    logger.info(f"[evaluate] refined chunk promoted to correct")
                    filtered_docs.append(refined)
                    filtered_metas.append(meta)
                    final_labels.append("correct") 

                elif refined_score > self.lower_threshold:
                    logger.info(f"[evaluate] refined chunk remains ambiguous")
                    filtered_docs.append(refined)
                    filtered_metas.append(meta)
                    final_labels.append("ambiguous")  

                else:
                    logger.info(f"[evaluate] refined chunk degraded to incorrect → discarded")
                    final_labels.append("incorrect")

            else:
                final_labels.append("incorrect")

        correct_count = final_labels.count("correct")
        ambiguous_count = final_labels.count("ambiguous")

        if all(label == "incorrect" for label in final_labels):
            decision = "incorrect"
        elif correct_count >= ambiguous_count:
            decision = "correct"
        elif correct_count < ambiguous_count:
            decision = "ambiguous"
        else:
            decision = "incorrect"  

        if not filtered_docs:
            decision = "incorrect"
            logger.info("[evaluate] all chunks discarded after refinement → escalate to external")

        return decision, filtered_docs, filtered_metas, best_score

    def _segment_doc(self, doc: str) -> List[str]:
        """
        Adaptive segmentation per CRAG Section 3.2:
        - Short (1-2 sentences) → single strip
        - Medium (3-6 sentences) → 2-sentence chunks
        - Long (>6 sentences)   → 3-sentence chunks
        """
        sentences = sent_tokenize(doc)
        n = len(sentences)

        if n <= 2:
            return [doc]
        
        chunk_size = 2 if n <= 6 else 3
        strips = []
        for i in range(0, n, chunk_size):
            strips.append(" ".join(sentences[i:i + chunk_size]))
        return strips

    def knowledge_refinement(self, query: str, doc: str) -> str:
        """
        Refine a single doc/chunk — CRAG paper Section 3.2.
        Decompose → score each strip → recompose relevant strips in order.
        Takes single doc, returns refined string (not concat of all docs).
        """
        strips = self._segment_doc(doc)

        if not strips:
            return ""

        if len(strips) == 1:
            return doc
        
        scores = self._score_texts(query, strips)
        
        if scores.size == 0:
            return doc


        relevant_with_idx = [
            (idx, strip)
            for idx, (strip, score) in enumerate(zip(strips, scores))
            if score > self.strip_threshold
        ]

        if not relevant_with_idx:
            top_idx = np.argsort(scores)[-3:]
            relevant_with_idx = [(i, strips[i]) for i in sorted(top_idx)]
            
        if not relevant_with_idx:
            return doc

        return " ".join(strip for _, strip in relevant_with_idx)