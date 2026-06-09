import logging
from typing import Any, List, Dict
from .corrective_evaluator import CRAGEvaluator
from .external_search import ExternalSearcher
from .query_expander import QueryExpander

logger = logging.getLogger(__name__)

class CorrectiveRAG:
    def __init__(self, retriever, config: Dict[str, Any]):
        self.retriever      = retriever
        self.evaluator      = CRAGEvaluator(config)
        self.external       = ExternalSearcher(config)
        self.query_expander = QueryExpander(config)
        # self.seen_urls: set = set()
        self.top_k          = getattr(retriever, 'top_k', 5)

    # def reset_seen_urls(self) -> None:
    #     self.seen_urls.clear()

    def _build_filter(self, seen_urls: set) -> Dict | None:
        return {"url": {"$nin": list(seen_urls)}} if seen_urls else None

    def _track_urls(self, metas: List[Dict], seen_urls: set) -> None:
        if not metas:
            return
        for meta in metas:
            if meta and isinstance(meta, dict):
                if url := meta.get("url"):
                    seen_urls.add(url)

    def retrieve_with_decision(
        self, query: str, keyword: str = None, where_filter: Dict = None, seen_urls: set = None
    ) -> tuple[List[str], List[Dict], str]:
        
        self.emitter.emit("corrective_pipeline", f"Starting retrieval for query: '{query[:80]}'")
        
        seen_urls = seen_urls if seen_urls is not None else set()

        keyword      = keyword or self.query_expander.to_keyword(query)
        where_filter = where_filter or self._build_filter(seen_urls)

        docs, metas = self.retriever.retrieve(
            query=query, keyword=keyword, where_filter=where_filter
        )

        if not docs:
            logger.info("[retrieve_with_decision] local DB empty — fetching external")
            return self._fetch_fully_external(query, local_docs=[], local_score=float("-inf"))

        self._track_urls(metas, seen_urls)
        decision, filtered, filtered_metas, local_score = self.evaluator.evaluate(query, docs, metas)
        
        self.emitter.emit("corrective_pipeline", f"Evaluation decision: {decision}, local_score: {local_score:.3f}")
        
        logger.info(f"[retrieve_with_decision] first attempt: decision={decision}, best_score={local_score:.3f}")

        if decision == "correct":
            self.emitter.emit("corrective_pipeline", f"Correct decision — returning {len(filtered)} chunks")
            return filtered, filtered_metas, "correct"

        elif decision == "ambiguous":
            self.emitter.emit("corrective_pipeline", "Ambiguous decision — attempting resolution")
            logger.info("[retrieve_with_decision] ambiguous — attempting to resolve")
            docs2, metas2, decision2 = self.handle_ambiguous(query, metas, seen_urls)

            if decision2 == "correct":
                return docs2, metas2, "ambiguous_resolved"

            logger.info("[retrieve_with_decision] ambiguous unresolved — fetching external")
            ext_docs, ext_metas = self._safe_external_search(query)
            self.emitter.emit("corrective_pipeline", "Fetching external results")
            refined_ext_docs  = []
            refined_ext_metas = []

            for doc, meta in zip(ext_docs, ext_metas):
                refined = self.evaluator.knowledge_refinement(query, doc)
                if refined:
                    refined_ext_docs.append(refined)
                    refined_ext_metas.append(meta)
                else:
                    refined_ext_docs.append(doc)
                    refined_ext_metas.append(meta)

            if refined_ext_docs:
                docs_to_score  = refined_ext_docs
                metas_to_use   = refined_ext_metas
            else:
                docs_to_score  = ext_docs
                metas_to_use   = ext_metas

            ext_scores = self.evaluator.score_docs(query, docs_to_score)
            best_ext   = max(ext_scores, default=float("-inf"))

            logger.info(f"[retrieve_with_decision] local_score={local_score:.3f}, best_ext={best_ext:.3f}")

            if docs_to_score and best_ext > local_score:
                combined   = filtered + docs2 + docs_to_score
                combined_m = filtered_metas + (metas2 or []) + metas_to_use
                logger.info("[retrieve_with_decision] external beats local — combining all")
            else:
                combined   = filtered + docs2
                combined_m = filtered_metas + (metas2 or [])
                logger.info("[retrieve_with_decision] local beats external — keeping local only")

            return combined, combined_m, "ambiguous_external"

        else:
            self.emitter.emit("corrective_pipeline", "Incorrect decision — fetching external")
            return self._fetch_fully_external(query, local_docs=docs, local_score=local_score)

    def handle_ambiguous(
        self, query: str, previous_metas: List[Dict], seen_urls: set = None
    ) -> tuple[List[str], List[Dict], str]:

        seen_urls = seen_urls if seen_urls is not None else seen_urls
        self._track_urls(previous_metas, seen_urls)
        where_filter = self._build_filter(seen_urls)

        reformulated = self.query_expander.reformulate(query)
        keyword      = self.query_expander.to_keyword(reformulated)
        n_ways       = self.query_expander.expand_multiple(query, n=3)

        logger.info(f"[handle_ambiguous] reformulated query: '{reformulated}'")
        logger.info(f"[handle_ambiguous] excluded URLs: {seen_urls}")

        docs, metas = self.retriever.retrieve(
            query        = reformulated + " " + keyword,
            keyword      = keyword,
            where_filter = where_filter
        )

        if not docs:
            logger.info("[handle_ambiguous] no docs after filter — escalating to external")
            return [], [], "incorrect"

        decision, filtered, filtered_metas, best_score = self.evaluator.evaluate(query, docs, metas)
        logger.info(f"[handle_ambiguous] decision: {decision}, best_score: {best_score:.3f}")

        if decision == "correct":
            return filtered, filtered_metas, "correct"

        elif decision == "ambiguous":
            snapshot = set(seen_urls)
            for alt_query in n_ways:
                iter_seen = set(snapshot)
                where_filter = self._build_filter(iter_seen)
                logger.info(f"[handle_ambiguous] trying alternative query: '{alt_query}'")
                alt_docs, alt_metas = self.retriever.retrieve(
                    query        = alt_query,
                    keyword      = self.query_expander.to_keyword(alt_query),
                    where_filter = where_filter
                )
                if not alt_docs:
                    continue

                alt_decision, alt_filtered, alt_filtered_metas, alt_score = self.evaluator.evaluate(query, alt_docs, alt_metas)

                if alt_decision == "correct":
                    return alt_filtered, alt_filtered_metas, "correct"

        self._track_urls(metas, seen_urls)
        logger.info(f"[handle_ambiguous] decision after retry: {decision}")
        return filtered, filtered_metas, decision

    def _safe_external_search(self, query: str) -> tuple[List[str], List[Dict]]:
        """
        Runs Wikipedia and News searches with individual error isolation.
        Always returns whatever was successfully fetched — never raises.
        """
        wiki_chunks, wiki_metas = [], []
        news_chunks, news_metas = [], []

        try:
            wiki_chunks, wiki_metas = self.external.search_wikipedia(query)
            logger.info(f"[_safe_external_search] Wikipedia returned {len(wiki_chunks)} chunks")
        except Exception as e:
            logger.warning(f"[_safe_external_search] Wikipedia failed entirely: {e}")

        try:
            news_chunks, news_metas = self.external.search_news(query)
            logger.info(f"[_safe_external_search] News returned {len(news_chunks)} chunks")
        except Exception as e:
            logger.warning(f"[_safe_external_search] News failed entirely: {e}")

        return wiki_chunks + news_chunks, wiki_metas + news_metas

    def _fetch_fully_external(
        self, query: str,
        local_docs: List[str] = None,
        local_score: float = float("-inf")
    ) -> tuple[List[str], List[Dict], str]:

        logger.info("[_fetch_fully_external] fetching external sources")
        ext_docs, ext_metas = self._safe_external_search(query)

        if not ext_docs:
            logger.warning("[_fetch_fully_external] external returned nothing — falling back to local")
            if local_docs:
                return local_docs, [], "incorrect_local_fallback"
            return [], [], "incorrect_empty"

        refined_ext_docs, refined_ext_metas = [], []
        for doc, meta in zip(ext_docs, ext_metas):
            try:
                refined = self.evaluator.knowledge_refinement(query, doc)
                if refined:
                    refined_ext_docs.append(refined)
                    refined_ext_metas.append(meta)
            except Exception as e:
                logger.warning(f"[_fetch_fully_external] refinement failed for chunk, keeping original: {e}")
                refined_ext_docs.append(doc)
                refined_ext_metas.append(meta)

        docs_to_score  = refined_ext_docs  if refined_ext_docs  else ext_docs
        metas_to_use   = refined_ext_metas if refined_ext_metas else ext_metas

        try:
            ext_scores = self.evaluator.score_docs(query, docs_to_score)
            best_ext   = max(ext_scores, default=float("-inf"))
        except Exception as e:
            logger.warning(f"[_fetch_fully_external] scoring failed, skipping score comparison: {e}")
            best_ext = float("-inf")

        logger.info(f"[_fetch_fully_external] local_score={local_score:.3f}, best_ext={best_ext:.3f}")

        if local_docs and local_score >= best_ext:
            logger.info("[_fetch_fully_external] local beats external — using local chunks")
            return local_docs, [], "incorrect_local_fallback"

        return docs_to_score, metas_to_use, "incorrect"

    def retrieve(
        self, query: str, keyword: str = None, where_filter: Dict = None, seen_urls: set = None
    ) -> tuple[List[str], List[Dict]]:
        results, metas, decision = self.retrieve_with_decision(query, keyword, where_filter, seen_urls)
        logger.info(f"[CRAGWrapper.retrieve] final decision: {decision}")
        self.emitter.emit("corrective_pipeline", f"Final results retrieved: {len(results)}")
        return results, metas
    
    def set_emitter(self, emitter):
        self.emitter = emitter
        self.retriever.set_emitter(emitter)