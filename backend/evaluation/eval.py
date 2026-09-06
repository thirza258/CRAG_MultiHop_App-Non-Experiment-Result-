import pandas as pd
from datasets import Dataset
from ragas.evaluation import evaluate
from ragas.metrics import (answer_relevancy, faithfulness)
import logging

logger = logging.getLogger(__name__)

#: The metrics this judge can compute, by the name the pipeline config uses.
#: Each one is a separate LLM-judge pass, so switching one off is a real saving
#: rather than just hiding a number.
AVAILABLE_METRICS = {
    "answer_relevancy": answer_relevancy,
    "faithfulness": faithfulness,
}


def ragas_llm_as_a_judge_generation_evaluation(
    dataset:          Dataset,
    llm_judge,        
    judge_embeddings,
    metrics:          list = None,
) -> pd.DataFrame:
    """Score a generated answer with the RAGAS judge.

    ``metrics`` names which of :data:`AVAILABLE_METRICS` to compute; omitting it
    runs both, which is what every caller did before they were selectable. A
    metric that was not requested comes back as None rather than missing, so the
    caller's column lookups keep working either way.
    """
    requested = [
        name for name in (metrics if metrics is not None else AVAILABLE_METRICS)
        if name in AVAILABLE_METRICS
    ]

    empty = pd.DataFrame([{name: None for name in AVAILABLE_METRICS}])

    if not requested:
        # Every metric switched off. Not an error — the caller asked for no
        # scores, and returning the empty frame keeps one code path.
        logger.info("No evaluation metrics requested — skipping the judge entirely")
        return empty

    try:

        result = evaluate(
            dataset=dataset,
            metrics=[AVAILABLE_METRICS[name] for name in requested],
            llm=llm_judge,
            embeddings=judge_embeddings
        )
      
        df_result = result.to_pandas()

        # A metric that was not requested has no column. Fill it in as None so
        # the returned frame always has the same shape.
        for name in AVAILABLE_METRICS:
            if name not in df_result.columns:
                df_result[name] = None

        return df_result[list(AVAILABLE_METRICS)]
    except Exception as e:
        logger.error(f"Error in ragas_llm_as_a_judge_generation_evaluation: {e}")
        return empty
