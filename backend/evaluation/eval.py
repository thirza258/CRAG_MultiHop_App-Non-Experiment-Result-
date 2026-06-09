import pandas as pd
from datasets import Dataset
from ragas.evaluation import evaluate
from ragas.metrics import (answer_relevancy, faithfulness)
import logging

logger = logging.getLogger(__name__)

def ragas_llm_as_a_judge_generation_evaluation(
    dataset:          Dataset,
    llm_judge,        
    judge_embeddings  
) -> pd.DataFrame:
    try:

        result = evaluate(
            dataset=dataset,
            metrics=[answer_relevancy, faithfulness],
            llm=llm_judge,
            embeddings=judge_embeddings
        )
      
        df_result = result.to_pandas()
        return df_result[[
            "faithfulness",
            "answer_relevancy"
        ]]
    except Exception as e:
        logger.error(f"Error in ragas_llm_as_a_judge_generation_evaluation: {e}")
        return pd.DataFrame([{
            "faithfulness":       None,
            "answer_relevancy":   None
        }])