from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from src.Agent import OpenAIEvalLLM  
from sklearn.feature_extraction.text import HashingVectorizer
import numpy as np

vectorizer = HashingVectorizer(n_features=512, alternate_sign=False)

def embed(text: str):
    return vectorizer.transform([text]).toarray()[0]

def evaluation(eval_dict):
    model = OpenAIEvalLLM(model_name="gpt-4o-mini")

    faith = FaithfulnessMetric(model=model)
    relevancy = AnswerRelevancyMetric(model=model)

    test_case = LLMTestCase(
        input=eval_dict["question"],
        actual_output=eval_dict["answer"],
        retrieval_context=[eval_dict["context"]]
    )

    faith_score = faith.measure(test_case)
    relevancy_score = relevancy.measure(test_case)

    return {
        "faithfulness": faith_score,
        "answer_relevancy": relevancy_score
    }

import numpy as np

def cosine(a, b):
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def context_precision(answer, contexts):
    ans_emb = embed(answer)
    ctx_embs = [embed(c) for c in contexts]
    return max(cosine(ans_emb, c) for c in ctx_embs)

def context_recall(answer, contexts):
    ans_emb = embed(answer)
    ctx_embs = [embed(c) for c in contexts]
    return float(np.mean([cosine(ans_emb, c) for c in ctx_embs]))
