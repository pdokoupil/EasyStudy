import numpy as np

from plugins.fastcompare.algo.algorithm_base import EvaluationMetricBase

class nDCG(EvaluationMetricBase):

    def __init__(self, *args, **kwargs):
        pass

    def evaluate(self, shown_items, selected_items):
        relevance_scores = [1 if item in selected_items else 0 for item in shown_items]

        # DCG
        dcg = relevance_scores[0]
        for i in range(1, len(relevance_scores)):
            dcg += relevance_scores[i] / np.log2(i + 1)
        
        # Ideal DCG
        ideal_relevance_scores = sorted(relevance_scores, reverse=True)  # Sort in descending order
        ideal_dcg = ideal_relevance_scores[0]
        for i in range(1, len(ideal_relevance_scores)):
            ideal_dcg += ideal_relevance_scores[i] / np.log2(i + 1)
        
        if ideal_dcg == 0:
            return 0
        
        return dcg / ideal_dcg
    
    @staticmethod
    def name():
        return "NDCG@K"
    
class Precision(EvaluationMetricBase):

    def __init__(self, *args, **kwargs):
        pass

    def evaluate(self, shown_items, selected_items):
        if len(shown_items) == 0:
            return 0
        return len(selected_items) / len(shown_items)
    
    @staticmethod
    def name():
        return "Precision@K"
    
class Count(EvaluationMetricBase):

    def __init__(self, *args, **kwargs):
        pass

    def evaluate(self, shown_items, selected_items):
        return len(selected_items)
    
    @staticmethod
    def name():
        return "Selection Count"

class ILD(EvaluationMetricBase):
    def __init__(self, rating_matrix, similarity_matrix):
        self.rating_matrix = rating_matrix
        self.similarity_matrix = similarity_matrix

    def evaluate(self, shown_items, selected_items):
        distance_matrix = 1.0 - self.similarity_matrix
        n = len(shown_items)
        div = 0
        for i in range(n):
            for j in range(i):
                div += distance_matrix[shown_items[i], shown_items[j]]
        return div / ((n - 1) * n / 2)

    @staticmethod
    def name():
        return "ILD"
