import numpy as np
import random
from sentence_transformers import SentenceTransformer

def get_sentence_embedding(sentence):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(sentence)
    return embeddings

def get_adj_matrix(graph_type, n):
    adj_matrix = np.zeros((n, n), dtype=int)
    if "tree" in graph_type:
        for i in range(n):
            left_child = 2 * i + 1
            right_child = 2 * i + 2
            if left_child < n:
                adj_matrix[i][left_child] = 1
                adj_matrix[left_child][i] = 1
            if right_child < n:
                adj_matrix[i][right_child] = 1
                adj_matrix[right_child][i] = 1
    if "chain" in graph_type:
        for i in range(n - 1):
            adj_matrix[i, i + 1] = 1
            adj_matrix[i + 1, i] = 1
    if "star" in graph_type:
        for i in range(1, n):
            adj_matrix[0][i] = 1
            adj_matrix[i][0] = 1
        for i in range(1, n - 1):
            adj_matrix[i][i + 1] = 1
            adj_matrix[i + 1][i] = 1
        adj_matrix[1][n - 1] = 1
        adj_matrix[n - 1][1] = 1
        
    return adj_matrix


if __name__ == "__main__": 
    data = get_adj_matrix("star", 8, 4)
    print(data)