import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
sbert = AutoModel.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
sbert.eval()


@torch.no_grad()
def sbert_predict(encoded_input):
    model_output = sbert(**encoded_input)
    token_embeddings = model_output[0]
    input_mask_expanded = encoded_input['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.columns].values


class AddTargetEncoding(BaseEstimator, TransformerMixin):
    def __init__(self, ):
        pass
    
    def encode_rating(self, rating): 
        return np.where(rating <= 2, 1, 0)  # 1 - негативный, 0 - (позитивный или нейтральный)
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return self.encode_rating(X.flatten())


class PreTrainSbertProcessing(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # X должен быть размера (*, 1)
        bert_brobs = []
        for string in tqdm(X):
            # string - список из одного элемента
            encoded_input = tokenizer(string[0], padding=True, truncation=True, max_length=24, return_tensors='pt')
            features = sbert_predict(encoded_input)
            bert_brobs.append(features.numpy()[0])
        return np.array(bert_brobs)