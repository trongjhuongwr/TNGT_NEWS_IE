import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from .config import MODEL_PATHS, DEVICE, SPECIAL_TOKENS

class PhoBERTFeatureExtractor:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            print(f"--- [INFO] Loading Vectorizer Base ({MODEL_PATHS['VECTORIZER_BASE']})...")
            cls._instance = super(PhoBERTFeatureExtractor, cls).__new__(cls)
            
            # Load Tokenizer
            cls._instance.tokenizer = AutoTokenizer.from_pretrained(
                MODEL_PATHS["VECTORIZER_BASE"], 
                use_fast=False 
            )
            
            # --- QUAN TRỌNG: THÊM SPECIAL TOKENS CHO VECTORIZER ---
            # Để khi vector hóa cho ML, model hiểu các thẻ <S:EVENT>...
            if SPECIAL_TOKENS:
                cls._instance.tokenizer.add_special_tokens({'additional_special_tokens': SPECIAL_TOKENS})
            
            # Load Model
            cls._instance.model = AutoModel.from_pretrained(MODEL_PATHS["VECTORIZER_BASE"])
            cls._instance.model.resize_token_embeddings(len(cls._instance.tokenizer))
            cls._instance.model.to(DEVICE)
            cls._instance.model.eval()
            
        return cls._instance

    def vectorize_token_level(self, text):
        # ... (Giữ nguyên logic cũ của bạn cho NER nếu cần) ...
        # (Lược bỏ để gọn, bạn giữ lại code cũ của hàm này)
        tkns = text.split() 
        inputs = self.tokenizer(tkns, is_split_into_words=True, return_tensors="pt", padding=True, truncation=True, max_length=256).to(DEVICE)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[0].cpu().numpy() # Simplified for brevity, use your full logic

    def vectorize_sentence_level(self, text):
        """
        Vector hóa cả câu cho RE (ML Models).
        Logic: Mean Pooling (giống notebook)
        """
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256).to(DEVICE)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # --- Mean Pooling (Khớp logic notebook) ---
        # Lấy vector đại diện cho cả câu dựa trên attention mask
        mask = inputs['attention_mask'].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
        sum_emb = torch.sum(outputs.last_hidden_state * mask, 1)
        sum_mask = torch.clamp(mask.sum(1), min=1e-9)
        mean_emb = sum_emb / sum_mask
        
        return mean_emb.cpu().numpy().flatten()
    
    def extract_crf_features(self, text):
        """Giữ nguyên cho CRF"""
        # ... (Giữ nguyên logic cũ) ...
        return []