import torch
import numpy as np
from transformers import pipeline
from .config import RE_ID2LABEL, SPECIAL_TOKENS, DEVICE

# --- HÀM PHỤ TRỢ: GỘP BIO TAGS THÀNH ENTITY ---
def aggregate_entities(tokens, tags):
    """
    Input: 
      tokens = ['Tai', 'nạn', 'tại', 'Hà', 'Nội']
      tags   = ['O',   'O',   'O',   'B-LOC', 'I-LOC']
    Output:
      [{'word': 'Hà Nội', 'entity_group': 'LOC'}]
    """
    entities = []
    current_entity = None
    
    for token, tag in zip(tokens, tags):
        if tag == 'O':
            if current_entity:
                entities.append(current_entity)
                current_entity = None
            continue
            
        # Tách B-LOC thành prefix=B, label=LOC
        parts = tag.split('-')
        label = parts[1] if len(parts) > 1 else tag
        prefix = parts[0] if len(parts) > 1 else ''
        
        if prefix == 'B':
            if current_entity:
                entities.append(current_entity)
            current_entity = {"word": token, "entity_group": label}
        elif prefix == 'I':
            if current_entity and current_entity['entity_group'] == label:
                current_entity['word'] += " " + token
            else:
                # Trường hợp I- nằm lẻ loi (coi như bắt đầu mới)
                if current_entity:
                    entities.append(current_entity)
                current_entity = {"word": token, "entity_group": label}
        else:
            # Trường hợp nhãn không có B/I (ít gặp nhưng đề phòng)
            if current_entity:
                entities.append(current_entity)
            current_entity = {"word": token, "entity_group": tag}
             
    if current_entity:
        entities.append(current_entity)
    return entities

# --- CÁC CLASS WRAPPER ---
class BasePredictor:
    def __init__(self, model_type):
        self.model_type = model_type

class NERPredictor(BasePredictor):
    def __init__(self, model_type, model, tokenizer=None, feature_extractor=None, label_map=None):
        super().__init__(model_type)
        self.model = model
        
        if model_type == 'DL':
            self.pipe = pipeline("token-classification", model=model, tokenizer=tokenizer, 
                                 aggregation_strategy="simple", device=0 if torch.cuda.is_available() else -1)
        else:
            self.feature_extractor = feature_extractor
            self.label_map = label_map # {0: 'O', 1: 'B-LOC'}

    def predict(self, text):
        if self.model_type == 'DL':
            # HuggingFace pipeline đã tự gộp entity
            return self.pipe(text)
        
        else:
            # --- LOGIC CHO ML (SVM/LogReg/CRF) ---
            
            # 1. Lấy vector
            # Lưu ý: feature_extractor.vectorize_token_level dùng split() nên ta cũng dùng split() để lấy token gốc
            tokens = text.split()
            
            if hasattr(self.model, "predict_marginals") or "CRF" in str(type(self.model)):
                vectors = self.feature_extractor.extract_crf_features(text)
                # CRF predict trả về list of list labels luôn
                preds = self.model.predict(vectors)[0] 
            else:
                vectors = self.feature_extractor.vectorize_token_level(text)
                # SVM/LogReg predict trả về mảng số
                pred_ids = self.model.predict(vectors)
                
                # Map ID -> Label String
                preds = []
                for pid in pred_ids:
                    # Xử lý trường hợp label_map key là string hoặc int
                    label = self.label_map.get(pid) or self.label_map.get(str(pid)) or 'O'
                    preds.append(label)

            # 2. Gộp Tokens + Tags thành Entities
            # Cắt ngắn nếu số lượng token và pred không khớp (do tokenizer)
            min_len = min(len(tokens), len(preds))
            entities = aggregate_entities(tokens[:min_len], preds[:min_len])
            
            return entities

class REPredictor(BasePredictor):
    def __init__(self, model_type, model, tokenizer=None, feature_extractor=None, label_encoder=None):
        super().__init__(model_type)
        self.device = DEVICE
        self.model = model
        
        if model_type == 'DL':
            self.tokenizer = tokenizer
            # Thêm Special Tokens cho model DL (nếu chưa có)
            if self.tokenizer:
                num_added = self.tokenizer.add_special_tokens({'additional_special_tokens': SPECIAL_TOKENS})
                if num_added > 0:
                    self.model.resize_token_embeddings(len(self.tokenizer))
            self.model.to(self.device)
            self.model.eval()
        else:
            # ML Model cần feature extractor để vector hóa text (có chứa tags)
            self.feature_extractor = feature_extractor
            self.label_encoder = label_encoder

    def predict(self, text):
        """
        Input: Text đã chèn thẻ Typed Markers. VD: "Tại <S:LOC> Hà Nội </S:LOC>..."
        """
        if self.model_type == 'DL':
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=True).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
            pred_id = logits.argmax().item()
            # Map ID -> Label từ Config
            return RE_ID2LABEL.get(pred_id, "NO_RELATION")
        
        else:
            # ML: Vector hóa text (đã có tags) -> Predict
            if not self.feature_extractor:
                return "ERROR: Missing Feature Extractor"
                
            vec = self.feature_extractor.vectorize_sentence_level(text)
            pred_id = self.model.predict([vec])[0]
            
            # Map ID -> Label
            # Trường hợp dùng LabelEncoder của Sklearn
            if self.label_encoder and hasattr(self.label_encoder, 'inverse_transform'):
                return self.label_encoder.inverse_transform([pred_id])[0]
            
            # Trường hợp model trả về thẳng ID khớp với RE_ID2LABEL
            return RE_ID2LABEL.get(int(pred_id), "NO_RELATION")