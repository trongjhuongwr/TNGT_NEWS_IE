from .preprocessing import sliding_window_extract, clean_text_basic

class TNGTPipeline:
    def __init__(self, ner_model, re_model):
        self.ner_predictor = ner_model
        self.re_predictor = re_model

        # Schema (Giữ nguyên)
        self.valid_pairs = {
            ('EVENT', 'LOC'), ('EVENT', 'TIME'), ('EVENT', 'CAUSE'),
            ('EVENT', 'VEH'), ('EVENT', 'CONSEQUENCE'),
            ('PER_VICTIM', 'VEH'), ('PER_DRIVER', 'VEH'), ('VEH', 'VEH')
        }

    def _generate_pairs(self, entities, text):
        candidates = []
        for i, e1 in enumerate(entities):
            for j, e2 in enumerate(entities):
                if i == j: continue
                l1 = (e1.get('entity_group') or e1.get('labels') or e1.get('entity', '')).replace("B-", "").replace("I-", "")
                l2 = (e2.get('entity_group') or e2.get('labels') or e2.get('entity', '')).replace("B-", "").replace("I-", "")
                if (l1, l2) in self.valid_pairs:
                    candidates.append({"source": e1, "target": e2, "text": text})
        return candidates

    def run(self, raw_text):
        cleaned_text = clean_text_basic(raw_text)
        windows = sliding_window_extract(cleaned_text, window_size=3, step_size=2)
        print(f"-> Đã chia văn bản thành {len(windows)} cửa sổ xử lý.")

        all_entities = []
        all_relations = []

        for idx, chunk in enumerate(windows):
            entities = self.ner_predictor.predict(chunk)
            for e in entities:               
                if 'word' in e:
                    e['word'] = e['word'].replace("@@", "")
                    
            pairs = self._generate_pairs(entities, chunk)
            for p in pairs:
                label = self.re_predictor.predict(chunk)
                if label != 'O' and label != 0 and label != 'UNKNOWN':
                    all_relations.append({
                        "source": p['source'].get('word'),
                        "target": p['target'].get('word'),
                        "relation": label,
                        "window_id": idx
                    })
            for e in entities:
                all_entities.append({
                    "text": e.get('word'), 
                    "label": (e.get('entity_group') or e.get('labels') or e.get('entity', '')).replace("B-", "").replace("I-", "")
                })

        return self._post_processing(all_entities, all_relations)

    def _post_processing(self, entities, relations):
        unique_entities = {}
        for e in entities:
            key = (e['text'], e['label'])
            unique_entities[key] = unique_entities.get(key, 0) + 1
        final_entities = [{"text": k[0], "label": k[1], "count": v} for k, v in unique_entities.items()]

        unique_relations = set()
        final_relations = []
        for r in relations:
            key = (r['source'], r['relation'], r['target'])
            if key not in unique_relations:
                unique_relations.add(key)
                final_relations.append({"subject": r['source'], "relation": r['relation'], "object": r['target']})
                
        return {"entities": final_entities, "relations": final_relations}