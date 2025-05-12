import torch
import logging
from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)

class WordDetect:
    def __init__(self, model: SentenceTransformer = None):
        self.model = model
        self._initialize_model()

    def _initialize_model(self):
        if not self.model:
            model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
            self.model = SentenceTransformer(model_name)
            logger.info(f'Loaded model: {model_name}')

    def _detect_nearest_word(self, query_word: str, word_list: list[str]):
        query_embedding = self.model.encode(query_word, convert_to_tensor=True)
        list_embedding = self.model.encode(word_list, convert_to_tensor=True)
        cosine_scores = util.pytorch_cos_sim(query_embedding, list_embedding)
        max_score_index = torch.argmax(cosine_scores)
        return word_list[max_score_index]

    def detect_nearest_word(self, query_word: str, word_list: list[str]):
        return self._detect_nearest_word(query_word, word_list)

