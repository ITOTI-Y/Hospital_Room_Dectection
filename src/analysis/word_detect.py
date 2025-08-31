import torch
import logging
from sentence_transformers import SentenceTransformer, util
from src.config import NetworkConfig

logger = logging.getLogger(__name__)

class WordDetect:
    def __init__(self, model: SentenceTransformer = None, config: NetworkConfig = None):
        self.model = model
        self.config = config
        self._initialize_model()

    def _initialize_model(self):
        if not self.model:
            model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
            self.model = SentenceTransformer(model_name)
            logger.info(f'Loaded model: {model_name}')

    def _detect_nearest_word(self, query_word: str, word_list: list[str] = None):
        if word_list is None:
            word_list = self.config.ALL_TYPES
        query_embedding = self.model.encode(query_word, convert_to_tensor=True)
        list_embedding = self.model.encode(word_list, convert_to_tensor=True)
        cosine_scores = util.pytorch_cos_sim(query_embedding, list_embedding)
        max_score_index = torch.argmax(cosine_scores)
        return word_list[max_score_index]

    def detect_nearest_word(self, query_word: str | list[str], word_list: list[str] | None = None):
        if isinstance(query_word, str):
            return self._detect_nearest_word(query_word, word_list)
        elif isinstance(query_word, list):
            return [self._detect_nearest_word(word, word_list) for word in query_word]
        else:
            raise ValueError(f"Invalid query_word type: {type(query_word)}")

