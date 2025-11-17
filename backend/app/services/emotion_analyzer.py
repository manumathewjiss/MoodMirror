from transformers import pipeline
from typing import Dict, List
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class EmotionAnalyzer:
    
    def __init__(self, model_name: str = "j-hartmann/emotion-english-distilroberta-base"):
        self.model_name = model_name
        self.classifier = None
        self._load_model()
    
    def _load_model(self):
        try:
            logger.info(f"Loading emotion model: {self.model_name}")
            self.classifier = pipeline(
                "text-classification",
                model=self.model_name,
                top_k=None
            )
            logger.info("Emotion model loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load emotion model: {e}")
            raise
    
    def analyze_emotion(self, text: str) -> Dict:
        if not text or not text.strip():
            return {
                "error": "Empty text provided",
                "text": text
            }
        
        try:
            results = self.classifier(text)[0]
            
            sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
            primary = sorted_results[0]
            
            all_emotions = {item['label']: round(item['score'], 4) for item in results}
            
            return {
                "text": text,
                "primary_emotion": primary['label'],
                "confidence": round(primary['score'], 4),
                "all_emotions": all_emotions,
                "timestamp": datetime.utcnow().isoformat(),
                "model_used": self.model_name
            }
            
        except Exception as e:
            logger.error(f"Error analyzing emotion: {e}")
            return {
                "error": str(e),
                "text": text
            }
    
    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        results = []
        for text in texts:
            result = self.analyze_emotion(text)
            results.append(result)
        
        return results
    
    def get_emotion_summary(self, emotions_list: List[str]) -> Dict:
        from collections import Counter
        
        if not emotions_list:
            return {"error": "No emotions provided"}
        
        emotion_counts = Counter(emotions_list)
        total = len(emotions_list)
        
        summary = {
            "total_entries": total,
            "emotion_distribution": {
                emotion: {
                    "count": count,
                    "percentage": round((count / total) * 100, 2)
                }
                for emotion, count in emotion_counts.items()
            },
            "most_common_emotion": emotion_counts.most_common(1)[0][0],
            "unique_emotions": len(emotion_counts)
        }
        
        return summary


_analyzer_instance = None


def get_emotion_analyzer() -> EmotionAnalyzer:
    global _analyzer_instance
    
    if _analyzer_instance is None:
        _analyzer_instance = EmotionAnalyzer()
    
    return _analyzer_instance
