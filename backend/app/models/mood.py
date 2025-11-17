from pydantic import BaseModel, Field
from typing import Dict, Optional, List
from datetime import datetime


class MoodEntryCreate(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000, description="User's mood description")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "I'm feeling overwhelmed with all my assignments today"
            }
        }


class EmotionAnalysis(BaseModel):
    primary_emotion: str
    confidence: float
    all_emotions: Dict[str, float]
    model_used: str


class MoodEntry(BaseModel):
    id: str = Field(alias="_id")
    user_id: str
    text: str
    emotion_analysis: EmotionAnalysis
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "_id": "507f1f77bcf86cd799439011",
                "user_id": "507f191e810c19729de860ea",
                "text": "I'm feeling overwhelmed with all my assignments today",
                "emotion_analysis": {
                    "primary_emotion": "sadness",
                    "confidence": 0.89,
                    "all_emotions": {
                        "sadness": 0.89,
                        "fear": 0.07,
                        "anger": 0.03,
                        "neutral": 0.01
                    },
                    "model_used": "j-hartmann/emotion-english-distilroberta-base"
                },
                "created_at": "2025-11-12T10:30:00"
            }
        }


class MoodEntryResponse(BaseModel):
    id: str
    text: str
    emotion_analysis: EmotionAnalysis
    created_at: datetime


class MoodHistoryResponse(BaseModel):
    total_entries: int
    entries: List[MoodEntryResponse]
    date_range: Dict[str, datetime]


class EmotionTrendsResponse(BaseModel):
    total_entries: int
    date_range: Dict[str, datetime]
    emotion_distribution: Dict[str, Dict[str, float]]
    most_common_emotion: str
    daily_emotions: List[Dict]
    weekly_summary: Optional[Dict] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "total_entries": 14,
                "date_range": {
                    "start": "2025-11-01T00:00:00",
                    "end": "2025-11-12T23:59:59"
                },
                "emotion_distribution": {
                    "joy": {"count": 5, "percentage": 35.71},
                    "sadness": {"count": 4, "percentage": 28.57},
                    "neutral": {"count": 3, "percentage": 21.43},
                    "anger": {"count": 2, "percentage": 14.29}
                },
                "most_common_emotion": "joy",
                "daily_emotions": [
                    {
                        "date": "2025-11-12",
                        "emotion": "sadness",
                        "confidence": 0.85
                    }
                ]
            }
        }
