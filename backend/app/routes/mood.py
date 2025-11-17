from fastapi import APIRouter, HTTPException, Depends, status
from typing import List
from datetime import datetime, timedelta
from bson import ObjectId

from app.models.mood import (
    MoodEntryCreate,
    MoodEntry,
    MoodEntryResponse,
    MoodHistoryResponse,
    EmotionTrendsResponse,
    EmotionAnalysis
)
from app.services.emotion_analyzer import get_emotion_analyzer
from app.config.database import get_moods_collection

router = APIRouter()

MOCK_USER_ID = "test_user_123"


@router.post("/submit", response_model=MoodEntryResponse, status_code=status.HTTP_201_CREATED)
async def submit_mood(
    mood_data: MoodEntryCreate,
    moods_collection=Depends(get_moods_collection)
):
    try:
        analyzer = get_emotion_analyzer()
        analysis_result = analyzer.analyze_emotion(mood_data.text)
        
        if "error" in analysis_result:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Error analyzing emotion: {analysis_result['error']}"
            )
        
        emotion_analysis = EmotionAnalysis(
            primary_emotion=analysis_result['primary_emotion'],
            confidence=analysis_result['confidence'],
            all_emotions=analysis_result['all_emotions'],
            model_used=analysis_result['model_used']
        )
        
        mood_entry = {
            "user_id": MOCK_USER_ID,
            "text": mood_data.text,
            "emotion_analysis": emotion_analysis.model_dump(),
            "created_at": datetime.utcnow()
        }
        
        result = await moods_collection.insert_one(mood_entry)
        
        return MoodEntryResponse(
            id=str(result.inserted_id),
            text=mood_data.text,
            emotion_analysis=emotion_analysis,
            created_at=mood_entry['created_at']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit mood entry: {str(e)}"
        )


@router.get("/history", response_model=MoodHistoryResponse)
async def get_mood_history(
    limit: int = 30,
    moods_collection=Depends(get_moods_collection)
):
    try:
        cursor = moods_collection.find(
            {"user_id": MOCK_USER_ID}
        ).sort("created_at", -1).limit(limit)
        
        entries = await cursor.to_list(length=limit)
        
        mood_entries = []
        for entry in entries:
            mood_entries.append(
                MoodEntryResponse(
                    id=str(entry['_id']),
                    text=entry['text'],
                    emotion_analysis=EmotionAnalysis(**entry['emotion_analysis']),
                    created_at=entry['created_at']
                )
            )
        
        date_range = {}
        if mood_entries:
            date_range = {
                "start": mood_entries[-1].created_at,
                "end": mood_entries[0].created_at
            }
        
        return MoodHistoryResponse(
            total_entries=len(mood_entries),
            entries=mood_entries,
            date_range=date_range
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve mood history: {str(e)}"
        )


@router.get("/trends", response_model=EmotionTrendsResponse)
async def get_emotion_trends(
    days: int = 7,
    moods_collection=Depends(get_moods_collection)
):
    try:
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        cursor = moods_collection.find({
            "user_id": MOCK_USER_ID,
            "created_at": {"$gte": start_date, "$lte": end_date}
        }).sort("created_at", 1)
        
        entries = await cursor.to_list(length=1000)
        
        if not entries:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No mood entries found in the specified date range"
            )
        
        emotions_list = [
            entry['emotion_analysis']['primary_emotion'] 
            for entry in entries
        ]
        
        analyzer = get_emotion_analyzer()
        summary = analyzer.get_emotion_summary(emotions_list)
        
        daily_emotions = []
        for entry in entries:
            daily_emotions.append({
                "date": entry['created_at'].strftime("%Y-%m-%d"),
                "emotion": entry['emotion_analysis']['primary_emotion'],
                "confidence": entry['emotion_analysis']['confidence']
            })
        
        return EmotionTrendsResponse(
            total_entries=summary['total_entries'],
            date_range={"start": start_date, "end": end_date},
            emotion_distribution=summary['emotion_distribution'],
            most_common_emotion=summary['most_common_emotion'],
            daily_emotions=daily_emotions
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve emotion trends: {str(e)}"
        )


@router.delete("/{mood_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_mood_entry(
    mood_id: str,
    moods_collection=Depends(get_moods_collection)
):
    try:
        result = await moods_collection.delete_one({
            "_id": ObjectId(mood_id),
            "user_id": MOCK_USER_ID
        })
        
        if result.deleted_count == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Mood entry not found"
            )
        
        return None
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete mood entry: {str(e)}"
        )
