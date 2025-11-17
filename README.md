# MoodMirror – AI-Powered Emotion Tracking

**Minimal Phase 1:** Text-based emotion detection using DistilRoBERTa

## What It Does

- Submit daily mood text (e.g., "I'm feeling overwhelmed with assignments")
- AI analyzes and detects 7 emotions: joy, sadness, anger, fear, surprise, disgust, neutral
- View mood history and emotional trends over time

## Quick Start

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Set Up MongoDB

**Option A - Local MongoDB:**
```bash
brew install mongodb-community
brew services start mongodb-community
```

**Option B - MongoDB Atlas (Cloud - Free):**
- Sign up at mongodb.com/cloud/atlas
- Create free cluster
- Get connection string

### 3. Configure Environment

```bash
cp env.example .env
```

Edit `.env`:
```
MONGODB_URL=mongodb://localhost:27017
DATABASE_NAME=moodmirror_db
SECRET_KEY=your-secret-key
```

### 4. Test the Model (Optional)

```bash
python tests/test_emotion_model.py
```

### 5. Run the Server

```bash
uvicorn app.main:app --reload
```

Server runs at: **http://localhost:8000**

## Test the API

### Interactive Docs (Easiest)
Open: **http://localhost:8000/docs**

### Submit Mood Entry
```bash
curl -X POST "http://localhost:8000/api/mood/submit" \
  -H "Content-Type: application/json" \
  -d '{"text": "I am feeling happy today!"}'
```

### Get Mood History
```bash
curl http://localhost:8000/api/mood/history
```

### Get Emotion Trends (Last 7 Days)
```bash
curl http://localhost:8000/api/mood/trends?days=7
```

## Project Structure

```
backend/
├── app/
│   ├── config/          # Settings & database
│   ├── models/          # Data structures
│   ├── routes/          # API endpoints
│   ├── services/        # Emotion analysis (AI core)
│   └── main.py          # App entry point
├── tests/               # Test scripts
└── requirements.txt     # Dependencies
```

## API Endpoints

- `POST /api/mood/submit` - Submit mood entry
- `GET /api/mood/history` - Get mood history
- `GET /api/mood/trends` - Get emotion trends
- `DELETE /api/mood/{id}` - Delete entry

## Tech Stack

- **Backend:** FastAPI (Python)
- **AI Model:** j-hartmann/emotion-english-distilroberta-base
- **Database:** MongoDB
- **NLP:** HuggingFace Transformers

## Troubleshooting

**MongoDB connection failed?**
```bash
brew services start mongodb-community
```

**Port 8000 in use?**
```bash
uvicorn app.main:app --reload --port 8001
```

**Module not found?**
```bash
pip install -r requirements.txt --force-reinstall
```

## Next Steps

- Add user authentication (JWT)
- Build frontend (React/Vue)
- Add audio emotion detection (Phase 2)
- Add video emotion detection (Phase 2)

---

**Model:** DistilRoBERTa | **Accuracy:** 90%+ | **Phase:** 1 (Text Analysis)
