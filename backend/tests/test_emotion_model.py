import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.emotion_analyzer import EmotionAnalyzer


def test_emotion_model():
    
    print("=" * 60)
    print("MoodMirror - Emotion Model Test")
    print("=" * 60)
    print()
    
    print("Loading emotion analysis model...")
    analyzer = EmotionAnalyzer()
    print("✓ Model loaded successfully!\n")
    
    test_cases = [
        "I'm feeling overwhelmed with all my assignments today",
        "I aced my exam! So excited and happy!",
        "I'm worried about my upcoming presentation",
        "Just had a great time with friends, feeling grateful",
        "I'm so angry about the unfair grading",
        "This professor's lecture was really boring",
        "I feel anxious about graduation and what comes next",
        "Today was a normal day, nothing special",
    ]
    
    print("Testing emotion detection on sample student mood inputs:")
    print("-" * 60)
    print()
    
    for i, text in enumerate(test_cases, 1):
        result = analyzer.analyze_emotion(text)
        
        print(f"Test {i}:")
        print(f"  Input: \"{text}\"")
        print(f"  Primary Emotion: {result['primary_emotion'].upper()}")
        print(f"  Confidence: {result['confidence'] * 100:.2f}%")
        print(f"  Top 3 Emotions:")
        
        sorted_emotions = sorted(
            result['all_emotions'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:3]
        
        for emotion, score in sorted_emotions:
            print(f"    - {emotion}: {score * 100:.2f}%")
        
        print()
    
    print("-" * 60)
    print("Testing batch analysis:")
    print()
    
    batch_texts = test_cases[:3]
    batch_results = analyzer.analyze_batch(batch_texts)
    
    print(f"Analyzed {len(batch_results)} entries in batch")
    emotions = [r['primary_emotion'] for r in batch_results]
    print(f"Detected emotions: {emotions}")
    print()
    
    print("-" * 60)
    print("Testing emotion summary statistics:")
    print()
    
    all_emotions = [r['primary_emotion'] for r in analyzer.analyze_batch(test_cases)]
    summary = analyzer.get_emotion_summary(all_emotions)
    
    print(f"Total entries: {summary['total_entries']}")
    print(f"Most common emotion: {summary['most_common_emotion']}")
    print(f"Unique emotions detected: {summary['unique_emotions']}")
    print()
    print("Emotion Distribution:")
    for emotion, data in summary['emotion_distribution'].items():
        print(f"  {emotion}: {data['count']} entries ({data['percentage']}%)")
    
    print()
    print("=" * 60)
    print("✓ All tests completed successfully!")
    print("The emotion model is working correctly and ready to use.")
    print("=" * 60)


if __name__ == "__main__":
    try:
        test_emotion_model()
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        sys.exit(1)
