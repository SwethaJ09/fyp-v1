from flask import Flask, render_template, request
import speech_recognition as sr
from textblob import TextBlob
import librosa
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Function to analyze pauses in speech
def analyze_pauses(audio_signal, sr_rate):
    pauses = librosa.effects.split(audio_signal, top_db=20)
    pause_durations = []
    for i in range(1, len(pauses)):
        pause_durations.append((pauses[i][0] - pauses[i-1][1]) / sr_rate)
    total_pause_duration = sum(pause_durations)
    return total_pause_duration, pause_durations

# Function to analyze speech rate and fluency
def analyze_fluency(text):
    word_count = len(text.split())
    stopwords = set(['a', 'the', 'and', 'is', 'in', 'at', 'of', 'to', 'with'])
    stopword_count = len([word for word in text.split() if word.lower() in stopwords])
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    vocab_rating = len(set(text.split())) / word_count * 100
    return word_count, stopword_count, sentiment, vocab_rating

# Function to evaluate the clarity of speech
def evaluate_clarity(total_pause_duration, audio_duration, sentiment, vocab_rating):
    clarity_score = (1 - (total_pause_duration / audio_duration)) * 50 + (sentiment * 50)
    clarity_score += vocab_rating / 2
    return clarity_score

# Function to generate a pie chart and return it as a base64 string
def generate_pie_chart(data, labels, title):
    fig, ax = plt.subplots()
    ax.pie(data, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#99ff99', '#ff9999'])
    ax.axis('equal')
    plt.title(title)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)

    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return img_base64

# Function to calculate speaking pace in WPM
def calculate_speaking_pace(word_count, audio_duration):
    return (word_count / audio_duration) * 60

# Function to calculate listenability (a hypothetical metric for illustration)
def calculate_listenability(clarity_score, sentiment):
    return (clarity_score + (sentiment * 100)) / 2

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Fixed audio file path
        #audio_path = r'C:\Users\Subash\OneDrive\Desktop\fyp\fyp\fyp\Assets\DynamicRecordedAudio.wav'
        audio_path = r'C:\Users\Subash\OneDrive\Desktop\fyp\fyp\fyp\Assets\DynamicRecordedAudio.wav'

        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)
            audio_signal, sr_rate = librosa.load(audio_path)

        try:
            text = recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            text = "Audio not recognized"
        except sr.RequestError:
            text = "Error with Google Speech Recognition API"

        total_pause_duration, pause_durations = analyze_pauses(audio_signal, sr_rate)
        word_count, stopword_count, sentiment, vocab_rating = analyze_fluency(text)
        audio_duration = librosa.get_duration(y=audio_signal, sr=sr_rate)
        clarity_score = evaluate_clarity(total_pause_duration, audio_duration, sentiment, vocab_rating)
        wpm = calculate_speaking_pace(word_count, audio_duration)
        listenability_score = calculate_listenability(clarity_score, sentiment)

        results = {
            'text': text,
            'word_count': word_count,
            'stopword_count': stopword_count,
            'sentiment': sentiment,
            'vocab_rating': vocab_rating,
            'total_pause_duration': total_pause_duration,
            'audio_duration': audio_duration,
            'clarity_score': clarity_score,
            'wpm': wpm,
            'listenability_score': listenability_score
        }

        sentiment_pie = generate_pie_chart([results['sentiment'], 1 - results['sentiment']], ['Sentiment', ''], 'Sentiment Analysis')
        clarity_pie = generate_pie_chart([results['clarity_score'], 100 - results['clarity_score']], ['Clarity', ''], 'Clarity Score')
        stopwords_pie = generate_pie_chart([results['stopword_count'], results['word_count'] - results['stopword_count']], ['Stop Words', ''], 'Stop Words')
        listenability_pie = generate_pie_chart([results['listenability_score'], 100 - results['listenability_score']], ['Listenability', ''], 'Listenability')
        duration_pie = generate_pie_chart([results['audio_duration'], results['total_pause_duration']], ['Duration', 'Pauses'], 'Audio Duration')
        wpm_pie = generate_pie_chart([results['wpm'], 300 - results['wpm']], ['WPM', ''], 'Speaking Pace (WPM)')

        return render_template('report_template.html', results=results, sentiment_pie=sentiment_pie,
                               clarity_pie=clarity_pie, stopwords_pie=stopwords_pie,
                               listenability_pie=listenability_pie, duration_pie=duration_pie,
                               wpm_pie=wpm_pie)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
