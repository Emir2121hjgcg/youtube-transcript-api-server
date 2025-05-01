from flask import Flask, jsonify
from youtube_transcript_api import YouTubeTranscriptApi

app = Flask(__name__)

@app.route('/')
def home():
    return "YouTube Transcript API is live!"

@app.route('/transcript/<video_id>', methods=['GET'])
def transcript(video_id):
    try:
        result = YouTubeTranscriptApi.get_transcript(video_id)
        return jsonify({"transcript": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if name == '__main__':
    app.run(host='0.0.0.0', port=5000)


