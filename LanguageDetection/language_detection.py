import whisper
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wavfile
from datetime import datetime
from flask import Flask,request,jsonify
import os
app=Flask(__name__)
model=whisper.load_model("base")
SAMPLE_RATE=16000
DURATION=10
OUTPUT_FILE="recorded_audio.wav"

def record():
    print(f"Recording for {DURATION} seconds")
    audio=sd.rec(int(DURATION * SAMPLE_RATE),samplerate=SAMPLE_RATE,channels=1)
    sd.wait()
    wavfile.write(OUTPUT_FILE,SAMPLE_RATE,audio)
    print("Recording finished")
    return OUTPUT_FILE

def detect(audio_path):
    audio=whisper.load_audio(audio_path)
    audio=whisper.pad_or_trim(audio)

    mel=whisper.log_mel_spectrogram(audio).to(model.device)
    _,probs=model.detect_language(mel)

    detected_lang=max(probs,key=probs.get)
    conf=probs[detected_lang]

    res=model.transcribe(audio_path,language=detected_lang)
    trans=res["text"]

    return detected_lang,conf,trans

@app.route('/detect-language',methods=['POST'])
def detect_endpoint():
    try:
        if 'audio' in request.files:
            audio_file=request.files['audio']
            audio_path=os.path.join("uploads",audio_file.filename)
            os.makedirs("uploads",exist_ok=True)
            audio_file.save(audio_path)
        else:
            audio_path=record()
        det_lang,conf,_,_=detect(audio_path)
        if os.path.exists(audio_path) and audio_path!=OUTPUT_FILE:
            os.remove(audio_path)
        return jsonify({
            "status":"success",
            "detected_language":det_lang,
            "confidence":f"{conf:.2%}"
        })
    except Exception as e:
        return jsonify({"status":"error","message":str(e)}),500
    

@app.route('/transcription',methods=['POST'])
def transcription_endpoint():
    try:
        if 'audio' in request.files:
            audio_file=request.files['audio']
            audio_path=os.path.join("uploads",audio_file.filename)
            os.makedirs("uploads",exist_ok=True)
            audio_file.save(audio_path)
        else:
            audio_path=record()
        _,_,trans,_=detect(audio_path)
        if os.path.exists(audio_path) and audio_path!=OUTPUT_FILE:
            os.remove(audio_path)
        return jsonify({
            "status":"success",
            "transcription":trans
        })
    except Exception as e:
        return jsonify({"status":"error","message":str(e)}),500
    

@app.route('/translation',methods=['POST'])
def translation_endpoint():
    try:
        if 'audio' in request.files:
            audio_file=request.files['audio']
            audio_path=os.path.join("uploads",audio_file.filename)
            os.makedirs("uploads",exist_ok=True)
            audio_file.save(audio_path)
        else:
            audio_path=record()
        det_lang,_,_,trans=detect(audio_path)
        if os.path.exists(audio_path) and audio_path!=OUTPUT_FILE:
            os.remove(audio_path)
        return jsonify({
            "status":"success",
            "detected_language":det_lang,
            "translation":trans
        })
    except Exception as e:
        return jsonify({"status":"error","message":str(e)}),500

if __name__=="__main__":
    app.run(debug=True,host="0.0.0.0",port=5000)