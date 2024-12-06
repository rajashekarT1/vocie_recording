import streamlit as st
import whisper
import tempfile
import os
import ffmpeg
import requests
from streamlit.components.v1 import html
import pandas as pd
import wave
import io

model = whisper.load_model("base")

def convert_mp3_to_wav(mp3_path, wav_path):
    try:
        ffmpeg.input(mp3_path).output(wav_path, ac=1, ar="16k").run(overwrite_output=True)
        st.text(f"Audio converted to WAV: {wav_path}")
    except Exception as e:
        st.error(f"Error during conversion: {e}")

def record_audio():
    html_code = """
    <style>
        body {
            background-color: #f0f8ff;
            font-family: 'Arial', sans-serif;
            text-align: center;
            color: #333;
        }

        button {
            background-color: grey;
            color: white;
            font-size: 24px;
            padding: 10px 15px;
            border: none;
            border-radius: 8px;
            margin: 10px;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }

        button:hover {
            background-color: #45a049;
        }

        #recordStatus {
            font-size: 18px;
            color: #FF6347;
        }

        #playButton {
            display: none;
        }

        .icon {
            font-size: 30px;
        }
    </style>

<script>
    let audioData = null;
    let recorder = null;
    let audioBlob = null;
    let audioUrl = null;
    let isRecording = false;
    let currentAudio = null; // Store current audio object to manage playback

    function Start() {
        navigator.mediaDevices.getUserMedia({ audio: true })
        .then(function(stream) {
            recorder = new MediaRecorder(stream);
            recorder.ondataavailable = function(event) {
                audioData = event.data;
                audioBlob = new Blob([audioData], { type: "audio/wav" });
                audioUrl = URL.createObjectURL(audioBlob);
                document.getElementById("recordStatus").innerHTML = "Recording... Click stop to finish.";
                sessionStorage.setItem('audioBlob', audioUrl);
                window.parent.postMessage({ type: 'start_recording', audioUrl: audioUrl }, '*');
            };
            recorder.start();
            isRecording = true;
            document.getElementById("recordStatus").innerHTML = "Recording...";
        }).catch(function(err) {
            alert("Error accessing microphone: " + err);
        });
    }

    function Stop() {
        if (recorder && isRecording) {
            recorder.stop();
            document.getElementById("recordStatus").innerHTML = "Stopped. You can now play or download the recording.";
            window.parent.postMessage({ type: 'stop_recording', audioUrl: audioUrl }, '*');
            document.getElementById("playButton").style.display = "inline-block";
        }
    }

    function Play() {
        if (currentAudio) {
            // If there is already an audio playing, stop it
            currentAudio.pause();
            currentAudio.currentTime = 0;
        }

        let audio = new Audio(audioUrl);
        currentAudio = audio; // Update the current audio
        audio.play();
        audio.onended = function() {
            document.getElementById("playButton").style.display = "inline-block";
        };
    }

    function Download() {
        let a = document.createElement('a');
        a.href = audioUrl;
        a.download = "audio.wav";
        a.click();
    }
</script>


    <div>
        <button onclick="Start()" title="Start"><span class="icon">🔘</span></button>
        <button onclick="Stop()" title="Stop"><span class="icon">⏹️</span></button>
        <button onclick="Play()" title="Play"><span class="icon">▶️</span></button>
        <button onclick="Download()" title="Download"><span class="icon">⬇️</span></button>
        <p id="recordStatus">Not Recording</p>
    </div>
    """
    return html_code

def transcribe_audio(audio_path):
    try:
        result = model.transcribe(audio_path, language="en")
        return result["text"]
    except Exception as e:
        st.error(f"Error during transcription: {e}")
        return None

def save_recorded_file(audio_url, transcription):
    if 'audio_files' not in st.session_state:
        st.session_state.audio_files = []

    st.session_state.audio_files.append({"audio_url": audio_url, "transcription": transcription, "timestamp": pd.to_datetime("now")})

    df = pd.DataFrame(st.session_state.audio_files)
    df.to_csv('audio.csv', index=False)

def load_audio_history():
    if os.path.exists('audio.csv'):
        df = pd.read_csv('audio.csv')
        return df
    else:
        return pd.DataFrame(columns=["audio_url", "transcription", "timestamp"])

def main():
    st.markdown("### Recorder")

    html_code = record_audio()
    html(html_code)

    transcription = ""

    if 'audio_url' in st.session_state:
        audio_url = st.session_state['audio_url']

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(requests.get(audio_url).content)
                tmp_file_path = tmp_file.name

            transcription = transcribe_audio(tmp_file_path)

            if transcription:
                st.text_area("Transcribed Text", transcription, height=200)

                save_recorded_file(audio_url, transcription)

        except Exception as e:
            st.error(f"Error during processing: {e}")
        
        finally:
            os.remove(tmp_file_path)

    audio_file = st.file_uploader("Upload Audio", type=["mp3", "wav"])

    if st.sidebar.button("Transcribe Audio"):
        if audio_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                tmp_file.write(audio_file.getbuffer())
                tmp_file_path = tmp_file.name

            try:
                if audio_file.type == "audio/mp3":
                    st.text("Converting MP3 to WAV...")
                    wav_path = tmp_file_path.replace(".mp3", ".wav")
                    convert_mp3_to_wav(tmp_file_path, wav_path)
                else:
                    wav_path = tmp_file_path
                transcription = transcribe_audio(wav_path)

                if transcription:
                    st.text_area("Transcribed Text", transcription, height=200)

                    save_recorded_file(wav_path, transcription)

            except Exception as e:
                st.error(f"Error during transcription: {e}")

            finally:
                os.remove(tmp_file_path)

if __name__ == "__main__":
    main()





# import streamlit as st
# import whisper
# import tempfile
# import os
# import ffmpeg
# import requests
# from streamlit.components.v1 import html
# import pandas as pd
# import wave
# import io
# from pyannote.audio import Inference
# from pyannote.core import Segment
# import torch

# model = whisper.load_model("base")
# from pyannote.audio import Inference
# diarization_model = Inference("pyannote/speaker-diarization", use_auth_token="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")
# def convert_mp3_to_wav(mp3_path, wav_path):
#     try:
#         ffmpeg.input(mp3_path).output(wav_path, ac=1, ar="16k").run(overwrite_output=True)
#         st.text(f"Audio converted to WAV: {wav_path}")
#     except Exception as e:
#         st.error(f"Error during conversion: {e}")

# def record_audio():
#     html_code = """
#     <style>
#         body {
#             background-color: #f0f8ff;
#             font-family: 'Arial', sans-serif;
#             text-align: center;
#             color: #333;
#         }

#         button {
#             background-color: grey;
#             color: white;
#             font-size: 24px;
#             padding: 10px 15px;
#             border: none;
#             border-radius: 8px;
#             margin: 10px;
#             cursor: pointer;
#             transition: background-color 0.3s ease;
#         }

#         button:hover {
#             background-color: #45a049;
#         }

#         #recordStatus {
#             font-size: 18px;
#             color: #FF6347;
#         }

#         #playButton {
#             display: none;
#         }

#         .icon {
#             font-size: 30px;
#         }
#     </style>

# <script>
#     let audioData = null;
#     let recorder = null;
#     let audioBlob = null;
#     let audioUrl = null;
#     let isRecording = false;
#     let currentAudio = null; // Store current audio object to manage playback

#     function Start() {
#         navigator.mediaDevices.getUserMedia({ audio: true })
#         .then(function(stream) {
#             recorder = new MediaRecorder(stream);
#             recorder.ondataavailable = function(event) {
#                 audioData = event.data;
#                 audioBlob = new Blob([audioData], { type: "audio/wav" });
#                 audioUrl = URL.createObjectURL(audioBlob);
#                 document.getElementById("recordStatus").innerHTML = "Recording... Click stop to finish.";
#                 sessionStorage.setItem('audioBlob', audioUrl);
#                 window.parent.postMessage({ type: 'start_recording', audioUrl: audioUrl }, '*');
#             };
#             recorder.start();
#             isRecording = true;
#             document.getElementById("recordStatus").innerHTML = "Recording...";
#         }).catch(function(err) {
#             alert("Error accessing microphone: " + err);
#         });
#     }

#     function Stop() {
#         if (recorder && isRecording) {
#             recorder.stop();
#             document.getElementById("recordStatus").innerHTML = "Stopped. You can now play or download the recording.";
#             window.parent.postMessage({ type: 'stop_recording', audioUrl: audioUrl }, '*');
#             document.getElementById("playButton").style.display = "inline-block";
#         }
#     }

#     function Play() {
#         if (currentAudio) {
#             // If there is already an audio playing, stop it
#             currentAudio.pause();
#             currentAudio.currentTime = 0;
#         }

#         let audio = new Audio(audioUrl);
#         currentAudio = audio; // Update the current audio
#         audio.play();
#         audio.onended = function() {
#             document.getElementById("playButton").style.display = "inline-block";
#         };
#     }

#     function Download() {
#         let a = document.createElement('a');
#         a.href = audioUrl;
#         a.download = "audio.wav";
#         a.click();
#     }
# </script>


#     <div>
#         <button onclick="Start()" title="Start"><span class="icon">🔘</span></button>
#         <button onclick="Stop()" title="Stop"><span class="icon">⏹️</span></button>
#         <button onclick="Play()" title="Play"><span class="icon">▶️</span></button>
#         <button onclick="Download()" title="Download"><span class="icon">⬇️</span></button>
#         <p id="recordStatus">Not Recording</p>
#     </div>
#     """
#     return html_code

# def transcribe_audio_with_diarization(audio_path):
#     try:
#         # Perform speaker diarization
#         diarization = diarization_model({'uri': 'filename', 'audio': audio_path})

#         # Transcribe using whisper
#         result = model.transcribe(audio_path, language="en")

#         # Get transcriptions and segment speakers
#         transcription = []
#         for segment, _, speaker in diarization.itertracks(yield_label=True):
#             speaker_text = result["text"]  # Whisper already gives transcriptions
#             transcription.append(f"Speaker {speaker}: {speaker_text}")
        
#         return transcription
#     except Exception as e:
#         st.error(f"Error during transcription or diarization: {e}")
#         return None

# def save_recorded_file(audio_url, transcription):
#     if 'audio_files' not in st.session_state:
#         st.session_state.audio_files = []

#     st.session_state.audio_files.append({"audio_url": audio_url, "transcription": transcription, "timestamp": pd.to_datetime("now")})

#     df = pd.DataFrame(st.session_state.audio_files)
#     df.to_csv('audio.csv', index=False)

# def load_audio_history():
#     if os.path.exists('audio.csv'):
#         df = pd.read_csv('audio.csv')
#         return df
#     else:
#         return pd.DataFrame(columns=["audio_url", "transcription", "timestamp"])

# def main():
#     st.markdown("### Recorder with Speaker Diarization")

#     html_code = record_audio()
#     html(html_code)

#     transcription = ""

#     if 'audio_url' in st.session_state:
#         audio_url = st.session_state['audio_url']

#         try:
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
#                 tmp_file.write(requests.get(audio_url).content)
#                 tmp_file_path = tmp_file.name

#             # Perform transcription with diarization
#             transcription = transcribe_audio_with_diarization(tmp_file_path)

#             if transcription:
#                 st.text_area("Transcribed Text with Speaker Diarization", "\n".join(transcription), height=200)

#                 save_recorded_file(audio_url, transcription)

#         except Exception as e:
#             st.error(f"Error during processing: {e}")
        
#         finally:
#             os.remove(tmp_file_path)

#     audio_file = st.file_uploader("Upload Audio", type=["mp3", "wav"])

#     if st.sidebar.button("Transcribe Audio"):
#         if audio_file is not None:
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
#                 tmp_file.write(audio_file.getbuffer())
#                 tmp_file_path = tmp_file.name

#             try:
#                 if audio_file.type == "audio/mp3":
#                     st.text("Converting MP3 to WAV...")
#                     wav_path = tmp_file_path.replace(".mp3", ".wav")
#                     convert_mp3_to_wav(tmp_file_path, wav_path)
#                 else:
#                     wav_path = tmp_file_path

#                 # Perform transcription with diarization
#                 transcription = transcribe_audio_with_diarization(wav_path)

#                 if transcription:
#                     st.text_area("Transcribed Text with Speaker Diarization", "\n".join(transcription), height=200)

#                     save_recorded_file(wav_path, transcription)

#             except Exception as e:
#                 st.error(f"Error during transcription: {e}")

#             finally:
#                 os.remove(tmp_file_path)

# if __name__ == "__main__":
#     main()
