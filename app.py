# -*- coding: utf-8 -*-
import os
import subprocess
import tempfile
import logging
from pathlib import Path
from typing import Optional, List, Dict
import requests
import streamlit as st
from faster_whisper import WhisperModel
from streamlit_mic_recorder import mic_recorder

# Configuration (peuvent être surchargées par variables d'environnement)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
DEFAULT_OLLAMA_MODEL = os.getenv("DEFAULT_OLLAMA_MODEL", "qwen2.5:3b")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "1800"))  # secondes

# Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit page
st.set_page_config(page_title="Transcripteur IA local", page_icon="🎙", layout="wide")

# ---------- Utils ----------
def get_file_extension(filename: str) -> str:
    return Path(filename).suffix.lower()

def save_uploaded_file(uploaded_file) -> str:
    """Sauvegarde un fichier Streamlit dans un fichier temporaire et retourne le chemin."""
    suffix = get_file_extension(uploaded_file.name) or ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name

def save_audio_bytes(audio_bytes: bytes, extension: str = ".wav") -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp:
        tmp.write(audio_bytes)
        return tmp.name

def ensure_ffmpeg_available():
    from shutil import which
    if which("ffmpeg") is None:
        raise RuntimeError("FFmpeg n'est pas installé ou n'est pas accessible dans le PATH. Installez-le pour pouvoir convertir les médias.")

def extract_audio(input_path: str) -> str:
    """Convertit un fichier audio/vidéo en WAV mono 16 kHz. FFmpeg requis."""
    ensure_ffmpeg_available()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as out:
        output_path = out.name

    command = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-sample_fmt",
        "s16",
        output_path,
    ]

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        logger.error("ffmpeg error: %s", result.stderr[-2000:])
        raise RuntimeError("Erreur pendant la conversion audio :\n\n" + (result.stderr or "ffmpeg a échoué"))
    return output_path

# ---------- Whisper ----------
@st.cache_resource
def load_whisper_model(model_size: str, device: str, compute_type: str) -> WhisperModel:
    return WhisperModel(model_size, device=device, compute_type=compute_type)

def transcribe_audio(audio_path: str, model_size: str, language: Optional[str], device: str, compute_type: str) -> Dict:
    model = load_whisper_model(model_size=model_size, device=device, compute_type=compute_type)
    segments, info = model.transcribe(audio_path, language=language, beam_size=5, vad_filter=True, word_timestamps=False)

    collected_segments = []
    full_text_parts = []
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        collected_segments.append({"start": segment.start, "end": segment.end, "text": text})
        full_text_parts.append(text)

    full_text = " ".join(full_text_parts)
    return {"text": full_text, "segments": collected_segments, "language": info.language, "language_probability": info.language_probability}

def format_timestamp(seconds: float) -> str:
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    remaining_seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"

def format_transcript_with_timestamps(segments: List[Dict]) -> str:
    lines = []
    for segment in segments:
        start = format_timestamp(segment["start"])
        end = format_timestamp(segment["end"])
        lines.append(f"[{start} - {end}] {segment['text']}")
    return "\n".join(lines)

def segments_to_srt(segments: List[Dict]) -> str:
    def fmt_srt_time(s: float) -> str:
        ms = int((s - int(s)) * 1000)
        h = int(s) // 3600
        m = (int(s) % 3600) // 60
        sec = int(s) % 60
        return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"
    parts = []
    for i, seg in enumerate(segments, start=1):
        parts.append(f"{i}\n{fmt_srt_time(seg['start'])} --> {fmt_srt_time(seg['end'])}\n{seg['text']}\n")
    return "\n".join(parts)

# ---------- Ollama ----------
def ask_ollama(prompt: str, model_name: str, temperature: float = 0.2, timeout: int = OLLAMA_TIMEOUT) -> str:
    payload = {"model": model_name, "prompt": prompt, "stream": False, "options": {"temperature": temperature}}
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(OLLAMA_URL, json=payload, headers=headers, timeout=timeout)
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Impossible de contacter Ollama. Vérifiez qu'il est démarré avec 'ollama serve' et que OLLAMA_URL est correct.")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Erreur réseau lors de l'appel à Ollama : {e}")

    if response.status_code != 200:
        raise RuntimeError(f"Ollama a retourné une erreur HTTP {response.status_code} :\n{response.text}")

    data = response.json()
    if "response" not in data:
        raise RuntimeError("Réponse inattendue d'Ollama :\n" + str(data))
    return data["response"].strip()

def split_text(text: str, max_characters: int = 12000) -> List[str]:
    text = text.strip()
    if len(text) <= max_characters:
        return [text]
    paragraphs = text.split("\n")
    chunks = []
    current = ""
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        candidate = (current + "\n" + p).strip() if current else p
        if len(candidate) > max_characters:
            if current:
                chunks.append(current)
            current = p
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks

def generate_summary(transcript: str, model_name: str, summary_type: str) -> str:
    chunks = split_text(transcript)
    if summary_type == "Résumé court":
        instruction = (
            "Rédige un résumé court et clair en français. Conserve uniquement les informations importantes. "
            "Ne fais pas de liste d'actions si elles ne sont pas explicitement mentionnées. N'invente aucune information."
        )
    elif summary_type == "Compte rendu détaillé":
        instruction = (
            "Rédige un compte rendu détaillé en français avec les sections : Compte rendu, Sujet principal, Synthèse, "
            "Points importants, Décisions prises, Actions à réaliser (responsable/échéance si présents), Questions/points à clarifier. "
            "N'invente aucune information ; écris 'Non précisé' si absent."
        )
    else:
        instruction = (
            "Analyse cette transcription et produis une synthèse structurée en français identifiant le sujet principal, "
            "les idées importantes, décisions, actions, personnes citées, dates citées et points non résolus. N'invente rien."
        )

    partial_results = []
    for i, chunk in enumerate(chunks, start=1):
        prompt = (
            f"Tu es un assistant spécialisé dans l'analyse de transcriptions audio.\n"
            f"{instruction}\n"
            f"Voici la partie {i} de la transcription :\n---------------- TRANSCRIPTION ----------------\n{chunk}\n---------------- FIN TRANSCRIPTION -------------\n"
            "Réponds uniquement avec le résultat demandé."
        )
        partial_results.append(ask_ollama(prompt=prompt, model_name=model_name, temperature=0.2))

    if len(partial_results) == 1:
        return partial_results[0]

    combined = "\n\n".join(f"### Partie {idx}\n{res}" for idx, res in enumerate(partial_results, start=1))
    final_prompt = (
        "Tu es un assistant chargé de fusionner plusieurs analyses d'une même transcription. Fusionne les éléments ci-dessous "
        "en un document cohérent en français. Supprime les répétitions. Ne crée aucune information qui n'existe pas dans les analyses.\n\n"
        "Utilise les sections : Compte rendu final, Sujet principal, Synthèse, Points importants, Décisions prises, Actions à réaliser, Questions ou points à clarifier.\n\n"
        "Analyses à fusionner :\n" + combined
    )
    return ask_ollama(prompt=final_prompt, model_name=model_name, temperature=0.2)

def build_markdown_export(transcript: str, summary: str, detected_language: str) -> str:
    return (
        "# Transcription et compte rendu\n\n"
        f"Langue détectée\n{detected_language}\n\n"
        "Compte rendu\n"
        f"{summary}\n\n"
        "Transcription\n"
        f"{transcript}\n"
    )

# ---------- UI ----------
st.title("🎙️ Transcripteur audio/vidéo avec IA")
st.caption("Whisper transcrit localement et Ollama génère le résumé ou le compte rendu.")

with st.sidebar:
    st.header("⚙️ Configuration")
    whisper_model_size = st.selectbox(
        "Modèle Whisper",
        options=["tiny", "base", "small", "medium", "large-v3"],
        index=2,
        help=("small est un bon compromis. medium ou large-v3 donnent de meilleurs résultats mais demandent davantage de ressources.")
    )
    whisper_device = st.selectbox("Appareil de calcul", options=["cpu", "cuda"], index=0)
    if whisper_device == "cpu":
        whisper_compute_type = st.selectbox("Type de calcul", options=["int8", "float32"], index=0)
    else:
        whisper_compute_type = st.selectbox("Type de calcul", options=["float16", "int8_float16"], index=0)

    whisper_language_choice = st.selectbox(
        "Langue audio",
        options=["Détection automatique", "Français", "Anglais", "Espagnol", "Allemand", "Italien"],
        index=0,
    )
    language_mapping = {
        "Détection automatique": None,
        "Français": "fr",
        "Anglais": "en",
        "Espagnol": "es",
        "Allemand": "de",
        "Italien": "it",
    }
    whisper_language = language_mapping[whisper_language_choice]
    ollama_model = st.text_input("Modèle Ollama", value=DEFAULT_OLLAMA_MODEL)
    ollama_timeout = st.number_input("Timeout Ollama (s)", value=OLLAMA_TIMEOUT, min_value=30, max_value=3600)
    summary_type = st.selectbox("Type de résultat", options=["Résumé court", "Compte rendu détaillé", "Analyse structurée"], index=1)

# Enregistrement / upload
st.header("1. Enregistrer ou importer un média")
recorded_audio = None
tab_record, tab_upload = st.tabs(["🎤 Enregistrer un audio", "📁 Importer un fichier"])
with tab_record:
    st.write("Cliquez sur le bouton pour enregistrer un message audio depuis votre navigateur.")
    recorded_audio = mic_recorder(
        start_prompt="Démarrer l'enregistrement",
        stop_prompt="Arrêter l'enregistrement",
        just_once=False,
        use_container_width=True,
        format="wav",
        key="audio_recorder",
    )
    if recorded_audio and recorded_audio.get("bytes"):
        st.audio(recorded_audio["bytes"], format="audio/wav")
        st.success("Enregistrement prêt à être transcrit.")
with tab_upload:
    uploaded_file = st.file_uploader("Sélectionnez un fichier audio ou vidéo", type=["wav","mp3","m4a","ogg","flac","webm","mp4","mov","avi","mkv"])
    if uploaded_file:
        st.write(f"Fichier sélectionné : **{uploaded_file.name}** ({uploaded_file.size / 1024 / 1024:.2f} Mo)")
        ext = get_file_extension(uploaded_file.name)
        if ext in [".mp4", ".mov", ".avi", ".mkv", ".webm"]:
            st.video(uploaded_file)
        else:
            st.audio(uploaded_file)

st.divider()
st.header("2. Transcription et analyse")
if "processing" not in st.session_state:
    st.session_state["processing"] = False

process_button = st.button("🚀 Lancer la transcription et générer le compte rendu", type="primary", use_container_width=True)

if process_button and not st.session_state["processing"]:
    st.session_state["processing"] = True
    input_path: Optional[str] = None
    converted_audio_path: Optional[str] = None
    try:
        # Préparation du fichier d'entrée
        if recorded_audio and recorded_audio.get("bytes"):
            with st.spinner("Préparation de l'enregistrement..."):
                input_path = save_audio_bytes(recorded_audio["bytes"], extension=".wav")
        elif 'uploaded_file' in locals() and uploaded_file:
            with st.spinner("Préparation du fichier..."):
                input_path = save_uploaded_file(uploaded_file)
        else:
            st.warning("Veuillez enregistrer un audio ou importer un fichier.")
            st.session_state["processing"] = False
            st.stop()

        with st.status("Traitement en cours...", expanded=True) as status:
            status.write("Conversion du média en WAV mono 16 kHz...")
            converted_audio_path = extract_audio(input_path)

            status.write(f"Transcription avec Whisper ({whisper_model_size})...")
            transcription_result = transcribe_audio(
                audio_path=converted_audio_path,
                model_size=whisper_model_size,
                language=whisper_language,
                device=whisper_device,
                compute_type=whisper_compute_type,
            )

            transcript_with_timestamps = format_transcript_with_timestamps(transcription_result["segments"])

            if not transcript_with_timestamps.strip():
                raise RuntimeError("Aucun texte n'a été détecté dans l'enregistrement.")

            status.write(f"Langue détectée : {transcription_result.get('language')}")
            status.write(f"Analyse avec Ollama ({ollama_model})...")

            summary = generate_summary(
                transcript=transcript_with_timestamps,
                model_name=ollama_model,
                summary_type=summary_type,
            )

            status.update(label="Traitement terminé", state="complete", expanded=False)

        st.session_state["transcript"] = transcript_with_timestamps
        st.session_state["summary"] = summary
        st.session_state["detected_language"] = transcription_result.get("language", "inconnue")

    except Exception as e:
        logger.exception("Erreur durant le traitement")
        st.error(str(e))
    finally:
        # nettoyage des fichiers temporaires
        for path in (input_path, converted_audio_path):
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    logger.warning("Impossible de supprimer le fichier temporaire %s", path)
        st.session_state["processing"] = False

# ---------- Affichage des résultats ----------
if "transcript" in st.session_state:
    st.divider()
    st.header("3. Résultats")
    transcript = st.session_state.get("transcript", "")
    summary = st.session_state.get("summary", "")
    detected_language = st.session_state.get("detected_language", "inconnue")

    result_tab, transcript_tab = st.tabs(["📝 Compte rendu", "📄 Transcription"])
    with result_tab:
        st.markdown(summary)
        if st.button("⬇️ Télécharger le compte rendu (MD)", key="dl_md"):
            md = build_markdown_export(transcript=transcript, summary=summary, detected_language=detected_language)
            st.download_button(label="Télécharger (Markdown)", data=md, file_name="compte_rendu.md", mime="text/markdown", use_container_width=True)

    with transcript_tab:
        st.text_area("Transcription avec horodatage", value=transcript, height=500)
        if st.button("⬇️ Télécharger la transcription (SRT)", key="dl_srt"):
            srt = segments_to_srt(st.session_state.get("segments", [])) if st.session_state.get("segments") else ""
            if not srt:
                # reconstruire SRT à partir du texte si les segments manquent
                # ici, on utilise la transcription brut (sans timestamps) comme fallback minimal
                srt = transcript
            st.download_button(label="Télécharger (SRT)", data=srt, file_name="transcription.srt", mime="text/plain", use_container_width=True)
