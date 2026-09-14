import os
import subprocess
import tempfile
from pathlib import Path
import requests
import streamlit as st
from faster_whisper import WhisperModel
from streamlit_mic_recorder import mic_recorder


#CONFIGURATION
st.set_page_config(page_title="Transcripteur IA local", page_icon="🎙  ",layout="wide",)
OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_OLLAMA_MODEL = "qwen2.5:3b" #"qwen2.5-coder:3b" #"qwen2.5:7b"


# ROUTINES
def get_file_extension(filename: str) -> str:
    """ Retourne l'extension d'un fichier en minuscules. """
    return Path(filename).suffix.lower()
def save_uploaded_file(uploaded_file) -> str:
    """ Sauvegarde un fichier Streamlit dans un fichier temporaire. Retourne le chemin du fichier. """
    suffix = get_file_extension(uploaded_file.name)
    temporary_file = tempfile.NamedTemporaryFile(
    delete=False,
    suffix=suffix)
    temporary_file.write(uploaded_file.getbuffer())
    temporary_file.close()
    return temporary_file.name

def save_audio_bytes(audio_bytes: bytes, extension: str = ".wav") -> str:
    """ Sauvegarde des octets audio dans un fichier temporaire. """
    temporary_file = tempfile.NamedTemporaryFile(delete=False,suffix=extension)
    temporary_file.write(audio_bytes)
    temporary_file.close()
    return temporary_file.name

def extract_audio(input_path: str) -> str:
    """ Convertit un fichier audio ou vidéo en WAV mono 16 kHz. FFmpeg doit être installé sur le système. """
    output_file = tempfile.NamedTemporaryFile(delete=False,suffix=".wav")
    output_path = output_file.name
    output_file.close()
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

    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "FFmpeg n'est pas installé ou n'est pas accessible dans le PATH."
        )
    if result.returncode != 0:
        raise RuntimeError(
            "Erreur pendant la conversion audio :\n\n"
            + result.stderr[-3000:]
        )
    return output_path

@st.cache_resource
def load_whisper_model(model_size: str, device: str, compute_type: str):
    """ Charge le modèle Whisper une seule fois. """
    return WhisperModel(model_size, device=device, compute_type=compute_type,)

def transcribe_audio(audio_path: str, model_size: str, language: str | None, device: str, compute_type: str,):
    """ Transcrit un fichier audio avec faster-whisper. """
    model = load_whisper_model(model_size=model_size, device=device, compute_type=compute_type,)
    segments, info = model.transcribe(
    audio_path,
    language=language,
    beam_size=5,
    vad_filter=True,
    word_timestamps=False,
    )

    collected_segments = []
    full_text_parts = []

    for segment in segments:
        text = segment.text.strip()

        if not text:
            continue

        collected_segments.append({
            "start": segment.start,
            "end": segment.end,
            "text": text,
        })

        full_text_parts.append(text)

    full_text = " ".join(full_text_parts)

    return {
        "text": full_text,
        "segments": collected_segments,
        "language": info.language,
        "language_probability": info.language_probability,
    }
def format_timestamp(seconds: float) -> str:
    """ Convertit un nombre de secondes en HH:MM:SS. """
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    remaining_seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"

def format_transcript_with_timestamps(segments: list[dict]) -> str:
    """ Construit une transcription avec horodatage. """
    lines = []
    for segment in segments:
        start = format_timestamp(segment["start"])
        end = format_timestamp(segment["end"])
        lines.append(
            f"[{start} - {end}] {segment['text']}"
        )
    return "\n".join(lines)

def split_text(text: str, max_characters: int = 12000) -> list[str]:
    """ Découpe un texte en blocs raisonnables pour Ollama. """
    text = text.strip()
    if len(text) <= max_characters:
        return [text]

    paragraphs = text.split("\n")
    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        paragraph = paragraph.strip()

        if not paragraph:
            continue

        proposed_chunk = (
            current_chunk + "\n" + paragraph
        ).strip()

        if len(proposed_chunk) > max_characters:
            if current_chunk:
                chunks.append(current_chunk)

            current_chunk = paragraph
        else:
            current_chunk = proposed_chunk

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def ask_ollama(prompt: str,model_name: str,temperature: float = 0.2,) -> str:
    """ Appelle l'API locale d'Ollama. """
    payload = {"model": model_name,"prompt": prompt,"stream": False,"options": {"temperature": temperature,},}
    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=1800,
        )
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Impossible de contacter Ollama. "
            "Vérifiez qu'il est démarré avec la commande 'ollama serve'."
        )

    if response.status_code != 200:
        raise RuntimeError(
            f"Ollama a retourné une erreur HTTP {response.status_code} :\n"
            f"{response.text}"
        )

    data = response.json()

    if "response" not in data:
        raise RuntimeError(
            "Réponse inattendue d'Ollama :\n"
            + str(data)
        )
    return data["response"].strip()

def generate_summary(transcript: str, model_name: str, summary_type: str,) -> str:
    """ Génère un résumé ou un compte rendu avec Ollama. """
    chunks = split_text(transcript)
    if summary_type == "Résumé court":
        instruction = """
                Rédige un résumé court et clair en français. Conserve uniquement les informations importantes.
                Ne fais pas de liste d'actions si elles ne sont pas explicitement mentionnées.
                N'invente aucune information.
                """
    elif summary_type == "Compte rendu détaillé":
        instruction = """ Rédige un compte rendu détaillé en français avec les sections suivantes :
                            Compte rendu
                            Sujet principal
                            Synthèse
                            Points importants
                            Décisions prises
                            Actions à réaliser
                            Pour chaque action, indique le responsable et l'échéance uniquement si ces informations sont explicitement présentes.
                            Questions ou points à clarifier
                            N'invente aucune information absente de la transcription. Si une information n'est pas disponible, écris "Non précisé". """
    else:
        instruction = """ Analyse cette transcription et produis une synthèse structurée en français. Identifie :
                                • le sujet principal ;
                                • les idées importantes ;
                                • les décisions ;
                                • les actions ;
                                • les personnes citées ;
                                • les dates citées ;
                                • les points non résolus.
                            N'invente aucune information. """

    partial_results = []

    for index, chunk in enumerate(chunks):
        prompt = f"""
    Tu es un assistant spécialisé dans l'analyse de transcriptions audio.
    {instruction}
    Voici la partie {index + 1} de la transcription :
    ---------------- TRANSCRIPTION ---------------- {chunk} ---------------- FIN TRANSCRIPTION -------------
    Réponds uniquement avec le résultat demandé. """
        partial_result = ask_ollama(
            prompt=prompt,
            model_name=model_name,
            temperature=0.2,
        )

        partial_results.append(partial_result)

    if len(partial_results) == 1:
        return partial_results[0]

    combined = "\n\n".join(
        f"### Partie {index + 1}\n{result}"
        for index, result in enumerate(partial_results)
    )

    final_prompt = f"""
    Tu es un assistant chargé de fusionner plusieurs analyses d'une même transcription.
    Fusionne les éléments ci-dessous en un document cohérent en français. Supprime les répétitions. Ne crée aucune information qui n'existe pas dans les analyses.
    Utilise les sections suivantes :
    Compte rendu final
    Sujet principal
    Synthèse
    Points importants
    Décisions prises
    Actions à réaliser
    Questions ou points à clarifier
    Analyses à fusionner :
    {combined} """
    return ask_ollama(
        prompt=final_prompt,
        model_name=model_name,
        temperature=0.2,
    )

def build_markdown_export(transcript: str, summary: str, detected_language: str,) -> str:
    """ Prépare le contenu Markdown exportable. """
    return f"""# Transcription et compte rendu
    Langue détectée
    {detected_language}
    Compte rendu
    {summary}
    Transcription
    {transcript} """

## INTERFCE DE VISUALISATION STREAMLIT

st.title("🎙️ Transcripteur audio/vidéo avec IA")
st.caption("Whisper transcrit localement et Ollama génère le résumé ou le compte rendu.")
with st.sidebar: st.header("⚙️ Configuration")
whisper_model_size = st.selectbox(
    "Modèle Whisper",
    options=["tiny", "base", "small", "medium", "large-v3"],
    index=2,
    help=(
        "small est un bon compromis. "
        "medium ou large-v3 donnent de meilleurs résultats mais "
        "demandent davantage de ressources."
    ),
)

whisper_device = st.selectbox(
    "Appareil de calcul",
    options=["cpu", "cuda"],
    index=0,
)

if whisper_device == "cpu":
    whisper_compute_type = st.selectbox(
        "Type de calcul",
        options=["int8", "float32"],
        index=0,
    )
else:
    whisper_compute_type = st.selectbox(
        "Type de calcul",
        options=["float16", "int8_float16"],
        index=0,
    )

whisper_language_choice = st.selectbox(
    "Langue audio",
    options=[
        "Détection automatique",
        "Français",
        "Anglais",
        "Espagnol",
        "Allemand",
        "Italien",
    ],
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

ollama_model = st.text_input(
    "Modèle Ollama",
    value=DEFAULT_OLLAMA_MODEL,
)

summary_type = st.selectbox(
    "Type de résultat",
    options=[
        "Résumé court",
        "Compte rendu détaillé",
        "Analyse structurée",
    ],
    index=1,
)
st.header("1. Enregistrer ou importer un média")
recorded_audio = None
tab_record, tab_upload = st.tabs(["🎤 Enregistrer un audio", "📁 Importer un fichier"])
with tab_record:
    st.write("Cliquez sur le bouton pour enregistrer un message audio " "directement depuis votre navigateur.")
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
    uploaded_file = st.file_uploader("Sélectionnez un fichier audio ou vidéo", type=["wav","mp3","m4a","ogg","flac","webm","mp4","mov","avi","mkv",],)
if uploaded_file:
    st.write(
        f"Fichier sélectionné : **{uploaded_file.name}** "
        f"({uploaded_file.size / 1024 / 1024:.2f} Mo)"
    )

    extension = get_file_extension(uploaded_file.name)

    if extension in [".mp4",".mov",".avi",".mkv",".webm"]:
        st.video(uploaded_file)
    else:
        st.audio(uploaded_file)
st.divider()
st.header("2. Transcription et analyse")
process_button = st.button("🚀 Lancer la transcription et générer le compte rendu",type="primary",use_container_width=True,)
if process_button:
    input_path = None
    converted_audio_path = None
try:
    if recorded_audio and recorded_audio.get("bytes"):
        with st.spinner("Préparation de l'enregistrement..."):
            input_path = save_audio_bytes(
                recorded_audio["bytes"],
                extension=".wav",
            )

    elif uploaded_file:
        with st.spinner("Préparation du fichier..."):
            input_path = save_uploaded_file(uploaded_file)
    else:
        st.warning(
            "Veuillez enregistrer un audio ou importer un fichier."
        )
        st.stop()

    with st.status(
        "Traitement en cours...",
        expanded=True,
    ) as status:
        st.write("Conversion du média en WAV mono 16 kHz...")
        converted_audio_path = extract_audio(input_path)

        st.write(
            f"Transcription avec Whisper ({whisper_model_size})..."
        )

        transcription_result = transcribe_audio(
            audio_path=converted_audio_path,
            model_size=whisper_model_size,
            language=whisper_language,
            device=whisper_device,
            compute_type=whisper_compute_type,
        )

        transcript_with_timestamps = (
            format_transcript_with_timestamps(
                transcription_result["segments"]
            )
        )

        if not transcript_with_timestamps.strip():
            raise RuntimeError(
                "Aucun texte n'a été détecté dans l'enregistrement."
            )

        st.write(
            f"Langue détectée : "
            f"{transcription_result['language']}"
        )

        st.write(
            f"Analyse avec Ollama ({ollama_model})..."
        )

        summary = generate_summary(
            transcript=transcript_with_timestamps,
            model_name=ollama_model,
            summary_type=summary_type,
        )

        status.update(
            label="Traitement terminé",
            state="complete",
            expanded=False,
        )

    st.session_state["transcript"] = transcript_with_timestamps
    st.session_state["summary"] = summary
    st.session_state["detected_language"] = (
        transcription_result["language"]
    )

except Exception as error:
    st.error(str(error))

finally:
    for path in [input_path, converted_audio_path]:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
##============================================================
##Affichage des résultats
##============================================================
if "transcript" in st.session_state:
    st.divider()
    st.header("3. Résultats")
    transcript = st.session_state["transcript"]
summary = st.session_state["summary"]
detected_language = st.session_state["detected_language"]
result_tab, transcript_tab = st.tabs(
 ["📝 Compte rendu", "📄 Transcription"]
)
with result_tab:
    st.markdown(summary)

with transcript_tab:
    st.text_area(
     "Transcription avec horodatage",
     value=transcript,
     height=500,
                )

markdown_export = build_markdown_export(
    transcript=transcript,
    summary=summary,
    detected_language=detected_language,
)

st.download_button(
    label="⬇️ Télécharger le résultat en Markdown",
    data=markdown_export,
    file_name="compte_rendu.md",
    mime="text/markdown",
    use_container_width=True,
)
