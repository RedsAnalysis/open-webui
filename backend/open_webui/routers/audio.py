import hashlib
import json
import logging
import os
import uuid
from functools import lru_cache
from pathlib import Path
from pydub import AudioSegment
from pydub.silence import split_on_silence

import aiohttp
import aiofiles
import requests
import mimetypes

from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
    status,
    APIRouter,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from fastapi import Query
from typing import Optional  

from open_webui.utils.auth import get_admin_user, get_verified_user
from open_webui.config import (
    WHISPER_MODEL_AUTO_UPDATE,
    WHISPER_MODEL_DIR,
    CACHE_DIR,
)

from open_webui.constants import ERROR_MESSAGES
from open_webui.env import (
    AIOHTTP_CLIENT_TIMEOUT,
    ENV,
    SRC_LOG_LEVELS,
    DEVICE_TYPE,
    ENABLE_FORWARD_USER_INFO_HEADERS,
)


router = APIRouter()

# Constants
MAX_FILE_SIZE_MB = 25
MAX_FILE_SIZE = MAX_FILE_SIZE_MB * 1024 * 1024  # Convert MB to bytes
AZURE_MAX_FILE_SIZE_MB = 200
AZURE_MAX_FILE_SIZE = AZURE_MAX_FILE_SIZE_MB * 1024 * 1024  # Convert MB to bytes

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["AUDIO"])

SPEECH_CACHE_DIR = CACHE_DIR / "audio" / "speech"
SPEECH_CACHE_DIR.mkdir(parents=True, exist_ok=True)


##########################################
#
# Utility functions
#
##########################################

from pydub import AudioSegment
from pydub.utils import mediainfo


def get_audio_format(file_path):
    """Check if the given file needs to be converted to a different format."""
    if not os.path.isfile(file_path):
        log.error(f"File not found: {file_path}")
        return False

    info = mediainfo(file_path)
    if (
        info.get("codec_name") == "aac"
        and info.get("codec_type") == "audio"
        and info.get("codec_tag_string") == "mp4a"
    ):
        return "mp4"
    elif info.get("format_name") == "ogg":
        return "ogg"
    elif info.get("format_name") == "matroska,webm":
        return "webm"
    return None


def convert_audio_to_wav(file_path, output_path, conversion_type):
    """Convert MP4/OGG audio file to WAV format."""
    audio = AudioSegment.from_file(file_path, format=conversion_type)
    audio.export(output_path, format="wav")
    log.info(f"Converted {file_path} to {output_path}")


def set_faster_whisper_model(model: str, auto_update: bool = False):
    whisper_model = None
    if model:
        from faster_whisper import WhisperModel

        faster_whisper_kwargs = {
            "model_size_or_path": model,
            "device": DEVICE_TYPE if DEVICE_TYPE and DEVICE_TYPE == "cuda" else "cpu",
            "compute_type": "int8",
            "download_root": WHISPER_MODEL_DIR,
            "local_files_only": not auto_update,
        }

        try:
            whisper_model = WhisperModel(**faster_whisper_kwargs)
        except Exception:
            log.warning(
                "WhisperModel initialization failed, attempting download with local_files_only=False"
            )
            faster_whisper_kwargs["local_files_only"] = False
            whisper_model = WhisperModel(**faster_whisper_kwargs)
    return whisper_model


##########################################
#
# Audio API
#
##########################################


class TTSConfigForm(BaseModel):
    OPENAI_API_BASE_URL: str
    OPENAI_API_KEY: str
    # Added by RED: Configuration fields for CustomTTS OpenAPI compatible endpoint
    CUSTOMTTS_OPENAPI_BASE_URL: str = "" # Add default empty string
    CUSTOMTTS_OPENAPI_KEY: str = ""    # Add default empty string

    API_KEY: str
    ENGINE: str
    MODEL: str
    VOICE: str
    SPLIT_ON: str
    AZURE_SPEECH_REGION: str
    AZURE_SPEECH_OUTPUT_FORMAT: str


class STTConfigForm(BaseModel):
    OPENAI_API_BASE_URL: str
    OPENAI_API_KEY: str
    ENGINE: str
    MODEL: str
    WHISPER_MODEL: str
    DEEPGRAM_API_KEY: str
    AZURE_API_KEY: str
    AZURE_REGION: str
    AZURE_LOCALES: str


class AudioConfigUpdateForm(BaseModel):
    tts: TTSConfigForm
    stt: STTConfigForm


@router.get("/config")
async def get_audio_config(request: Request, user=Depends(get_admin_user)):
    return {
        "tts": {
            "OPENAI_API_BASE_URL": request.app.state.config.TTS_OPENAI_API_BASE_URL,
            "OPENAI_API_KEY": request.app.state.config.TTS_OPENAI_API_KEY,
            # Added by RED: Return CustomTTS OpenAPI config
            "CUSTOMTTS_OPENAPI_BASE_URL": getattr(request.app.state.config, 'CUSTOMTTS_OPENAPI_BASE_URL', ''),
            "CUSTOMTTS_OPENAPI_KEY": getattr(request.app.state.config, 'CUSTOMTTS_OPENAPI_KEY', ''),

            "API_KEY": request.app.state.config.TTS_API_KEY,
            "ENGINE": request.app.state.config.TTS_ENGINE,
            "MODEL": request.app.state.config.TTS_MODEL,
            "VOICE": request.app.state.config.TTS_VOICE,
            "SPLIT_ON": request.app.state.config.TTS_SPLIT_ON,
            "AZURE_SPEECH_REGION": request.app.state.config.TTS_AZURE_SPEECH_REGION,
            "AZURE_SPEECH_OUTPUT_FORMAT": request.app.state.config.TTS_AZURE_SPEECH_OUTPUT_FORMAT,
        },
        "stt": {
            "OPENAI_API_BASE_URL": request.app.state.config.STT_OPENAI_API_BASE_URL,
            "OPENAI_API_KEY": request.app.state.config.STT_OPENAI_API_KEY,
            "ENGINE": request.app.state.config.STT_ENGINE,
            "MODEL": request.app.state.config.STT_MODEL,
            "WHISPER_MODEL": request.app.state.config.WHISPER_MODEL,
            "DEEPGRAM_API_KEY": request.app.state.config.DEEPGRAM_API_KEY,
            "AZURE_API_KEY": request.app.state.config.AUDIO_STT_AZURE_API_KEY,
            "AZURE_REGION": request.app.state.config.AUDIO_STT_AZURE_REGION,
            "AZURE_LOCALES": request.app.state.config.AUDIO_STT_AZURE_LOCALES,
        },
    }


@router.post("/config/update")
async def update_audio_config(
    request: Request, form_data: AudioConfigUpdateForm, user=Depends(get_admin_user)
):
    request.app.state.config.TTS_OPENAI_API_BASE_URL = form_data.tts.OPENAI_API_BASE_URL
    request.app.state.config.TTS_OPENAI_API_KEY = form_data.tts.OPENAI_API_KEY
    # Added by RED: Update CustomTTS OpenAPI config
    request.app.state.config.CUSTOMTTS_OPENAPI_BASE_URL = form_data.tts.CUSTOMTTS_OPENAPI_BASE_URL
    request.app.state.config.CUSTOMTTS_OPENAPI_KEY = form_data.tts.CUSTOMTTS_OPENAPI_KEY

    request.app.state.config.TTS_API_KEY = form_data.tts.API_KEY
    request.app.state.config.TTS_ENGINE = form_data.tts.ENGINE
    request.app.state.config.TTS_MODEL = form_data.tts.MODEL
    request.app.state.config.TTS_VOICE = form_data.tts.VOICE
    request.app.state.config.TTS_SPLIT_ON = form_data.tts.SPLIT_ON
    request.app.state.config.TTS_AZURE_SPEECH_REGION = form_data.tts.AZURE_SPEECH_REGION
    request.app.state.config.TTS_AZURE_SPEECH_OUTPUT_FORMAT = (
        form_data.tts.AZURE_SPEECH_OUTPUT_FORMAT
    )

    request.app.state.config.STT_OPENAI_API_BASE_URL = form_data.stt.OPENAI_API_BASE_URL
    request.app.state.config.STT_OPENAI_API_KEY = form_data.stt.OPENAI_API_KEY
    request.app.state.config.STT_ENGINE = form_data.stt.ENGINE
    request.app.state.config.STT_MODEL = form_data.stt.MODEL
    request.app.state.config.WHISPER_MODEL = form_data.stt.WHISPER_MODEL
    request.app.state.config.DEEPGRAM_API_KEY = form_data.stt.DEEPGRAM_API_KEY
    request.app.state.config.AUDIO_STT_AZURE_API_KEY = form_data.stt.AZURE_API_KEY
    request.app.state.config.AUDIO_STT_AZURE_REGION = form_data.stt.AZURE_REGION
    request.app.state.config.AUDIO_STT_AZURE_LOCALES = form_data.stt.AZURE_LOCALES

    if request.app.state.config.STT_ENGINE == "":
        request.app.state.faster_whisper_model = set_faster_whisper_model(
            form_data.stt.WHISPER_MODEL, WHISPER_MODEL_AUTO_UPDATE
        )

    return {
        "tts": {
            "OPENAI_API_BASE_URL": request.app.state.config.TTS_OPENAI_API_BASE_URL,
            "OPENAI_API_KEY": request.app.state.config.TTS_OPENAI_API_KEY,
            # Added by RED: Return updated CustomTTS OpenAPI config
            "CUSTOMTTS_OPENAPI_BASE_URL": getattr(request.app.state.config, 'CUSTOMTTS_OPENAPI_BASE_URL', ''),
            "CUSTOMTTS_OPENAPI_KEY": getattr(request.app.state.config, 'CUSTOMTTS_OPENAPI_KEY', ''),

            "API_KEY": request.app.state.config.TTS_API_KEY,
            "ENGINE": request.app.state.config.TTS_ENGINE,
            "MODEL": request.app.state.config.TTS_MODEL,
            "VOICE": request.app.state.config.TTS_VOICE,
            "SPLIT_ON": request.app.state.config.TTS_SPLIT_ON,
            "AZURE_SPEECH_REGION": request.app.state.config.TTS_AZURE_SPEECH_REGION,
            "AZURE_SPEECH_OUTPUT_FORMAT": request.app.state.config.TTS_AZURE_SPEECH_OUTPUT_FORMAT,
        },
        "stt": {
            "OPENAI_API_BASE_URL": request.app.state.config.STT_OPENAI_API_BASE_URL,
            "OPENAI_API_KEY": request.app.state.config.STT_OPENAI_API_KEY,
            "ENGINE": request.app.state.config.STT_ENGINE,
            "MODEL": request.app.state.config.STT_MODEL,
            "WHISPER_MODEL": request.app.state.config.WHISPER_MODEL,
            "DEEPGRAM_API_KEY": request.app.state.config.DEEPGRAM_API_KEY,
            "AZURE_API_KEY": request.app.state.config.AUDIO_STT_AZURE_API_KEY,
            "AZURE_REGION": request.app.state.config.AUDIO_STT_AZURE_REGION,
            "AZURE_LOCALES": request.app.state.config.AUDIO_STT_AZURE_LOCALES,
        },
    }


def load_speech_pipeline(request):
    from transformers import pipeline
    from datasets import load_dataset

    if request.app.state.speech_synthesiser is None:
        request.app.state.speech_synthesiser = pipeline(
            "text-to-speech", "microsoft/speecht5_tts"
        )

    if request.app.state.speech_speaker_embeddings_dataset is None:
        request.app.state.speech_speaker_embeddings_dataset = load_dataset(
            "Matthijs/cmu-arctic-xvectors", split="validation"
        )


@router.post("/speech")
async def speech(request: Request, user=Depends(get_verified_user)):
    body = await request.body()
    name = hashlib.sha256(
        body
        + str(request.app.state.config.TTS_ENGINE).encode("utf-8")
        + str(request.app.state.config.TTS_MODEL).encode("utf-8")
    ).hexdigest()

    file_path = SPEECH_CACHE_DIR.joinpath(f"{name}.mp3")
    file_body_path = SPEECH_CACHE_DIR.joinpath(f"{name}.json")

    # Check if the file already exists in the cache
    if file_path.is_file():
        return FileResponse(file_path)

    payload = None
    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception as e:
        log.exception(e)
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    if request.app.state.config.TTS_ENGINE == "openai":
        payload["model"] = request.app.state.config.TTS_MODEL

        try:
            timeout = aiohttp.ClientTimeout(total=AIOHTTP_CLIENT_TIMEOUT)
            async with aiohttp.ClientSession(
                timeout=timeout, trust_env=True
            ) as session:
                async with session.post(
                    url=f"{request.app.state.config.TTS_OPENAI_API_BASE_URL}/audio/speech",
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {request.app.state.config.TTS_OPENAI_API_KEY}",
                        **(
                            {
                                "X-OpenWebUI-User-Name": user.name,
                                "X-OpenWebUI-User-Id": user.id,
                                "X-OpenWebUI-User-Email": user.email,
                                "X-OpenWebUI-User-Role": user.role,
                            }
                            if ENABLE_FORWARD_USER_INFO_HEADERS
                            else {}
                        ),
                    },
                ) as r:
                    r.raise_for_status()

                    async with aiofiles.open(file_path, "wb") as f:
                        await f.write(await r.read())

                    async with aiofiles.open(file_body_path, "w") as f:
                        await f.write(json.dumps(payload))

            return FileResponse(file_path)

        except Exception as e:
            log.exception(e)
            detail = None

            try:
                if r.status != 200:
                    res = await r.json()

                    if "error" in res:
                        detail = f"External: {res['error'].get('message', '')}"
            except Exception:
                detail = f"External: {e}"

            raise HTTPException(
                status_code=getattr(r, "status", 500) if r else 500,
                detail=detail if detail else "Open WebUI: Server Connection Error",
            )

        # Added by RED: Logic for customTTS_openapi engine
    elif request.app.state.config.TTS_ENGINE == "customTTS_openapi":
        # Use the model specified in the request payload if provided, otherwise use the configured default
        # Ensure payload exists before accessing it
        if payload and "model" not in payload:
             payload["model"] = request.app.state.config.TTS_MODEL

        # Safely get custom URL and Key from config
        custom_tts_url = getattr(request.app.state.config, 'CUSTOMTTS_OPENAPI_BASE_URL', None)
        custom_tts_key = getattr(request.app.state.config, 'CUSTOMTTS_OPENAPI_KEY', None)

        # Check if the URL is configured
        if not custom_tts_url:
             log.error("CustomTTS OpenAPI URL not configured.")
             raise HTTPException(status_code=500, detail="CustomTTS OpenAPI URL not configured.")

        # Construct the full endpoint URL, ensuring no double slashes
        speech_endpoint_url = f"{custom_tts_url.rstrip('/')}/audio/speech"
        log.info(f"Attempting to contact CustomTTS OpenAPI endpoint: {speech_endpoint_url}")

        r = None # Initialize response variable for error handling
        try:
            timeout = aiohttp.ClientTimeout(total=AIOHTTP_CLIENT_TIMEOUT)
            async with aiohttp.ClientSession(
                timeout=timeout, trust_env=True # trust_env=True respects system proxy settings
            ) as session:
                # Prepare headers, including Authorization if key exists
                headers = {
                    "Content-Type": "application/json",
                    **( {"Authorization": f"Bearer {custom_tts_key}"} if custom_tts_key else {} ), # Add Auth header only if key is present
                    **(
                        { # Forward user info headers if enabled
                            "X-OpenWebUI-User-Name": user.name,
                            "X-OpenWebUI-User-Id": user.id,
                            "X-OpenWebUI-User-Email": user.email,
                            "X-OpenWebUI-User-Role": user.role,
                        }
                        if ENABLE_FORWARD_USER_INFO_HEADERS
                        else {}
                    ),
                }
                log.debug(f"CustomTTS Request Headers: { {k: (v[:10] + '...' if k == 'Authorization' and v else v) for k, v in headers.items()} }") # Log headers safely
                log.debug(f"CustomTTS Request Payload: {payload}")

                # Make the POST request
                async with session.post(
                    url=speech_endpoint_url,
                    json=payload, # Send the parsed JSON payload from the request body
                    headers=headers,
                ) as r:
                    log.debug(f"CustomTTS Response Status: {r.status}")
                    # Raise HTTP errors (4xx, 5xx)
                    r.raise_for_status()

                    # Save the received audio file
                    async with aiofiles.open(file_path, "wb") as f:
                        await f.write(await r.read())
                    log.info(f"Saved CustomTTS audio response to cache: {file_path}")

                    # Save the request payload for caching/debugging purposes (optional but good practice)
                    async with aiofiles.open(file_body_path, "w") as f:
                        await f.write(json.dumps(payload))

            # If successful, return the cached audio file
            return FileResponse(file_path)

        # Handle exceptions during the request or response processing
        except Exception as e:
            log.exception(f"Error contacting or processing response from customTTS_openapi endpoint ({speech_endpoint_url}): {e}")
            detail = None

            # Attempt to get more specific error details from the response if available
            try:
                if r is not None and r.status != 200:
                    # Try parsing standard error structures first
                    try:
                        res = await r.json()
                        if isinstance(res, dict):
                             if "error" in res: # OpenAI-like structure
                                 error_content = res['error']
                                 if isinstance(error_content, dict):
                                     detail = f"External CustomTTS: {error_content.get('message', 'Unknown error')}"
                                 else: # If error is just a string
                                     detail = f"External CustomTTS: {error_content}"
                             elif "detail" in res: # FastAPI-like structure
                                 detail = f"External CustomTTS: {res.get('detail', 'Unknown error')}"
                             else: # Fallback if structure is unknown but is JSON
                                 detail = f"External CustomTTS Error (Status {r.status}): {str(res)[:200]}"
                        else: # If response is JSON but not a dict
                             detail = f"External CustomTTS Error (Status {r.status}): {str(res)[:200]}"

                    except (json.JSONDecodeError, aiohttp.ClientResponseError):
                         # If response is not JSON or other response error
                         try:
                             error_text = await r.text()
                             detail = f"External CustomTTS Error (Status {r.status}): {error_text[:200]}" # Limit response text
                         except Exception as text_exc:
                             log.error(f"Could not read error text from CustomTTS response: {text_exc}")
                             detail = f"External CustomTTS Error (Status {r.status}): Could not read response body."
            except Exception as parse_exc:
                # If reading/parsing the error response itself fails
                log.error(f"Could not parse error response from customTTS_openapi: {parse_exc}")
                detail = f"External CustomTTS Error: {e}" # Use original exception message

            # Raise the final HTTP exception to the client
            raise HTTPException(
                status_code=getattr(r, "status", 500) if r is not None else 500, # Use response status code if available
                detail=detail if detail else f"Open WebUI: CustomTTS Server Connection Error ({e.__class__.__name__})",
            )
    # End Added by RED block

    elif request.app.state.config.TTS_ENGINE == "elevenlabs":
        voice_id = payload.get("voice", "")

        if voice_id not in get_available_voices(request):
            raise HTTPException(
                status_code=400,
                detail="Invalid voice id",
            )

        try:
            timeout = aiohttp.ClientTimeout(total=AIOHTTP_CLIENT_TIMEOUT)
            async with aiohttp.ClientSession(
                timeout=timeout, trust_env=True
            ) as session:
                async with session.post(
                    f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                    json={
                        "text": payload["input"],
                        "model_id": request.app.state.config.TTS_MODEL,
                        "voice_settings": {"stability": 0.5, "similarity_boost": 0.5},
                    },
                    headers={
                        "Accept": "audio/mpeg",
                        "Content-Type": "application/json",
                        "xi-api-key": request.app.state.config.TTS_API_KEY,
                    },
                ) as r:
                    r.raise_for_status()

                    async with aiofiles.open(file_path, "wb") as f:
                        await f.write(await r.read())

                    async with aiofiles.open(file_body_path, "w") as f:
                        await f.write(json.dumps(payload))

            return FileResponse(file_path)

        except Exception as e:
            log.exception(e)
            detail = None

            try:
                if r.status != 200:
                    res = await r.json()
                    if "error" in res:
                        detail = f"External: {res['error'].get('message', '')}"
            except Exception:
                detail = f"External: {e}"

            raise HTTPException(
                status_code=getattr(r, "status", 500) if r else 500,
                detail=detail if detail else "Open WebUI: Server Connection Error",
            )

    elif request.app.state.config.TTS_ENGINE == "azure":
        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception as e:
            log.exception(e)
            raise HTTPException(status_code=400, detail="Invalid JSON payload")

        region = request.app.state.config.TTS_AZURE_SPEECH_REGION
        language = request.app.state.config.TTS_VOICE
        locale = "-".join(request.app.state.config.TTS_VOICE.split("-")[:1])
        output_format = request.app.state.config.TTS_AZURE_SPEECH_OUTPUT_FORMAT

        try:
            data = f"""<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="{locale}">
                <voice name="{language}">{payload["input"]}</voice>
            </speak>"""
            timeout = aiohttp.ClientTimeout(total=AIOHTTP_CLIENT_TIMEOUT)
            async with aiohttp.ClientSession(
                timeout=timeout, trust_env=True
            ) as session:
                async with session.post(
                    f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1",
                    headers={
                        "Ocp-Apim-Subscription-Key": request.app.state.config.TTS_API_KEY,
                        "Content-Type": "application/ssml+xml",
                        "X-Microsoft-OutputFormat": output_format,
                    },
                    data=data,
                ) as r:
                    r.raise_for_status()

                    async with aiofiles.open(file_path, "wb") as f:
                        await f.write(await r.read())

                    async with aiofiles.open(file_body_path, "w") as f:
                        await f.write(json.dumps(payload))

                    return FileResponse(file_path)

        except Exception as e:
            log.exception(e)
            detail = None

            try:
                if r.status != 200:
                    res = await r.json()
                    if "error" in res:
                        detail = f"External: {res['error'].get('message', '')}"
            except Exception:
                detail = f"External: {e}"

            raise HTTPException(
                status_code=getattr(r, "status", 500) if r else 500,
                detail=detail if detail else "Open WebUI: Server Connection Error",
            )

    elif request.app.state.config.TTS_ENGINE == "transformers":
        payload = None
        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception as e:
            log.exception(e)
            raise HTTPException(status_code=400, detail="Invalid JSON payload")

        import torch
        import soundfile as sf

        load_speech_pipeline(request)

        embeddings_dataset = request.app.state.speech_speaker_embeddings_dataset

        speaker_index = 6799
        try:
            speaker_index = embeddings_dataset["filename"].index(
                request.app.state.config.TTS_MODEL
            )
        except Exception:
            pass

        speaker_embedding = torch.tensor(
            embeddings_dataset[speaker_index]["xvector"]
        ).unsqueeze(0)

        speech = request.app.state.speech_synthesiser(
            payload["input"],
            forward_params={"speaker_embeddings": speaker_embedding},
        )

        sf.write(file_path, speech["audio"], samplerate=speech["sampling_rate"])

        async with aiofiles.open(file_body_path, "w") as f:
            await f.write(json.dumps(payload))

        return FileResponse(file_path)


def transcribe(request: Request, file_path):
    log.info(f"transcribe: {file_path}")
    filename = os.path.basename(file_path)
    file_dir = os.path.dirname(file_path)
    id = filename.split(".")[0]

    if request.app.state.config.STT_ENGINE == "":
        if request.app.state.faster_whisper_model is None:
            request.app.state.faster_whisper_model = set_faster_whisper_model(
                request.app.state.config.WHISPER_MODEL
            )

        model = request.app.state.faster_whisper_model
        segments, info = model.transcribe(
            file_path,
            beam_size=5,
            vad_filter=request.app.state.config.WHISPER_VAD_FILTER,
        )
        log.info(
            "Detected language '%s' with probability %f"
            % (info.language, info.language_probability)
        )

        transcript = "".join([segment.text for segment in list(segments)])
        data = {"text": transcript.strip()}

        # save the transcript to a json file
        transcript_file = f"{file_dir}/{id}.json"
        with open(transcript_file, "w") as f:
            json.dump(data, f)

        log.debug(data)
        return data
    elif request.app.state.config.STT_ENGINE == "openai":
        audio_format = get_audio_format(file_path)
        if audio_format:
            os.rename(file_path, file_path.replace(".wav", f".{audio_format}"))
            # Convert unsupported audio file to WAV format
            convert_audio_to_wav(
                file_path.replace(".wav", f".{audio_format}"),
                file_path,
                audio_format,
            )

        r = None
        try:
            r = requests.post(
                url=f"{request.app.state.config.STT_OPENAI_API_BASE_URL}/audio/transcriptions",
                headers={
                    "Authorization": f"Bearer {request.app.state.config.STT_OPENAI_API_KEY}"
                },
                files={"file": (filename, open(file_path, "rb"))},
                data={"model": request.app.state.config.STT_MODEL},
            )

            r.raise_for_status()
            data = r.json()

            # save the transcript to a json file
            transcript_file = f"{file_dir}/{id}.json"
            with open(transcript_file, "w") as f:
                json.dump(data, f)

            return data
        except Exception as e:
            log.exception(e)

            detail = None
            if r is not None:
                try:
                    res = r.json()
                    if "error" in res:
                        detail = f"External: {res['error'].get('message', '')}"
                except Exception:
                    detail = f"External: {e}"

            raise Exception(detail if detail else "Open WebUI: Server Connection Error")

    elif request.app.state.config.STT_ENGINE == "deepgram":
        try:
            # Determine the MIME type of the file
            mime, _ = mimetypes.guess_type(file_path)
            if not mime:
                mime = "audio/wav"  # fallback to wav if undetectable

            # Read the audio file
            with open(file_path, "rb") as f:
                file_data = f.read()

            # Build headers and parameters
            headers = {
                "Authorization": f"Token {request.app.state.config.DEEPGRAM_API_KEY}",
                "Content-Type": mime,
            }

            # Add model if specified
            params = {}
            if request.app.state.config.STT_MODEL:
                params["model"] = request.app.state.config.STT_MODEL

            # Make request to Deepgram API
            r = requests.post(
                "https://api.deepgram.com/v1/listen",
                headers=headers,
                params=params,
                data=file_data,
            )
            r.raise_for_status()
            response_data = r.json()

            # Extract transcript from Deepgram response
            try:
                transcript = response_data["results"]["channels"][0]["alternatives"][
                    0
                ].get("transcript", "")
            except (KeyError, IndexError) as e:
                log.error(f"Malformed response from Deepgram: {str(e)}")
                raise Exception(
                    "Failed to parse Deepgram response - unexpected response format"
                )
            data = {"text": transcript.strip()}

            # Save transcript
            transcript_file = f"{file_dir}/{id}.json"
            with open(transcript_file, "w") as f:
                json.dump(data, f)

            return data

        except Exception as e:
            log.exception(e)
            detail = None
            if r is not None:
                try:
                    res = r.json()
                    if "error" in res:
                        detail = f"External: {res['error'].get('message', '')}"
                except Exception:
                    detail = f"External: {e}"
            raise Exception(detail if detail else "Open WebUI: Server Connection Error")

    elif request.app.state.config.STT_ENGINE == "azure":
        # Check file exists and size
        if not os.path.exists(file_path):
            raise HTTPException(status_code=400, detail="Audio file not found")

        # Check file size (Azure has a larger limit of 200MB)
        file_size = os.path.getsize(file_path)
        if file_size > AZURE_MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File size exceeds Azure's limit of {AZURE_MAX_FILE_SIZE_MB}MB",
            )

        api_key = request.app.state.config.AUDIO_STT_AZURE_API_KEY
        region = request.app.state.config.AUDIO_STT_AZURE_REGION
        locales = request.app.state.config.AUDIO_STT_AZURE_LOCALES

        # IF NO LOCALES, USE DEFAULTS
        if len(locales) < 2:
            locales = [
                "en-US",
                "es-ES",
                "es-MX",
                "fr-FR",
                "hi-IN",
                "it-IT",
                "de-DE",
                "en-GB",
                "en-IN",
                "ja-JP",
                "ko-KR",
                "pt-BR",
                "zh-CN",
            ]
            locales = ",".join(locales)

        if not api_key or not region:
            raise HTTPException(
                status_code=400,
                detail="Azure API key and region are required for Azure STT",
            )

        r = None
        try:
            # Prepare the request
            data = {
                "definition": json.dumps(
                    {
                        "locales": locales.split(","),
                        "diarization": {"maxSpeakers": 3, "enabled": True},
                    }
                    if locales
                    else {}
                )
            }
            url = f"https://{region}.api.cognitive.microsoft.com/speechtotext/transcriptions:transcribe?api-version=2024-11-15"

            # Use context manager to ensure file is properly closed
            with open(file_path, "rb") as audio_file:
                r = requests.post(
                    url=url,
                    files={"audio": audio_file},
                    data=data,
                    headers={
                        "Ocp-Apim-Subscription-Key": api_key,
                    },
                )

            r.raise_for_status()
            response = r.json()

            # Extract transcript from response
            if not response.get("combinedPhrases"):
                raise ValueError("No transcription found in response")

            # Get the full transcript from combinedPhrases
            transcript = response["combinedPhrases"][0].get("text", "").strip()
            if not transcript:
                raise ValueError("Empty transcript in response")

            data = {"text": transcript}

            # Save transcript to json file (consistent with other providers)
            transcript_file = f"{file_dir}/{id}.json"
            with open(transcript_file, "w") as f:
                json.dump(data, f)

            log.debug(data)
            return data

        except (KeyError, IndexError, ValueError) as e:
            log.exception("Error parsing Azure response")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to parse Azure response: {str(e)}",
            )
        except requests.exceptions.RequestException as e:
            log.exception(e)
            detail = None

            try:
                if r is not None and r.status_code != 200:
                    res = r.json()
                    if "error" in res:
                        detail = f"External: {res['error'].get('message', '')}"
            except Exception:
                detail = f"External: {e}"

            raise HTTPException(
                status_code=getattr(r, "status_code", 500) if r else 500,
                detail=detail if detail else "Open WebUI: Server Connection Error",
            )


def compress_audio(file_path):
    if os.path.getsize(file_path) > MAX_FILE_SIZE:
        file_dir = os.path.dirname(file_path)
        audio = AudioSegment.from_file(file_path)
        audio = audio.set_frame_rate(16000).set_channels(1)  # Compress audio
        compressed_path = f"{file_dir}/{id}_compressed.opus"
        audio.export(compressed_path, format="opus", bitrate="32k")
        log.debug(f"Compressed audio to {compressed_path}")

        if (
            os.path.getsize(compressed_path) > MAX_FILE_SIZE
        ):  # Still larger than MAX_FILE_SIZE after compression
            raise Exception(ERROR_MESSAGES.FILE_TOO_LARGE(size=f"{MAX_FILE_SIZE_MB}MB"))
        return compressed_path
    else:
        return file_path


@router.post("/transcriptions")
def transcription(
    request: Request,
    file: UploadFile = File(...),
    user=Depends(get_verified_user),
):
    log.info(f"file.content_type: {file.content_type}")

    supported_filetypes = ("audio/mpeg", "audio/wav", "audio/ogg", "audio/x-m4a")

    if not file.content_type.startswith(supported_filetypes):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.FILE_NOT_SUPPORTED,
        )

    try:
        ext = file.filename.split(".")[-1]
        id = uuid.uuid4()

        filename = f"{id}.{ext}"
        contents = file.file.read()

        file_dir = f"{CACHE_DIR}/audio/transcriptions"
        os.makedirs(file_dir, exist_ok=True)
        file_path = f"{file_dir}/{filename}"

        with open(file_path, "wb") as f:
            f.write(contents)

        try:
            try:
                file_path = compress_audio(file_path)
            except Exception as e:
                log.exception(e)

                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=ERROR_MESSAGES.DEFAULT(e),
                )

            data = transcribe(request, file_path)
            file_path = file_path.split("/")[-1]
            return {**data, "filename": file_path}
        except Exception as e:
            log.exception(e)

            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ERROR_MESSAGES.DEFAULT(e),
            )

    except Exception as e:
        log.exception(e)

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(e),
        )


def get_available_models(
    request: Request,
    verify_url: Optional[str] = Query(None, alias="verify_url"),
    verify_key: Optional[str] = Query(None, alias="verify_key")
) -> list[dict]:
    log.warning(f"--- GET_AVAILABLE_MODELS CALLED --- Backend TTS_ENGINE State: '{request.app.state.config.TTS_ENGINE}' Verify URL: '{verify_url}'")
    available_models = []
    tts_engine = request.app.state.config.TTS_ENGINE # Use saved engine state
    default_openai_models = [{"id": "tts-1"}, {"id": "tts-1-hd"}]
    default_custom_models = [{"id": "default-custom-model"}]

    if tts_engine == "openai":
        # --- UNCHANGED OpenAI LOGIC ---
        openai_base_url = getattr(request.app.state.config, 'TTS_OPENAI_API_BASE_URL', '')
        if openai_base_url and not openai_base_url.startswith("https://api.openai.com"):
            try:
                response = requests.get(
                    f"{openai_base_url.rstrip('/')}/audio/models",
                     headers={"Authorization": f"Bearer {request.app.state.config.TTS_OPENAI_API_KEY}"} if request.app.state.config.TTS_OPENAI_API_KEY else {},
                     timeout=5
                )
                response.raise_for_status()
                data = response.json()
                if isinstance(data, dict) and "models" in data and isinstance(data["models"], list):
                     model_list = data["models"]
                     available_models = [{"id": m.get("id"), "name": m.get("name", m.get("id"))} for m in model_list if m.get("id")]
                elif isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
                     model_list = data["data"]
                     available_models = [{"id": m.get("id"), "name": m.get("id")} for m in model_list if m.get("id")]
                elif isinstance(data, list):
                     available_models = [{"id": m.get("id"), "name": m.get("name", m.get("id"))} for m in data if isinstance(m, dict) and m.get("id")]
                else:
                     log.warning(f"Unrecognized model list format from custom OpenAI endpoint: {openai_base_url}. Using defaults.")
                     available_models = default_openai_models
                available_models = [m for m in available_models if m.get("id")]
                if not available_models:
                     log.warning(f"Empty/invalid model list from custom OpenAI endpoint: {openai_base_url}. Using defaults.")
                     available_models = default_openai_models
            except Exception as e:
                log.error(f"Error fetching models from custom OpenAI endpoint {openai_base_url}: {str(e)}. Using defaults.")
                available_models = default_openai_models
        else:
            available_models = default_openai_models
        # --- END UNCHANGED OpenAI LOGIC ---

    elif tts_engine == "customTTS_openapi":
        # --- MODIFIED CustomTTS Block ---
        is_verify_call = bool(verify_url)

        # Determine URL/Key: Use verify params IF provided, otherwise use saved config
        custom_base_url = verify_url if is_verify_call else getattr(request.app.state.config, 'CUSTOMTTS_OPENAPI_BASE_URL', None)
        custom_api_key = verify_key if is_verify_call else getattr(request.app.state.config, 'CUSTOMTTS_OPENAPI_KEY', None)

        if custom_base_url:
            models_url = f"{custom_base_url.rstrip('/')}/models" # Correct path
            try:
                log.debug(f"Attempting models fetch from {'VERIFY' if is_verify_call else 'CONFIG'} URL: {models_url}")
                headers = {"Authorization": f"Bearer {custom_api_key}"} if custom_api_key else {}
                response = requests.get(models_url, headers=headers, timeout=5)
                response.raise_for_status()
                data = response.json()

                # Parsing logic for {"data": [...]}
                if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
                     model_list = data["data"]
                     available_models = [{"id": m.get("id"), "name": m.get("id")} for m in model_list if m.get("id")]
                     log.info(f"{'VERIFY ' if is_verify_call else ''}Loaded {len(available_models)} models from {models_url}")
                else:
                     log.warning(f"Unrecognized model list format from {models_url}. Expected '{{data: [...]}}'. {'Verification indicates format issue.' if is_verify_call else 'Using default.'}")
                     # Return empty list OR raise error on verify format fail? Raising is clearer.
                     if is_verify_call:
                          raise HTTPException(status_code=400, detail=f"Verification failed: Unexpected response format from {models_url}.")
                     else:
                          available_models = default_custom_models

                available_models = [m for m in available_models if m.get("id")] # Ensure IDs exist
                if not available_models and not is_verify_call: # Only use default if not verifying
                    log.warning(f"Model list from {models_url} was empty/invalid. Using default.")
                    available_models = default_custom_models

                # If verifying, return immediately after successful fetch and parse
                if is_verify_call:
                    return available_models

            except Exception as e:
                error_message = f"Error during {'VERIFY' if is_verify_call else 'CONFIG'} fetch from {models_url}: {str(e)}."
                log.error(error_message)
                if is_verify_call:
                    # Re-raise exception for verification failure
                    detail = f"Verification failed: Could not fetch models. Error: {e.__class__.__name__}"
                    status_code = 400
                    if isinstance(e, requests.exceptions.HTTPError):
                        detail = f"Verification failed: Server at {models_url} responded with status {e.response.status_code}."
                        status_code = e.response.status_code
                    elif isinstance(e, requests.exceptions.ConnectionError):
                        detail = f"Verification failed: Could not connect to {models_url}."
                    elif isinstance(e, requests.exceptions.Timeout):
                         detail = f"Verification failed: Request to {models_url} timed out."
                    elif isinstance(e, json.JSONDecodeError):
                         detail = f"Verification failed: Invalid JSON response received from {models_url}."
                    raise HTTPException(status_code=status_code, detail=detail)
                else:
                    # Fallback to default only for normal config load errors
                    available_models = default_custom_models
        else:
             # Handle missing URL
             if is_verify_call:
                 # This case should ideally not happen if frontend requires URL for verify
                 raise HTTPException(status_code=400, detail="Verification failed: Base URL not provided.")
             log.warning(f"{'VERIFY: ' if is_verify_call else ''}CUSTOMTTS_OPENAPI_BASE_URL not configured. Using default model.")
             available_models = default_custom_models if not is_verify_call else []
        # --- END MODIFIED CustomTTS Block ---

    elif tts_engine == "elevenlabs":
        # --- UNCHANGED ElevenLabs logic ---
        try:
            elevenlabs_api_key = getattr(request.app.state.config, 'TTS_API_KEY', None)
            if not elevenlabs_api_key:
                log.warning("ElevenLabs engine selected but TTS_API_KEY not configured.")
                raise ValueError("ElevenLabs API Key not configured.")
            response = requests.get(
                "https://api.elevenlabs.io/v1/models",
                headers={ "xi-api-key": elevenlabs_api_key, "Content-Type": "application/json" },
                timeout=5,
            )
            response.raise_for_status()
            models = response.json()
            available_models = [{"name": model["name"], "id": model["model_id"]} for model in models]
            log.info(f"Loaded {len(available_models)} models from ElevenLabs.")
        except requests.RequestException as e:
            log.error(f"Error fetching ElevenLabs models: {str(e)}")
            available_models = []
        except Exception as e:
            log.error(f"Unexpected error fetching ElevenLabs models: {str(e)}")
            available_models = []
        # --- End UNCHANGED ElevenLabs logic ---

    return available_models


@router.get("/models")
async def get_models(request: Request, user=Depends(get_verified_user)):
    return {"models": get_available_models(request)}


def get_available_voices(
    request: Request, # Add type hint
    verify_url: Optional[str] = Query(None, alias="verify_url"),
    verify_key: Optional[str] = Query(None, alias="verify_key")
) -> dict: # Return type is dict {id: name}
    """Returns {voice_id: voice_name} dict"""
    log.warning(f"--- GET_AVAILABLE_VOICES CALLED --- Backend TTS_ENGINE State: '{request.app.state.config.TTS_ENGINE}' Verify URL: '{verify_url}'")

    # --- Verification Logic: Only runs if verify_url is provided ---
    if verify_url:
        # This block handles ONLY the verification request for a CustomTTS endpoint
        custom_base_url = verify_url
        custom_api_key = verify_key
        # Path for custom verify (based on user curl)
        voices_url = f"{custom_base_url.rstrip('/')}/audio/voices"
        try:
            log.debug(f"Attempting VERIFY voices fetch from URL: {voices_url}")
            headers = {"Authorization": f"Bearer {custom_api_key}"} if custom_api_key else {}
            response = requests.get(voices_url, headers=headers, timeout=5)
            response.raise_for_status()
            data = response.json()

            # Parse response specific to verify (customTTS format {"voices": ["id1", "id2", ...]})
            available_voices = {} # Initialize as dict
            if isinstance(data, dict) and "voices" in data and isinstance(data["voices"], list):
                voices_id_list = data["voices"]
                # Check if the list contains strings
                if all(isinstance(voice_id, str) for voice_id in voices_id_list):
                    # Create the {id: name} mapping, using the ID as the name
                    available_voices = {voice_id: voice_id for voice_id in voices_id_list}
                    log.info(f"VERIFY loaded {len(available_voices)} voices (from string list) from {voices_url}")
                # Check if it's unexpectedly a list of dictionaries
                elif all(isinstance(voice_dict, dict) for voice_dict in voices_id_list):
                     available_voices = {voice.get("id"): voice.get("name", voice.get("id")) for voice in voices_id_list if voice.get("id")}
                     log.info(f"VERIFY loaded {len(available_voices)} voices (from dict list) from {voices_url}")
                else:
                     log.warning(f"VERIFY Voice list from {voices_url} contains mixed types. Verification indicates format issue.")
                     raise HTTPException(status_code=400, detail=f"Verification failed: Unexpected voice list item format from {voices_url}.")

                # Verification successful (even if list is empty), return immediately
                return available_voices # <<<--- RETURN verification result directly
            else:
                 log.warning(f"Unrecognized VERIFY voice list format from {voices_url}. Expected '{{voices: [...]}}'. Verification indicates format issue.")
                 raise HTTPException(status_code=400, detail=f"Verification failed: Unexpected response format from {voices_url}.")

        except Exception as e:
            # Verification failed, raise HTTPException
            error_message = f"Error during VERIFY fetch from {voices_url}: {str(e)}."
            log.error(error_message)
            detail = f"Verification failed: Could not fetch voices. Error: {e.__class__.__name__}"
            status_code = 400
            if isinstance(e, requests.exceptions.HTTPError):
                detail = f"Verification failed: Server at {voices_url} responded with status {e.response.status_code}."
                status_code = e.response.status_code
            elif isinstance(e, requests.exceptions.ConnectionError):
                detail = f"Verification failed: Could not connect to {voices_url}."
            elif isinstance(e, requests.exceptions.Timeout):
                 detail = f"Verification failed: Request to {voices_url} timed out."
            elif isinstance(e, json.JSONDecodeError):
                 detail = f"Verification failed: Invalid JSON response received from {voices_url}."
            raise HTTPException(status_code=status_code, detail=detail) # <<<--- RAISE exception on verify fail

    # --- Normal Logic: Runs if verify_url is NOT provided ---
    else:
        available_voices = {} # Initialize for normal flow
        tts_engine = request.app.state.config.TTS_ENGINE # Use SAVED engine state
        default_openai_voices = {
            "alloy": "alloy", "echo": "echo", "fable": "fable",
            "onyx": "onyx", "nova": "nova", "shimmer": "shimmer",
        }
        default_custom_voices = {"default-voice": "Default Custom Voice"}

        if tts_engine == "openai":
            # --- UNCHANGED OpenAI logic ---
            openai_base_url = getattr(request.app.state.config, 'TTS_OPENAI_API_BASE_URL', '')
            if openai_base_url and not openai_base_url.startswith("https://api.openai.com"):
                try:
                    response = requests.get(
                        f"{openai_base_url.rstrip('/')}/audio/voices",
                        headers={"Authorization": f"Bearer {request.app.state.config.TTS_OPENAI_API_KEY}"} if request.app.state.config.TTS_OPENAI_API_KEY else {},
                        timeout=5
                    )
                    response.raise_for_status()
                    data = response.json()
                    if isinstance(data, dict) and "voices" in data and isinstance(data["voices"], list):
                         voices_list = data["voices"]
                         available_voices = {voice.get("id"): voice.get("name", voice.get("id")) for voice in voices_list if voice.get("id")}
                         if not available_voices:
                              log.warning(f"Empty voice list from custom OpenAI endpoint {openai_base_url}. Using defaults.")
                              available_voices = default_openai_voices
                    else:
                         log.warning(f"Unrecognized voice list format from custom OpenAI endpoint {openai_base_url}. Using defaults.")
                         available_voices = default_openai_voices
                except Exception as e:
                    log.error(f"Error fetching voices from custom OpenAI endpoint {openai_base_url}: {str(e)}. Using defaults.")
                    available_voices = default_openai_voices
            else:
                available_voices = default_openai_voices
            # --- End UNCHANGED OpenAI logic ---

        elif tts_engine == "customTTS_openapi":
             # --- CORRECTED CustomTTS logic (using SAVED config) ---
            custom_base_url = getattr(request.app.state.config, 'CUSTOMTTS_OPENAPI_BASE_URL', None)
            custom_api_key = getattr(request.app.state.config, 'CUSTOMTTS_OPENAPI_KEY', None)
            if custom_base_url:
                voices_url = f"{custom_base_url.rstrip('/')}/audio/voices" # Correct path
                try:
                    log.debug(f"Attempting voices fetch from CONFIG URL: {voices_url}")
                    headers = {"Authorization": f"Bearer {custom_api_key}"} if custom_api_key else {}
                    response = requests.get(voices_url, headers=headers, timeout=5)
                    response.raise_for_status()
                    data = response.json()

                    # Parsing logic for {"voices": ["id1", ...]}
                    if isinstance(data, dict) and "voices" in data and isinstance(data["voices"], list):
                        voices_id_list = data["voices"]
                        if all(isinstance(voice_id, str) for voice_id in voices_id_list):
                            available_voices = {voice_id: voice_id for voice_id in voices_id_list}
                            log.info(f"Loaded {len(available_voices)} voices (from string list) from {voices_url}")
                        elif all(isinstance(voice_dict, dict) for voice_dict in voices_id_list):
                            available_voices = {voice.get("id"): voice.get("name", voice.get("id")) for voice in voices_id_list if voice.get("id")}
                            log.info(f"Loaded {len(available_voices)} voices (from dict list) from {voices_url}")
                        else:
                            log.warning(f"Voice list from {voices_url} contains mixed types. Using default.")
                            available_voices = default_custom_voices

                        if not available_voices:
                            log.warning(f"Empty voice list received or processed from {voices_url}. Using default.")
                            available_voices = default_custom_voices
                    else:
                         log.warning(f"Unrecognized voice list format from {voices_url} (expected '{{voices: [...]}}'). Using default.")
                         available_voices = default_custom_voices
                except Exception as e:
                    log.error(f"Error fetching voices from {voices_url}: {str(e)}. Using default.")
                    available_voices = default_custom_voices
            else:
                 log.warning("CUSTOMTTS_OPENAPI_BASE_URL not configured. Using default voice.")
                 available_voices = default_custom_voices
            # --- End CORRECTED CustomTTS logic ---

        elif tts_engine == "elevenlabs":
             # --- UNCHANGED ElevenLabs logic ---
            try:
                elevenlabs_api_key = getattr(request.app.state.config, 'TTS_API_KEY', None)
                if not elevenlabs_api_key:
                    log.warning("ElevenLabs engine selected but TTS_API_KEY not configured.")
                    raise ValueError("ElevenLabs API Key not configured.")
                available_voices = get_elevenlabs_voices(api_key=elevenlabs_api_key) # Call cached helper
            except RuntimeError as e:
                log.error(f"Failed to get ElevenLabs voices: {e}")
                available_voices = {}
            except Exception as e:
                log.exception(f"Unexpected error getting ElevenLabs voices: {e}")
                available_voices = {}
             # --- End UNCHANGED ElevenLabs logic ---

        elif tts_engine == "azure":
             # --- UNCHANGED Azure logic ---
            try:
                region = request.app.state.config.TTS_AZURE_SPEECH_REGION
                api_key = request.app.state.config.TTS_API_KEY
                if not region or not api_key:
                    log.error("Azure TTS region or API key not configured.")
                    raise ValueError("Azure region/key not configured.")

                url = f"https://{region}.tts.speech.microsoft.com/cognitiveservices/voices/list"
                headers = {"Ocp-Apim-Subscription-Key": api_key}

                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                voices = response.json()

                for voice in voices:
                    available_voices[voice["ShortName"]] = f"{voice['DisplayName']} ({voice['Locale']})"
            except requests.RequestException as e:
                log.error(f"Error fetching Azure voices: {str(e)}")
                available_voices = {}
            except Exception as e:
                 log.exception(f"Unexpected error getting Azure voices: {e}")
                 available_voices = {}
             # --- End UNCHANGED Azure logic ---

        # The return for the normal logic path happens HERE
        return available_voices