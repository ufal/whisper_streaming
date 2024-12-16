import io
import argparse
import asyncio
import numpy as np
import ffmpeg

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from whisper_online import asr_factory, add_shared_args

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--host", type=str, default='localhost')
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--warmup-file", type=str, dest="warmup_file", 
        help="The path to a speech audio wav file to warm up Whisper so that the very first chunk processing is fast. It can be e.g. https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav .")
add_shared_args(parser)
args = parser.parse_args()

# Initialize Whisper
asr, online = asr_factory(args)

# Load demo HTML for the root endpoint
with open("live_transcription.html", "r") as f:
    html = f.read()

@app.get("/")
async def get():
    return HTMLResponse(html)

# Streaming constants
SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLES_PER_SEC = SAMPLE_RATE * int(args.min_chunk_size)
BYTES_PER_SAMPLE = 2               # s16le = 2 bytes per sample
BYTES_PER_SEC = SAMPLES_PER_SEC * BYTES_PER_SAMPLE

async def start_ffmpeg_decoder():
    """
    Start an FFmpeg process in async streaming mode that reads WebM from stdin
    and outputs raw s16le PCM on stdout. Returns the process object.
    """
    process = (
        ffmpeg
        .input('pipe:0', format='webm')
        .output('pipe:1', format='s16le', acodec='pcm_s16le', ac=CHANNELS, ar=str(SAMPLE_RATE))
        .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
    )
    return process

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection opened.")

    ffmpeg_process = await start_ffmpeg_decoder()
    pcm_buffer = bytearray()

    # Continuously read decoded PCM from ffmpeg stdout in a background task
    async def ffmpeg_stdout_reader():
        nonlocal pcm_buffer
        loop = asyncio.get_event_loop()
        while True:
            try:
                chunk = await loop.run_in_executor(None, ffmpeg_process.stdout.read, 4096)
                if not chunk:  # FFmpeg might have closed
                    print("FFmpeg stdout closed.")
                    break

                pcm_buffer.extend(chunk)

                # Process in 3-second batches
                while len(pcm_buffer) >= BYTES_PER_SEC:
                    three_sec_chunk = pcm_buffer[:BYTES_PER_SEC]
                    del pcm_buffer[:BYTES_PER_SEC]

                    # Convert int16 -> float32
                    pcm_array = np.frombuffer(three_sec_chunk, dtype=np.int16).astype(np.float32) / 32768.0

                    # Send PCM data to Whisper
                    online.insert_audio_chunk(pcm_array)
                    transcription = online.process_iter()
                    buffer = online.to_flush(online.transcript_buffer.buffer)

                    # Return partial transcription results to the client
                    await websocket.send_json({
                        "transcription": transcription[2],
                        "buffer": buffer[2]
                    })
            except Exception as e:
                print(f"Exception in ffmpeg_stdout_reader: {e}")
                break

        print("Exiting ffmpeg_stdout_reader...")

    stdout_reader_task = asyncio.create_task(ffmpeg_stdout_reader())

    try:
        while True:
            # Receive incoming WebM audio chunks from the client
            message = await websocket.receive_bytes()
            # Pass them to ffmpeg via stdin
            ffmpeg_process.stdin.write(message)
            ffmpeg_process.stdin.flush()

    except WebSocketDisconnect:
        print("WebSocket connection closed.")
    except Exception as e:
        print(f"Error in websocket loop: {e}")
    finally:
        # Clean up ffmpeg and the reader task
        try:
            ffmpeg_process.stdin.close()
        except:
            pass
        stdout_reader_task.cancel()

        try:
            ffmpeg_process.stdout.close()
        except:
            pass

        ffmpeg_process.wait()
        
        
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("whisper_fastapi_online_server:app", host=args.host, port=args.port, reload=True)