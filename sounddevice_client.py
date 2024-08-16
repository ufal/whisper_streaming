"""
Simple (hopefully) multiplatform client mimicing the linux command 
    
    arecord -f S16_LE -c1 -r 16000 -t raw -D default | nc localhost 43001

which streams audio data from the microphone to the server.
Tested on Mac Os.
"""
import sys
import sounddevice as sd
import socket
import numpy as np
from argparse import ArgumentParser


parser = ArgumentParser(__doc__)
parser.add_argument('--host', type=str, default='localhost', help='Host name')
parser.add_argument('--port', type=int, default=43007, help='Port number')
parser.add_argument('--chunk', type=int, default=1000, help='Chunk size in ms')
args = parser.parse_args()



# Audio configuration needed for whisper_online_server.py
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK = 1024
# SIGNED INT16 LITTLE ENDIAN is setup as for sounddevice
DTYPE = np.int16

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((args.host, args.port))

print("Recording and streaming audio...")

def callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    # Convert the audio data to bytes and send it over the socket
    sock.sendall(indata.tobytes())

try:
    # Open the audio stream
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16', callback=callback, blocksize=CHUNK):
        print("Press Ctrl+C to stop the recording")
        while True:
            sd.sleep(args.chunk)
except KeyboardInterrupt:
    print("Stopping...")
finally:
    # Close the socket
    sock.close()
