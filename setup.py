from setuptools import setup, find_packages

setup(
    name="whisper_streaming",
    version="0.2.1",
    packages=find_packages(where="whisper_streaming"),
    include_package_data=True,
    py_modules=[
      "whisper_streaming.whisper_online",
      "whisper_streaming.whisper_online_server",
      "whisper_streaming.silero_vad_iterator",
      "whisper_streaming.__init__",
    ],
    license="MIT",
    author="Some Author",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "torch",
        "torchaudio",
        "faster-whisper",
        "librosa",
        "soundfile",
    ],
)
