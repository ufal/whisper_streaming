from setuptools import setup, find_packages

setup(
    name="whisper_streaming",
    version="0.1",
    packages=find_packages(),
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
