# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 15:39:29 2024

@author: Fatemeh Dalilian
"""

import torch
from datasets import load_dataset
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, GPT2LMHeadModel, GPT2Tokenizer, pipeline
import torchaudio
import soundfile as sf
from pydub import AudioSegment

# Convert .m4a to .wav
def convert_m4a_to_wav(m4a_file, wav_file):
    audio = AudioSegment.from_file(m4a_file, format="m4a")
    audio.export(wav_file, format="wav")

# Load pre-trained ASR model and processor
def load_asr_model():
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    return processor, model

# Enhance transcription with LLM
def enhance_with_gpt(transcription):
    tokenizer_gpt = GPT2Tokenizer.from_pretrained("gpt2")
    model_gpt = GPT2LMHeadModel.from_pretrained("gpt2")

    input_ids = tokenizer_gpt.encode(transcription, return_tensors='pt')
    gpt_output = model_gpt.generate(input_ids, max_length=500)
    context_enhanced_text = tokenizer_gpt.decode(gpt_output[0], skip_special_tokens=True)
    return context_enhanced_text

# TTS setup
def tts_setup():
    tts = pipeline("text-to-speech", model="facebook/fastspeech2-en-ljspeech")
    return tts

# ASR to TTS with LLM integration
def asr_to_tts_with_llm(audio_file, processor, asr_model, tts):
    print("Processing audio file through ASR, LLM, and TTS...")
    
    # Convert .m4a to .wav if necessary
    if audio_file.endswith(".m4a"):
        wav_file = audio_file.replace(".m4a", ".wav")
        convert_m4a_to_wav(audio_file, wav_file)
        audio_file = wav_file
    
    # Read the .wav file
    audio_input, sample_rate = sf.read(audio_file)
    input_values = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_values
    logits = asr_model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])

    # Enhance transcription with LLM
    enhanced_transcription = enhance_with_gpt(transcription)

    # Convert enhanced transcription to speech
    speech_output = tts(enhanced_transcription)[0]
    speech_data = speech_output["array"]
    sample_rate = speech_output["sample_rate"]

    # Save the generated speech as a .wav file
    output_path = "output_speech.wav"
    sf.write(output_path, speech_data, sample_rate)
    print(f"Saved the generated speech to {output_path}")

# Load and preprocess the dataset for demonstration purposes
def load_and_preprocess_dataset():
    print("Loading and preprocessing dataset...")
    dataset = load_dataset("librispeech_asr", "clean", split="train.100")

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    def preprocess_function(examples):
        audio = examples["audio"]
        examples["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
        examples["labels"] = processor.tokenizer(examples["text"]).input_ids
        return examples

    processed_dataset = dataset.map(preprocess_function, remove_columns=dataset.column_names)
    return processed_dataset

if __name__ == "__main__":
    # Load pre-trained ASR model and processor
    processor, asr_model = load_asr_model()
    
    # Load and preprocess the dataset (demonstration purposes)
    processed_dataset = load_and_preprocess_dataset()
    
    # Setup TTS
    tts = tts_setup()

    # Example usage
    asr_to_tts_with_llm("/content/Recording.m4a", processor, asr_model, tts)
