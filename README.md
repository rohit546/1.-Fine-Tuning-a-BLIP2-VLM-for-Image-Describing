Assignment 06:
1. Fine-Tuning a Generative VLM for Image Describing
“Design, implement and evaluate a fine-tuning pipeline for a generative vision-language
transformer (e.g., BLIP-2 or Flamingo) to produce rich image descriptions and short visual
narratives.”
Project Overview
Modern VLMs (e.g., BLIP-2 or Flamingo) can be adapted not just to describe the contents of
an image in natural, detailed prose. In this project, you will:
1. Curate & preprocess a small image–description dataset (≤ 5 K pairs).
2. Fine-tune a pre-trained generative VLM, exploring different training and prompt-tuning
strategies.
3. Implement generative decoding (beam search, sampling) to control creativity vs.
accuracy.
4. Evaluate your model with automatic metrics and qualitative case studies.
5. Analyze failures and iterate on prompts/model settings.
Learning Objectives
• - Navigate multimodal data pipelines (images + text).
• - Customize transformer-based VLMs for new describing tasks.
• - Understand and apply prompt-tuning versus full fine-tuning.
• - Implement and compare decoding algorithms (beam, top-k, top-p, temperature).
• - Use BLEU, METEOR, ROUGE-L, SPICE and diversity metrics (e.g., self-BLEU, Distinct-n).
• - Perform thorough error analysis and iterate on model/design.
Prerequisites & Tools
• - PyTorch (+ Lightning optional) or Hugging Face Transformers.
• - GPU with ≥ 12 GB VRAM.
• - Libraries: transformers, datasets, torchvision, scikit-learn, NLTK or SacréBLEU.
• - Git/GitHub for version control.
Dataset Recommendation
Flickr8k: 8,092 images, 5 captions each (subsample to 8,000 pairs) –
https://www.kaggle.com/datasets/adityajn105/flickr8k
COCO-100K: select 10,000 image–caption pairs – https://cocodataset.org/#download
Detailed Task Breakdown
A. Data Module: download, preprocess, tokenize, and split dataset.

B. Model & Training Pipeline: load BLIP-2, apply fine-tuning strategies (full vs. prompt-
tuning vs. layer-freeze).

C. Generative Decoding: implement beam search, top-k, top-p sampling, temperature
control.
D. Evaluation Metrics: BLEU-4, METEOR, ROUGE-L, SPICE, Self-BLEU, Distinct-n.
E. Qualitative & Error Analysis: select 20 examples to analyze hallucinations, repetition,
omissions.
Deliverables & Timeline
• Complete reproduceable code
• Detailed blog post
• LinkedIn post
• GitHub code post
Success Criteria
Quantitative: BLEU-4 > 25, CIDEr > 60 on test subset.
Qualitative: descriptions mention ≥ 3 key elements and avoid hallucinations in ≥ 80% of
samples.
Code Quality: modular, well-commented, reproducible.

2. Multimodal “Ask-the-Image” Mini-App
You are to build a proof-of-concept “Ask-the-Image” mini-app with the following
capabilities:
1. Speech-to-Text Interface
- Record up to 10 seconds of user speech via a simple GUI or web page.
- Transcribe the audio to text using a pre-trained ASR model (e.g., Whisper-small).
2. Image Question-Answering Module
- Allow users to upload or capture an image.
- Load a compact generative VLM (e.g., BLIP-2 ViT-base + Flan-T5-small).
- Combine the transcribed question with the image and generate an answer.
3. Text-to-Speech Output
- Render the VLM’s answer on-screen.
- Use a TTS library (e.g., pyttsx3 or Google TTS) to speak the answer back.
4. Integration & UX
- Provide controls for recording, uploading, and playing the spoken response.
- Handle simple edge cases (e.g., “What color is the car?” vs. “How many cars?”).
Deliverables
1. Code Repository
- Modular scripts: `asr.py`, `qa.py`, `tts.py`, `app.py` or equivalent.
- `requirements.txt` and a clear README with setup/run instructions.
2. Demo Video (≤ 3 minutes)
- Show: recording speech, transcription, image upload, answer generation, and TTS
playback.
3. Blog post
- Architecture Diagram of your pipeline.
- Challenges & Solutions (e.g., noisy audio, ambiguous queries, TTS prosody).
- Metrics: ASR WER on 10 test utterances, QA accuracy on 10 image-question pairs,
average end-to-end latency.
4. LinkedIn post
5: Code on Github
