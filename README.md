# 🛡️ IntegrityAI - Advanced AI Detection & Humanization Platform
[![Live Demo](https://img.shields.io/badge/Live-Demo-teal?style=for-the-badge)](https://integrity-ai-teal.vercel.app)

**IntegrityAI** is a full-stack forensic text analysis platform designed to bridge the trust gap in the age of generative AI. By combining **Transformer-based analysis**, **Keystroke Biometrics**, and **Deterministic NLP Paraphrasing**, it provides a complete ecosystem for content validation and authorship proof.

---

## 🚀 Key Functionality

### 1. 🕵️ AI Text Scanner (RoBERTa Engine)
* **Core Logic:** Utilizes the `RoBERTa-base-openai-detector` model via the Hugging Face Inference API to classify text as "Human-Written" or "AI-Generated."
* **X-Ray Analysis:** Deep-dive analysis that breaks down text sentence-by-sentence to pinpoint high-perplexity clusters that trigger detection flags.
* **Confidence Scoring:** Real-time probability metrics providing surgical precision (e.g., "98.5% AI Confidence").

### 2. ✍️ Smart Paraphraser (NLTK & WordNet)
* **Deterministic Mapping:** Employs a custom rule-based engine to handle high-frequency functional words ("so" → "therefore") to ensure context retention.
* **Synonym Replacement:** Leverages **NLTK** and **WordNet** for context-aware lexical substitution.
* **Tone Profiles:** Adaptive rewriting modes including **Standard**, **Fluent**, and **Formal**.

### 3. 📝 Live Writer (Proof of Authorship)
* **Keystroke Dynamics:** Monitors writing behavior, capturing metrics such as **WPM (Words Per Minute)**, **Backspace frequency**, and **Inter-key latency**.
* **Human Verification:** Sophisticated algorithms analyze writing "rhythm." Automated paste detection or zero-correction patterns trigger "Suspicious" flags, while natural editing generates a **Verified Human** certificate.

### 4. 🎮 "Fool the AI" Game Mode
* **Adversarial Challenge:** A gamified environment where users attempt to draft text that bypasses the detection engine for specific prompts.
* **Real-time Scoring:** Instant feedback loops scoring user ingenuity against the RoBERTa detection logic.

### 5. 🔐 Authentication & Persistent History
* **Supabase Auth:** Secure JWT-based sign-up and login functionality.
* **Cloud Ledger:** All forensic scans, paraphrases, and writing sessions are logged to a Supabase PostgreSQL database for cross-device access.

---

## 🛠️ Technical Tech Stack

### **Frontend (Visual Layer)**
* **Framework:** React.js (Vite)
* **Styling:** Tailwind CSS + Lucide React for a modern, forensic UI.
* **Networking:** Axios for high-concurrency API communication.

### **Backend (Intelligence Layer)**
* **Framework:** FastAPI (Python) for asynchronous, low-latency processing.
* **ML Pipeline:** Hugging Face Inference API (Serverless architecture).
* **NLP Tools:** NLTK (Natural Language Toolkit) & WordNet.
* **ASGI Server:** Uvicorn.

### **Infrastructure & DevOps**
* **Frontend Hosting:** Vercel
* **Backend Hosting:** Render (Python 3.10+)
* **Database/Auth:** Supabase (PostgreSQL)

---

## 🏁 Installation & Development

1.  **Clone the Repository**:
    ```bash
    git clone [https://github.com/your-username/integrity-ai.git](https://github.com/your-username/integrity-ai.git)
    ```
2.  **Frontend Setup**:
    ```bash
    cd frontend && npm install && npm run dev
    ```
3.  **Backend Setup**:
    ```bash
    cd backend && pip install -r requirements.txt
    uvicorn main:app --reload
    ```

---
**Live Demo:** [https://integrity-ai-teal.vercel.app](https://integrity-ai-teal.vercel.app)

