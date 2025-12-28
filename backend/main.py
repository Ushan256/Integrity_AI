from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline, GPT2LMHeadModel, GPT2TokenizerFast
import torch
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize, word_tokenize
import random
import re

# --- 1. SETUP & DOWNLOADS ---
print("⏳ CHECKING DICTIONARY DATA...")
resources = ['wordnet', 'omw-1.4', 'averaged_perceptron_tagger', 'punkt', 'punkt_tab']
for res in resources:
    try:
        nltk.data.find(f'corpora/{res}' if 'wordnet' in res else f'tokenizers/{res}')
    except LookupError:
        print(f"⬇️ Downloading {res}...")
        nltk.download(res)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. CONFIG & AI MODELS ---
device = 0 if torch.cuda.is_available() else -1
torch_device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ RUNNING ON: {torch_device.upper()}")

try:
    classifier = pipeline("text-classification", model="roberta-large-openai-detector", device=device)
    perplexity_model = GPT2LMHeadModel.from_pretrained("gpt2").to(torch_device)
    perplexity_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
except Exception as e:
    print(f"⚠️ Model Warning: {e}")
    classifier = None
    perplexity_model = None

# --- 3. SMART GRAMMAR MAP (High Quality Replacements) ---
# Used for words that usually don't have good dictionary synonyms
FUNCTIONAL_SYNONYMS = {
    "when": ["while", "at the time", "during which", "as soon as"],
    "as": ["since", "because", "while", "in the role of"],
    "the": ["this", "that", "said", "the specific"],
    "if": ["provided that", "assuming", "in case", "whether"],
    "but": ["however", "although", "yet", "nevertheless"],
    "and": ["plus", "along with", "as well as", "together with"],
    "or": ["alternatively", "conversely", "on the other hand"],
    "because": ["since", "due to the fact", "as", "owing to"],
    "so": ["therefore", "thus", "consequently", "hence"],
    "with": ["alongside", "using", "accompanied by"],
    "without": ["lacking", "void of", "minus", "free from"],
    "to": ["towards", "in the direction of", "until"],
    "for": ["on behalf of", "in favor of", "intended for"],
    "is": ["remains", "exists as", "constitutes", "represents"],
    "are": ["remain", "exist as", "constitute", "represent"],
    "was": ["remained", "existed as", "constituted"],
    "very": ["extremely", "highly", "exceedingly", "truly"],
    "good": ["excellent", "beneficial", "favorable", "positive"], # Common vague words fix
    "bad": ["negative", "detrimental", "poor", "adverse"],
    "use": ["utilize", "employ", "apply", "leverage"]
}

# --- 4. DATA MODELS ---
class ParaphraseRequest(BaseModel):
    content: str
    tone: str = "Standard" 

class SentenceRequest(BaseModel):
    sentence: str
    tone: str

class WordRequest(BaseModel):
    word: str

class TextRequest(BaseModel):
    title: str = ""
    content: str

# --- 5. SMART SYNONYM LOGIC ---
def get_synonyms(word, tone="Standard", strict=True):
    word_lower = word.lower()
    synonyms = set()

    # A. Check Manual Map (Best Quality for Functional Words)
    if word_lower in FUNCTIONAL_SYNONYMS:
        return FUNCTIONAL_SYNONYMS[word_lower]

    # B. Check WordNet (With Quality Filters)
    # 1. Get Synsets
    synsets = wordnet.synsets(word)
    
    # 2. QUALITY FILTER: Only look at the top 3 most common definitions.
    #    This avoids getting "slope" for "bank" (money) if "financial" is the primary meaning.
    most_common_synsets = synsets[:3] 

    for syn in most_common_synsets:
        for lemma in syn.lemmas():
            candidate = lemma.name().replace("_", " ")
            
            # Skip exact matches
            if candidate.lower() == word_lower: continue
            
            # Strict Tone Filtering
            if strict:
                if tone == "Fluent" or tone == "Standard":
                    # For standard text, avoid very long/complex academic words
                    if len(candidate) > len(word) + 5: continue
                elif tone == "Formal":
                    # For formal text, avoid very short/slang words
                    if len(candidate) < 4: continue 
            
            synonyms.add(candidate)
    
    # Sort by length to give variety
    return sorted(list(synonyms), key=len)

def rewrite_sentence_logic(sentence, tone, variance_level=0.5):
    words = word_tokenize(sentence)
    new_words = []
    
    for word in words:
        clean_word = re.sub(r'[^\w\s]', '', word)
        word_lower = clean_word.lower()

        # Check if we should swap this word
        is_functional = word_lower in FUNCTIONAL_SYNONYMS
        
        # Don't swap short words unless they are in our special grammar map
        if len(clean_word) < 3 and not is_functional:
            new_words.append(word)
            continue
        
        # Swap Probability
        swap_prob = 0.4
        if tone == "Fluent": swap_prob = 0.5
        
        if random.random() < (swap_prob * variance_level):
            syns = get_synonyms(clean_word, tone, strict=True)
            if syns:
                replacement = random.choice(syns)
                # Maintain Capitalization
                if word[0].isupper(): replacement = replacement.capitalize()
                new_words.append(replacement)
            else:
                new_words.append(word)
        else:
            new_words.append(word)
    
    reconstructed = " ".join(new_words)
    return re.sub(r'\s+([?.!,:;])', r'\1', reconstructed)

# --- 6. ENDPOINTS ---

@app.post("/paraphrase")
async def paraphrase_text(data: ParaphraseRequest):
    sentences = sent_tokenize(data.content)
    rewritten_sentences = []
    for sent in sentences:
        rewritten_sentences.append(rewrite_sentence_logic(sent, data.tone, variance_level=1.5))
    return {"paraphrased": " ".join(rewritten_sentences)}

@app.post("/rewrite_sentence")
async def fetch_sentence_variants(data: SentenceRequest):
    variants = []
    # Try 5 times to get 3 unique meaningful variations
    for _ in range(5):
        var = rewrite_sentence_logic(data.sentence, data.tone, variance_level=2.5)
        if var not in variants and var != data.sentence:
            variants.append(var)
            
    if not variants: variants = ["No suitable variations found."]
    return {"variants": variants[:3]}

@app.post("/synonyms")
async def fetch_synonyms(data: WordRequest):
    # strict=False allows seeing more options when user manually clicks
    syns = get_synonyms(data.word, "Standard", strict=False)
    # Return top 6 most relevant
    return {"synonyms": syns[:6]}

@app.post("/analyze")
async def analyze_text(data: TextRequest):
    if classifier is None: return {"prediction": "Error", "confidence": 0, "risk_level": "None"}
    
    try:
        # 1. RoBERTa Scan
        result = classifier(data.content)[0]
        score = result['score']
        label = result['label']
        
        # Label Mapping
        ai_prob = score * 100 if label in ['Fake', 'LABEL_0'] else (1 - score) * 100
        
        # 2. Perplexity Scan
        enc = perplexity_tokenizer(data.content, return_tensors="pt")
        inp = enc.input_ids.to(torch_device)
        with torch.no_grad():
            ppl = torch.exp(perplexity_model(inp, labels=inp).loss).item()

        # 3. Hybrid Decision Logic
        pred, risk, conf = "Human-Written", "Low", 100 - ai_prob
        
        if ai_prob > 80: pred, risk, conf = "AI-Generated", "High", ai_prob
        elif ppl < 25: pred, risk, conf = "Suspected AI (Modern)", "High", 88.5
        elif ppl < 45: pred, risk, conf = "Possible AI Edit", "Medium", 65.0
        
        return {"prediction": pred, "confidence": round(conf, 1), "risk_level": risk}
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
async def health():
    return {"status": "Ready"}