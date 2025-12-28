import torch
try:
    from transformers import pipeline
except ModuleNotFoundError:
    print("ERROR: The 'transformers' library is not installed. Please install it using 'pip install transformers'.")
    exit(1)

def launch_integrity_ai():
    # 1. Path to your unzipped folder
    model_path = "./integrity_ai_final_model"

    # 2. Automatically detect your RTX 4050 (device 0)
    device = 0 if torch.cuda.is_available() else -1
    print(f"🚀 Launching on: {'GPU (RTX 4050)' if device == 0 else 'CPU'}")

    # 3. Load the model pipeline
    classifier = pipeline("text-classification", model=model_path, device=device)

    # 4. Test it!
    test_title = "Modern AI Trends"
    test_abstract = "The rapid development of large language models has significantly changed the way we approach academic writing."

    result = classifier(f"{test_title} {test_abstract}")[0]

    label = "AI-Generated" if result['label'] == 'LABEL_1' else "Human-Written"
    print(f"\nAnalysis Result: {label}")
    print(f"Confidence: {result['score']*100:.2f}%")

if __name__ == "__main__":
    launch_integrity_ai()