from google import genai

client = genai.Client(api_key="AIzaSyAqZqaT4GUz84dHmjkRChsdDlSaGosVAFY")

# Test with different models
models_to_try = [
    'gemini-2.5-flash',
    'gemini-1.5-pro', 
    'gemini-1.0-pro',
    'gemini-pro'
]

for model_name in models_to_try:
    try:
        response = client.models.generate_content(
            model=model_name,
            contents="Say hello!"
        )
        print(f"✅ {model_name} works!")
        print(f"   Response: {response.text}\n")
        break
    except Exception as e:
        print(f"❌ {model_name} failed: {e}\n")  