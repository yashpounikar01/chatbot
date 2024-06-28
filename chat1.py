import os
import speech_recognition as sr
import pyttsx3
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
import time 
# Set environment variables to suppress TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations

# Initialize the BlenderBot model and tokenizer
tokenizer = BlenderbotTokenizer.from_pretrained('facebook/blenderbot-400M-distill')
model = BlenderbotForConditionalGeneration.from_pretrained('facebook/blenderbot-400M-distill')

def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
        
    try:
        text = recognizer.recognize_google(audio)
        print(f"You said: {text}")
        return text
    except sr.UnknownValueError:
        print("Sorry, I did not understand that.")
        return None
    except sr.RequestError:
        print("Sorry, the speech service is down.")
        return None

def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def get_response(prompt):
    inputs = tokenizer(prompt, return_tensors='pt')
    reply_ids = model.generate(**inputs)
    response = tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]
    return response

def main():
    while True:
        text = recognize_speech()
        if text:
            response = get_response(text)
            print(f"Chatbot says: {response}")
            speak_text(response)
        time.sleep(1)

if __name__ == "__main__":
    main()
