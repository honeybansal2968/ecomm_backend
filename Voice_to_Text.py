import speech_recognition as sr

r = sr.Recognizer()
def speech_to_text(audio_file):

  with sr.AudioFile(audio_file) as source:
    audio_text = r.listen(source)
    try:
        text = r.recognize_google(audio_text)
        print('Converting audio transcripts into text ...')
    except:

         print('Sorry.. run again...')

    return text