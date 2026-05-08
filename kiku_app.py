
########## Libraries import##############
import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from typing import Optional, Literal
import io
import tempfile
from audiorecorder import audiorecorder
import re
import hashlib

##########Fetching API key from Enviornment file########
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("API key not found. Check your .env file.")
client = OpenAI(api_key=OPENAI_API_KEY)

#st.write("API loaded:", bool(OPENAI_API_KEY))

##############################################################################
################ Speach - to- text - Speach Functions#########################
##############################################################################

def transcribe_audio(audio_bytes:bytes)->str:
    with tempfile.NamedTemporaryFile(suffix=".wav",delete = False) as tmp:
        tmp.write(audio_bytes)
        tmp_path=tmp.name
    with open(tmp_path,"rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model = "whisper-1",
            file=audio_file,
            language="en"
        )
    os.unlink(tmp_path)
    return transcript.text

def speak_response(text:str)->bytes:
    response = client.audio.speech.create(
        model = "tts-1",
        voice ="nova",
        input = text,
        speed =0.65

    )
    return response.content

#####################################
####### Dedection Class  ############
####### For Structured Output #######
#####################################
class detection(BaseModel):
  needs_correction: bool
  error_type: Optional[
        Literal[
            "subject_verb_agreement",
            "verb_tense",
            "plural_noun",
            "article_usage",
            "word_order",
            "word_choice",
            "other"
        ]
    ]
  corrected_sentence: Optional[str]

########################################
### Dedector and Classfier Function ###
#######################################
#######################################

def detect_grammar(sentence,conversation_history) -> detection:
    recent_context = conversation_history[-4:]

    user_prompt = f"""
Recent conversation:
{recent_context}

Child input:
{sentence}
"""

    system_prompt = f"""
You are a strict grammar detector for children's English.
Your only job is to check if the child's input needs a grammar correction and what type of error is made.

Rules:
- Do NOT be friendly.
- Do NOT teach grammar rules.
- Do NOT continue conversation.
- Do NOT add extra text.
- This app is for SPOKEN English only. 
   Ignore capitalization entirely. 
   'i am good' and 'I am good' are identical. 
   Never correct capitalization.
- If the input is acceptable for a child speaking English, treat it as correct.
- If it has a grammar mistake, provide the corrected sentence.
- If no correction is needed, error_type and corrected_sentence should be null.
- If the child input is a short conversational response, clarification, or question such as "what", "why", "huh", "yes", "no", "again", "I don't know", treat it as acceptable and do not correct it.
- Only correct if the child is clearly trying to say an English sentence.
- Do not correct short conversation replies or clarification requests.

Return ONLY structured output.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    response = client.responses.parse(
        model="gpt-4o-mini",
        input=messages,
        temperature=0,
        max_output_tokens=500,
        text_format=detection
    )

    return response.output_parsed

#####################################
####### Reponder Function ###########
#####################################
#####################################
def responder_fun(sentence: str, response_input, mode, conversation_history):
    
    # Extract last Kiku message for clarification mode
    last_kiku_message = ""
    for msg in reversed(conversation_history):
        if msg["role"] == "assistant":
            last_kiku_message = msg["content"]
            break

    # Build user_prompt based on mode
    if mode == "clarification":
        user_prompt = f"""
The child said: "{sentence}" — they did not understand your last message.

Your last message was:
"{last_kiku_message}"

Repeat your last message in simpler words.
Do NOT start a new topic.
Do NOT ask a new question.
Just say the same thing more simply.
"""
    else:
        if response_input is None:
            text = """
needs_correction: null
error_type: null
corrected_sentence: null
"""
        else:
            text = f"""
needs_correction: {response_input.needs_correction}
error_type: {response_input.error_type}
corrected_sentence: {response_input.corrected_sentence}
"""
        user_prompt = f"""
Child sentence: {sentence}

Detector result: {text}
"""

    system_prompt = f"""
You are Kiku, a friendly 8-year-old English buddy for children.

Personality:
- cheerful, curious, and a little playful
- talks like a friend, not a teacher
- uses very simple, natural English (short sentences)
- sounds excited and interested in what the child says
- sometimes reacts with fun expressions like "Oh wow!", "That's cool!", "Hehe!"
- keeps responses short (1–3 lines max)
- makes the child feel safe, confident, and happy

Style:
- speaks like a real kid (not formal, not robotic)
- uses everyday words only
- avoids long explanations
- may add a tiny fun idea or image (like animals, magic, or a silly situation) — max 1 short line

Strict Rules:
- NEVER use grammar terms (like verb, tense, article, subject)
- NEVER say "this is wrong"
- NEVER sound strict, serious, or judgmental
- NEVER give long explanations

Mode: {mode}

If mode = "correction":
- You MUST use the corrected_sentence provided in Detector result.
- NEVER repeat the original child sentence if needs_correction = true.
- ALWAYS say the corrected_sentence exactly as provided.
- Use phrases like: "You can say: …" or "Try this: …"
- Ask the child to try again in a friendly way.
- Do not ask a new topic question.

If mode = "conversation":
- briefly praise the child (1 short phrase)
- continue the conversation with ONE simple question

If mode = "clarification":
- look at your last assistant message
- if your last message was a correction, repeat the corrected sentence simply
- if your last message was a question, rephrase it more simply
- if your last message was a conversation, repeat it again simply
- do not introduce a new topic
- keep it very short
"""

    messages = (
        [{"role": "system", "content": system_prompt}] +
        conversation_history[-6:] +
        [{"role": "user", "content": user_prompt}]
    )

    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=messages,
        temperature=0.7,
        max_tokens=150
    )
    return response.choices[0].message.content



#########################################################
#########################################################
######## Main Body ######################################



#first_attempt = 0

#st.session_state.conversation_history = []
clarification_inputs = ["what", "huh", "what?", "huh?", "i don't know", "i dont know","repeat","repeat it"]
#st.session_state.correction_attempts = 0


#max_turns = 5
#st.session_state.turn = 0




########### Initializing Session State ##################

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "correction_attempts" not in st.session_state:
    st.session_state.correction_attempts = 0

if "turn" not in st.session_state:
    st.session_state.turn = 0

if "transcript" not in st.session_state:
    st.session_state.transcript = []

if "child_name" not in st.session_state:
    st.session_state.child_name = ""

if "session_ended" not in st.session_state:
    st.session_state.session_ended = False

if "feedback" not in st.session_state:
    st.session_state.feedback = {"rating": None, "comment": ""}

if "last_audio_hash" not in st.session_state:
    st.session_state.last_audio_hash = None



# Reset button AFTER init
if st.button("🔄 Reset Chat"):
    st.session_state.conversation_history = []
    st.session_state.correction_attempts = 0
    st.session_state.turn = 0
    st.session_state.transcript = []
    st.session_state.child_name = ""
    st.rerun()
######### Introduction message #####################

st.title("Kiku English Buddy 😊")
###############Session end code block ##############
if st.session_state.session_ended:
    st.title("Great job today! 🎉")

    # Session stats
    corrections = sum( 1 for e in st.session_state.transcript
     if e.get("mode") == "correction"
     )

    conversations =sum( 1 for e in st.session_state.transcript
      if e.get("mode") == 'conversation'
      )
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("💬 Turns", st.session_state.turn)
    with col2:
        st.metric("✅ Corrections", corrections)
    with col3:
        st.metric("😊 Conversations", conversations)
    st.markdown("---")
    st.markdown("### How was this session?")

    col1, col2 =st.columns(2)
    with col1:
        if st.button("👍 Helpful"):
            st.session_state.feedback["rating"]="helpful"
            st.toast("Thank you for the feedback! 🙏")
    with col2:
        if st.button("😕 Needs improvement"):
            st.session_state.feedback["rating"]="need improvement"
            st.toast("Thank you for the feedback! 🙏")
    comment= st.text_area(
        "Any comments? (optional)",placeholder="What worked well? What was confusing?"
    )
    if comment:
        st.session_state.feedback["comment"]=comment

    if st.button("🔄 Start New Session"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    st.stop()

########### Name Input Screen ##################
if not st.session_state.child_name:
    st.markdown("### Before we start...")
    name_input = st.text_input(
        "What is your name?",
        placeholder="Type your name here..."
    )
    if st.button("Start talking to Kiku! 🎉"):
        if name_input.strip():
            st.session_state.child_name = name_input.strip()
            st.rerun()
        else:
            st.warning("Please enter your name first!")
    st.stop()   # ← stops everything below from running until name is entered

####### Kiku Intro##################

kiku_intro = f"""Hi {st.session_state.child_name}! I'm Kiku 😊
I love talking and playing with words!
Press the mic button and say something — I'll talk with you!"""

with st.chat_message("assistant"):
    st.write(kiku_intro)



# Show previous messages
for msg in st.session_state.conversation_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


########### Bottom Controls: Mic + End Session ##################
st.markdown("---")
col1,col2 = st.columns([2,1])
with col1:
    st.markdown("**🎤 Speak to Kiku:**")
    audio = audiorecorder("⏺ Press & Speak", "⏹ Recording...",key="mic_input")
with col2:
    st.markdown("**Session**")
    end_clicked = st.button("⏹ End Session")
if end_clicked:
    st.session_state.session_ended=True
    st.rerun()

########### Text Input ##################

user_text= st.chat_input("Type your message...")
user_input = None



if len(audio) > 0:
    audio_bytes_io=io.BytesIO()
    audio.export(audio_bytes_io,format='wav')
    raw_bytes = audio_bytes_io.getvalue()
    # create unique fingerprint of current audio
    audio_hash = hashlib.md5(raw_bytes).hexdigest()
    #Only transcribe if this is a new Recording
    if audio_hash != st.session_state.last_audio_hash:
        st.session_state.last_audio_hash = audio_hash
        with st.spinner("Kiku is listening... 👂"):
            user_input=transcribe_audio(raw_bytes)
        if user_input:
            st.info(f"🎤 Kiku heard: *{user_input}*")
elif user_text:
    user_input=user_text




if user_input:
    # show user message
    with st.chat_message("user"):
        st.write(user_input)

    # store it
    st.session_state.conversation_history.append(
        {"role": "user", "content": user_input}
    )

    # temporary bot reply (dummy)
    sentence = user_input
    clean_sentence = re.sub(r'[^\w\s]', '', sentence.lower().strip())
    is_clarification = any(x in clean_sentence for x in clarification_inputs)

    if is_clarification:
      mode="clarification"
      result = None
    else:
      result = detect_grammar(sentence,st.session_state.conversation_history)
      if result.needs_correction and st.session_state.correction_attempts <3:
        mode="correction"
        st.session_state.correction_attempts +=1
      elif result.needs_correction and st.session_state.correction_attempts >=3:
        response="Let's move on for now and come back to it later. What else you want to talk about?"
        with st.chat_message("assistant"):
            st.write( response)
        with st.spinner("Kiku is speaking... 🔊"):
            audio_response = speak_response(response)
        st.audio(audio_response,format ="audio/mp3",autoplay = True)
        st.session_state.conversation_history.append({"role": "assistant", "content": response})
        st.session_state.transcript.append({"speaker":"child","text":user_input})
        st.session_state.transcript.append({"speaker":"Kiku","text":response,"mode":mode})

        st.session_state.correction_attempts = 0
        st.stop()
        
      else:
        mode="conversation"
        st.session_state.correction_attempts = 0
    response=responder_fun(sentence,result,mode,st.session_state.conversation_history)
    with st.chat_message("assistant"):
        st.write( response)
    with st.spinner("Kiku is speaking... 🔊"):
        audio_response=speak_response(response)
    st.audio(audio_response,format="audio/mp3",autoplay=True)

    st.session_state.conversation_history.append({"role": "assistant", "content": response})
    st.session_state.transcript.append({"speaker":"child","text":user_input})
    st.session_state.transcript.append({"speaker":"Kiku","text":response,"mode":mode})
    st.session_state.conversation_history = st.session_state.conversation_history[-40:]

if st.session_state.transcript:
    st.markdown("---")
    with st.expander("📝 View Full Transcript",expanded=False):
        for entry in st.session_state.transcript:
            if entry["speaker"]=="child":
                st.markdown(f"**🧒 Child:** {entry['text']}")
            else:
                mode_tag = entry.get("mode","")
                st.markdown(f"**🤖 Kiku** `[{mode_tag}]`: {entry['text']}")


