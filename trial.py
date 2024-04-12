import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # load all the environment variables
import os
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import BartForConditionalGeneration, BartTokenizer

# Load Google Gemini Pro API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Set page configuration
st.set_page_config(page_title="YouTube Summarizer & Q&A", page_icon="🤖")

# Set background color and padding for the main content
st.markdown(
    """
    <style>
    .main {
        background-color: #1;
        padding: 2rem;
    }
    .top-left {
        position: absolute;
        top: 10px;
        left: 10px;
        padding: 0rem;
    }
    .top-right {
        position: absolute;
        top: 10px;
        right: 10px;
    }
    .header {
        background-color: #1A1A1A;
        color:#FFFFFF; /* Changed text color to white */
        padding: 3rem;
        text-align: left; /* Align text to the left */
        position: fixed;
        width: 100%;
        top: 0;
        left: 0;
        z-index: 1000;
        font-size: 34px;
        display: flex;
        align-items: center;
        justify-content: space-between; /* Space between header content */
        height: 150px; /* Height of the header */
        line-height: 100px;
    }
    .footer {
        background-color: #AAAAAA;
        color: white; /* Changed text color to white */
        padding: 0.1rem 0;
        text-align: center;
        position: fixed;
        width: 100%;
        bottom: 0;
        left: 0;
        font-size:15px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Set the title and description
st.title("Summarize & Ask: YouTube Edition")
st.markdown(
    """
    
    """
)

# Function to extract transcript details from YouTube video
def extract_transcript_details(youtube_video_url):
    try:
        video_id = youtube_video_url.split("=")[1]
        transcript_text = YouTubeTranscriptApi.get_transcript(video_id)

        transcript = ""
        for i in transcript_text:
            transcript += " " + i["text"]

        return transcript

    except Exception as e:
        st.error("Error occurred while extracting transcript details.")
        st.error(e)

# Function to generate summary using Google Gemini Pro model
def generate_summary(transcript_text, prompt):
    try:
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt + transcript_text)
        summary = response.text
        return summary

    except Exception as e:
        st.error("Error occurred while generating summary.")
        st.error(e)

# Function to answer questions using BART model
def answer_question(question, transcript_text):
    try:
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
        inputs = tokenizer(question, transcript_text, add_special_tokens=True, return_tensors="pt", max_length=1024, truncation=True)
        answer_ids = model.generate(inputs['input_ids'], max_length=250, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
        answer = tokenizer.decode(answer_ids[0], skip_special_tokens=True)
        return answer

    except Exception as e:
        st.error("Error occurred while answering question.")
        st.error(e)

# User input for YouTube video link
youtube_link = st.text_input("Enter YouTube Video Link:")

# If link is provided
if youtube_link:
    # Display YouTube video thumbnail
    video_id = youtube_link.split("=")[1]
    st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True)

    # Button to trigger summarization
    if st.button("Get Summary"):
        if youtube_link:
            # Extract transcript text
            transcript_text = extract_transcript_details(youtube_link)
            if transcript_text:
                # Generate summary
                prompt = "You are YouTube video summarizer. You will be taking the transcript text " \
                         "and summarizing the entire video and providing the important summary in points " \
                         "within 250 words. Please provide the summary of the text given here:"
                summary = generate_summary(transcript_text, prompt)
                # Display the summary
                st.markdown("## Summary:")
                st.write(summary)

    # User input for question
    question = st.text_input("Ask a question about the video:")

    # Button to trigger question-answering
    if st.button("Get Answer"):
        if youtube_link:
            # Extract transcript text
            transcript_text = extract_transcript_details(youtube_link)
            if transcript_text:
                # Answer the question
                answer = answer_question(question, transcript_text)
                # Display the answer
                st.markdown("## Answer:")
                st.write(answer)

# Header with icons
st.markdown("""
<div class='header'>
    <div style='float: left;'>YouTube Video Summarizer cum Q&A</div>
    <div style='float: right;'>
        <a href="#" style="text-decoration: none;"><span style="font-size: 16px;">Login/Sigup</span></a>
      <a href="https://www.youtube.com/" style="text-decoration: none;"><span style='font-size: 30px;'>🎥</span></a>
    </div>
</div>
""", unsafe_allow_html=True)
# Footer
st.markdown("<div class='footer'>Made with ❤️ by Sanjana Gavada <br> <a href='https://www.linkedin.com/in/YourLinkedInProfile' target='_blank'><img src='https://content.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Logo.svg.original.svg' width='70' height='70' style='margin-right: 10px;'/></a></div>", unsafe_allow_html=True)
