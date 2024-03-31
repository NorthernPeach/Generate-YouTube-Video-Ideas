import streamlit as st
from utils.video_ideas import build_llm
from utils.extract_post_content import extract_page_content
from utils.generate_youtube_content_ideas import generate_youtube_ideas_from_content

def video(url_high_eng, url_low_eng):
    if url_high_eng and url_low_eng:
        with st.spinner("🔮 Doing  some magic..."):
            url_high_eng = url_high_eng.split(",")
            url_low_eng  = url_low_eng.split(",")
            
            response = build_llm(url_high_eng, url_low_eng)
            st.write("🔎 Here is what we got from your input!")
            st.dataframe(response)
    else:
        st.warning("⚠️Please enter video URL(s) first.")
        
def blog(url):
    if url:
        with st.spinner("🔮 Doing  some magic..."):
            page_content = extract_page_content(url)
            response     = generate_youtube_ideas_from_content(page_content)
            st.write("🔎 Here is what we found from your blog!")
            st.markdown(response)
    else:
        st.warning("⚠️Please enter a Blog URL first.")

# Sidebar Elements ----------------------------------------------
st.title("🎉Youtube Viral Video Ideas Generator")
st.write("❓What do you want to use as the source for Ideas❓")

with st.form("user_inputs"):
    video_button = st.form_submit_button("📹Videos")
    if video_button:
        url_high_eng = st.text_input("🔗Enter High Engagement Video URL(s) separated by comma:")
        url_low_eng  = st.text_input("🔗 Enter Low Engagement Video URL(s) separated by comma:")
        video(url_high_eng, url_low_eng)

with st.form("user_inputs2"):
    post_button = st.form_submit_button("📝Blog")
    if post_button:
        url = st.text_input("🔗 Enter Blog URL:")
        blog(url)


            