import requests
import csv
import time
from duckduckgo_search import DDGS
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import textwrap
import io
import os
import random
from llama_index.llms.groq import Groq
from llama_index.core import Settings
from llama_index.core.llms import ChatMessage
import streamlit as st
import nest_asyncio
import httpx
from moviepy.editor import ImageClip, AudioFileClip, CompositeVideoClip, VideoFileClip
from moviepy.video.fx.speedx import speedx
from gtts import gTTS
from instagrapi import Client
import logging

# Global variables for API keys
GROQ_API_KEY = "YOUR_GROQ_API_KEY"
PEXELS_API_KEY = "YOUR_PEXELS_API_KEY"
INSTAGRAM_USERNAME = "YOUR_INSTAGRAM_USERNAME"
INSTAGRAM_PASSWORD = "YOUR_INSTAGRAM_PASSWORD"

# Setup for using Groq with LlamaIndex
nest_asyncio.apply()

# Define the Groq LLM with your model and API key
llm = Groq(model="llama3-70b-8192", api_key=GROQ_API_KEY, client=httpx.Client(timeout=httpx.Timeout(60.0)))
Settings.llm = llm

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to read categories from CSV file
def read_categories_from_csv(csv_file):
    categories = []
    with open(csv_file, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            categories.extend(row)
    return categories

# Retry decorator for handling requests
def retry_on_failure(func):
    def wrapper(*args, **kwargs):
        retries = 3
        for attempt in range(retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                time.sleep(2 ** attempt)
        raise Exception(f"Failed after {retries} attempts")
    return wrapper

# Fetch multiple images from Pexels
@retry_on_failure
def fetch_images(query, api_key, num_images=10, size='large'):
    headers = {
        "Authorization": api_key
    }
    params = {
        "query": query,
        "per_page": num_images,
        "orientation": "portrait",
        "size": size
    }
    response = requests.get("https://api.pexels.com/v1/search", headers=headers, params=params)
    response.raise_for_status()
    data = response.json()
    images = []
    for photo in data['photos']:
        if (photo['width'] == 1080 and photo['height'] == 1920) or size != 'large':
            image_url = photo['src']['large']
            image_response = requests.get(image_url)
            image_response.raise_for_status()
            images.append(Image.open(io.BytesIO(image_response.content)))
    return images

# Add black vignetting effect
def add_vignette(image):
    width, height = image.size
    vignette = Image.new('L', (width, height))
    gradient = ImageDraw.Draw(vignette)
    
    # Adjust the vignette effect by changing the gradient calculation
    for y in range(height):
        for x in range(width):
            distance_to_center = ((x - width / 2) ** 2 + (y - height / 2) ** 2) ** 0.5
            distance_to_center = distance_to_center / (width / 1.5)  # Increase the denominator for a wider vignette
            gradient_value = int(255 * (distance_to_center ** 3))  # Increase the exponent for a more gradual transition
            vignette.putpixel((x, y), gradient_value)

    vignette = vignette.filter(ImageFilter.GaussianBlur(50))  # Increase blur for a softer transition
    vignette = vignette.resize(image.size)

    # Apply the vignette to the image
    black_vignette = Image.new('RGB', image.size, color=0)
    image = Image.composite(black_vignette, image, vignette)

    return image

# Generate Instagram Post Image
def create_instagram_post(news_title, image):
    # Make the image 9:16 aspect ratio (1080x1920)
    image = image.resize((1080, 1920), Image.Resampling.LANCZOS)
    news_title = news_title.replace("\"", '').replace("*", '').strip()
    
    # Add vignetting effect
    image = add_vignette(image)
    
    draw = ImageDraw.Draw(image)
    # Calculate proportional font size
    font_size = int(image.width * 0.06)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    # Calculate text size and wrap text
    max_width = image.width - 40  # Margins on both sides
    wrapped_title = textwrap.fill(news_title, width=40)
    lines = wrapped_title.split('\n')
    wrapped_title = ""
    for line in lines:
        if draw.textlength(line, font=font) > max_width:
            wrapped_title += textwrap.fill(line, width=30) + "\n"
        else:
            wrapped_title += line + "\n"

    wrapped_title = wrapped_title.strip()

    # Calculate text size and position
    text_size = draw.textbbox((0, 0), wrapped_title, font=font)
    text_width = text_size[2] - text_size[0]
    text_height = text_size[3] - text_size[1]
    position = ((image.width - text_width) / 2, image.height - text_height - 20)  # Center align text at the bottom
    
    # Create a black rectangle with decreased opacity behind the text
    rect_height = text_height + 20
    rect_position = (0, image.height - rect_height, image.width, image.height)
    rect_image = Image.new('RGBA', (rect_position[2] - rect_position[0], rect_position[3] - rect_position[1]), (0, 0, 0, 128))
    image.paste(rect_image, (rect_position[0], rect_position[1]), rect_image)
    
    draw.text(position, wrapped_title, (255, 255, 255), font=font, align="center")
    return image

# Save the edited image
def save_image(image, filename):
    if not os.path.exists('images'):
        os.makedirs('images')
    image_path = os.path.join('images', filename)
    image.save(image_path)
    print(f"Image saved to {image_path}")
    return image_path

# Check if news is an ad or mentions discounts
def is_advertisement(news_content):
    ad_keywords = ["discount", "sale", "offer", "promotion", "buy now", "limited time", "ad", "sponsored"]
    return any(keyword.lower() in news_content.lower() for keyword in ad_keywords)

# Check if news is interesting
def is_interesting(news_title, news_content):
    prompt = ChatMessage(role="user", content=f"Is this news interesting? Please respond with 'yes' or 'no':\n\nTitle: {news_title}\nContent: {news_content}")
    try:
        response = llm.chat([prompt])
        interesting = str(response).replace('assistant:', '').strip().lower()
        return 'yes' if 'yes' in interesting else None
    except Exception as e:
        logger.error(f"Error determining if news is interesting: {e}")
        return False

# Check if the news has already been posted
def is_duplicate(news_content, ledger_file='post_ledger.csv'):
    if not os.path.exists(ledger_file):
        return False
    with open(ledger_file, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            if news_content in row:
                return True
    return False

# Append post details to the ledger CSV file
def append_to_ledger(news_title, news_content, image_path, video_path, ledger_file='post_ledger.csv'):
    with open(ledger_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([news_title, news_content, image_path, video_path])

# Fetch and rewrite news
@retry_on_failure
def fetch_and_rewrite_news(topic):
    ddgs = DDGS()
    news_results = ddgs.news(topic, safesearch='Off', max_results=1)
    if news_results:
        news = news_results[0]
        news_title = news['title']
        news_content = news['body']
        
        # Logging fetched news
        print(f"Fetched news title: {news_title}")
        print(f"Fetched news content: {news_content[:200]}...")
        
        # Check if the news is an advertisement
        if is_advertisement(news_content):
            print("The fetched news is an advertisement or mentions discounts. Skipping...")
            return None, None

        # Rewriting the title and content using Groq
        title_prompt = ChatMessage(role="user", content=f"Rewrite this news title in the most truthful way possible while keeping it interesting. Make sure to keep it short and sweet: {news_title} \n Do not give any explanation and do not say anything that has nothing to do with the content itself such as \"discount\", \"sale\", \"offer\", \"promotion\", \"buy now\", \"limited time\", \"ad\", or \"sponsored\".")
        content_prompt = ChatMessage(role="user", content=f"Rewrite this news content in the most truthful way possible while keeping it interesting: {news_content}\n Do not give any explanation and do not say anything that has nothing to do with the content itself such as \"discount\", \"sale\", \"offer\", \"promotion\", \"buy now\", \"limited time\", \"ad\", or \"sponsored\". Do not say anything that is not related to the news itself. Your response should in no way show that it was written or rewritten by you")
        
        try:
            rewritten_title = str(llm.chat([title_prompt])).replace('assistant:', '').replace('*', '').strip()
            rewritten_content = str(llm.chat([content_prompt])).replace('assistant:', '').replace('*', '').strip()
            print(f"Rewritten title: {rewritten_title}")
            print(f"Rewritten content: {rewritten_content}")
            return rewritten_title, rewritten_content
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return None, None
    else:
        print("No news results found.")
    return None, None

# Generate image query based on news content
def generate_image_query(news_content):
    prompt = ChatMessage(role="user", content=f"You are searching for an image that best matches the content. What would be the most relevant words in this: {news_content} \n\n Just answer in 1 word. Do not search for images of people or specific names. make sure the query is general.")
    try:
        query_response = llm.chat([prompt])
        image_query = str(query_response).replace('assistant:', '').replace("*",'').strip()
        print(f"Generated image query: {image_query}")
        return image_query.replace(' ', '+')
    except Exception as e:
        logger.error(f"Error generating image query: {e}")
        return None

# Determine the mood of the news
def determine_mood(news_title):
    prompt = ChatMessage(role="user", content=f"Describe the mood of this news in one word: {news_title}")
    try:
        mood_response = llm.chat([prompt])
        mood = str(mood_response).replace('assistant:', '').replace("*", "").strip()
        print(f"Determined mood: {mood}")
        return mood
    except Exception as e:
        logger.error(f"Error determining mood: {e}")
        return "neutral"

# Generate hashtags based on news content
def generate_hashtags(news_content):
    prompt = ChatMessage(role="user", content=f"Generate relevant hashtags for this content. Provide up to 10 hashtags:\n\n{news_content}")
    try:
        response = llm.chat([prompt])
        hashtags = str(response).replace('assistant:', '').replace('*', '').strip()
        hashtags = [tag.strip() for tag in hashtags.split() if tag.startswith('#')]
        return ' '.join(hashtags)
    except Exception as e:
        logger.error(f"Error generating hashtags: {e}")
        return ""

# Generate TTS audio
def generate_tts_audio(text, output_path):
    tts = gTTS(text)
    tts.save(output_path)
    return output_path

# Create video with TTS audio
def create_video(image_path, audio_path, output_path):
    image_clip = ImageClip(image_path)
    audio_clip = AudioFileClip(audio_path)
    video = CompositeVideoClip([image_clip.set_audio(audio_clip)])
    video = video.set_duration(audio_clip.duration)  # Set video duration to audio duration
    video.fps = 24  # Set FPS for the video clip
    video.write_videofile(output_path, codec='libx264', audio_codec='aac', fps=24)
    return output_path

# Speed up the video
def speed_up_video(video_path, output_path, factor=1.2):
    video_clip = VideoFileClip(video_path)
    sped_up_clip = speedx(video_clip, factor)
    sped_up_clip.write_videofile(output_path, codec='libx264', audio_codec='aac', fps=24)

# Post video to Instagram using instagrapi
def post_on_instagram(video_path, username, password, title, content, hashtags):
    cl = Client()
    cl.login(username, password)
    caption = f"{title}\n\n{content}\n\n{hashtags}"
    cl.clip_upload(video_path, caption)

# Main function
def main():
    categories_csv = 'categories.csv'
    categories = read_categories_from_csv(categories_csv)
    topic_attempts = 0
    delay = 2  # Initial delay time in seconds

    while topic_attempts < len(categories):
        topic = random.choice(categories)  # Randomly select a topic from the categories
        print(f"Selected topic: {topic}")
        
        news_title, news_content = fetch_and_rewrite_news(topic)
        if news_title and news_content:
            if not is_duplicate(news_content) and is_interesting(news_title, news_content):
                image_query = generate_image_query(news_content)
                if image_query:
                    images = fetch_images(image_query, PEXELS_API_KEY)
                    if not images:  # Try fetching with relaxed size criteria
                        images = fetch_images(image_query, PEXELS_API_KEY, size='medium')
                    if images:
                        image = random.choice(images)  # Select a random image from the fetched images
                        edited_image = create_instagram_post(news_title, image)
                        image_path = save_image(edited_image, "edited_image.png")
                        
                        tts_audio_path = generate_tts_audio(news_content, "news_narration.mp3")
                        
                        if tts_audio_path:
                            video_path = create_video(image_path, tts_audio_path, "instagram_reel.mp4")
                            speed_up_video(video_path, "instagram_reel_sped_up.mp4", factor=1.2)
                            print("Instagram reel created and sped up successfully!")
                            
                            # Append post details to the ledger
                            append_to_ledger(news_title, news_content, image_path, "instagram_reel_sped_up.mp4")

                            # Generate hashtags
                            hashtags = generate_hashtags(news_content)

                            # Post on Instagram
                            post_on_instagram("instagram_reel_sped_up.mp4", INSTAGRAM_USERNAME, INSTAGRAM_PASSWORD, news_title, news_content, hashtags)
                            return
                        else:
                            print("Failed to generate TTS audio.")
                    else:
                        print("Failed to fetch images. Trying a different topic...")
                else:
                    print("Failed to generate image query.")
            else:
                print("News is either duplicate or not interesting.")
        else:
            print("Failed to fetch or rewrite news.")
        
        topic_attempts += 1
        categories.remove(topic)  # Remove the current topic and try another
        time.sleep(delay)  # Wait before retrying
        delay *= 2  # Exponential backoff

    print("Failed to create an Instagram reel after trying all topics.")

if __name__ == "__main__":
    main()
