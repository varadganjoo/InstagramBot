# Instagram News Reel Bot

This project is an Instagram bot that fetches the latest news, rewrites the title and content to be more engaging, generates a corresponding image with text overlay, and creates an Instagram reel. The bot then posts the reel to an Instagram account with an optimized caption to maximize reach.

## Features

- Fetches news from DuckDuckGo News.
- Rewrites news title and content using LlamaIndex and Groq API.
- Generates images from Pexels.
- Adds vignetting effect to images and adds the title of the news.
- Generates a text to speech narration of the .
- Posts the reel to Instagram with an optimized caption.

## Installation

1. Clone the repository:
    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Setup

1. Create a `config.py` file with your API keys and Instagram credentials:
    ```python
    # config.py
    GROQ_API_KEY = "your_groq_api_key"
    PEXELS_API_KEY = "your_pexels_api_key"
    INSTAGRAM_USERNAME = "your_instagram_username"
    INSTAGRAM_PASSWORD = "your_instagram_password"
    ```

2. Prepare a CSV file named `categories.csv` containing the topics for fetching news:
    ```csv
    Technology
    AI
    Science
    Physics
    Biology
    Chemistry
    Mathematics
    Business
    Finance
    Archaeology
    ```

## Usage

Run the main script to start the bot:
```sh
python main.py
```

## Implementation

You can see the implementation of this on the Instagram account: @aitechnews1

