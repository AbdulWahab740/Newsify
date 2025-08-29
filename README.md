# **Newsify**
Newsify is a cutting-edge news aggregation platform that leverages advanced AI to provide users with a personalized and efficient news consumption experience.  
It's designed to help you stay informed without being overwhelmed by the sheer volume of daily news.

--- 

## 🌟 Features
- **Personalized News Feed:** Tailored to your interests.

- **Efficient Search:** Find exactly what you're looking for with a powerful, fast, and accurate search.

- **Summarized Articles:** Get the key takeaways of a story at a glance.

- **Ad-Free Experience:** Focus on the content, not the clutter.

## 💻 How It Works:

Newsify operates through a series of intelligent steps to deliver a superior news experience.

- **Extraction:** The system continuously extracts the latest news from various sources using an external API.

- **Vectorization & Storage:** Each news article is processed and converted into a numerical representation (vector) and stored in a specialized vector database. This approach allows for rapid and semantic similarity-based searching.

- **Advanced Retrieval:** To ensure you get the most relevant and unique results, Newsify employs a sophisticated retrieval pipeline:

- **Hybrid Search:** This method combines the strengths of both keyword and semantic search. It's like having the best of both worlds: you can find articles based on specific terms (like "AI" or "Tesla") while also discovering articles that are conceptually related, even if they don't use the exact keywords.

- **Re-ranking:** After the initial search, a re-ranking model evaluates the retrieved articles to prioritize those that are most relevant to your query. This step ensures that the best results rise to the top.

- **Deduplication:** The system identifies and removes duplicate or near-duplicate articles from the search results, ensuring you don't see the same story multiple times from different sources.

## 🚀 Getting Started
Follow these steps to get a local copy of Newsify up and running.

Prerequisites
You'll need the following installed on your machine:

Python 3.10+

Installation

Clone the repository:

```
git clone https://github.com/AbdulWahab740/Newsify.git
cd newsify
```

Install dependencies:

```
pip install -r requirements.txt
```

Configure API keys:
Create a .env file in the root directory and add your API keys:

```
API_KEY = "your_gnews_api_key"
GROK_API_KEY = "your_grok_api_key
```
Run the application:

```
streamlit run app.py
```

🤝 Contributing
I welcome contributions! Please feel free to submit a pull request or open an issue.

Created by:

`AbdulWahab` 
Ai/ML Engineer ([Linkedin](www.linkedin.com/in/abwahab07))
