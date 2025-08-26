import requests

def get_news(query):
    if (len(query.split(" "))> 1):
        query = "+".join(query.split(" "))
    print(query)
    response = requests.get(f"https://gnews.io/api/v4/search?q={query}&lang=en&max=5&apikey=56df04756b170b5d44e74c4ed9eb8159")
    response.raise_for_status()
    data = response.json()
    return data
