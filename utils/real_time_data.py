import requests
from config import SERPER_API_KEY



def fetch_real_time_data(query: str):
    '''
    Fetch general real-time data from google
    '''
    # Replace with your actual Serper API call
    serper_api_url = f"https://google.serper.dev/search?q={query}&api_key={SERPER_API_KEY}"
    
    # Make the API request to Serper for real-time data
    response = requests.get(serper_api_url)
    if response.status_code == 200:
        data = response.json()
        print(data)
        # Extract the most relevant data from the response
        # This may vary based on the Serper API response format
        return data.get("results", [{}])[0].get("snippet", "No data found")
    else:
        return "Error fetching data from Serper API"
    
