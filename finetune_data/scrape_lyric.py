import requests
from bs4 import BeautifulSoup
import json
from tqdm import tqdm
page = requests.get('https://www.letras.com/mais-acessadas/')
soup = BeautifulSoup(page.text,'html.parser')
# Find the ol element with class 'top-list_art'
top_list_art = soup.find('ol', class_='top-list_art')

# Find all li elements within the ol
list_items = top_list_art.find_all('li') if top_list_art else []

# Extract the href from each a element within the li elements
urls = [li.find('a')['href'] for li in list_items if li.find('a')]
base_url = 'https://www.letras.com'
suffix = 'mais_acessadas.html'
full_urls = [base_url + url + suffix for url in urls]


# Print the URLs
lyric = []
# print(full_urls[:20])
for full_url in tqdm(full_urls[:100], desc="Processing URLs"):
    #artist
    page = requests.get(full_url)
    soup = BeautifulSoup(page.text,'html.parser')
    listOfSongs = []
    song_links = soup.find_all('a', class_='songList-table-songName')
    # Extract the href attribute from each element
    #songs
    hrefs = [link['href'] for link in song_links if 'href' in link.attrs]
    hrefs = [base_url + url for url in hrefs]
    # for i in soup.find_all('li'):
    #     listOfSongs.append(i)
    # print(hrefs[:20])
    
    # listOfSongs = [str(x) for x in listOfSongs]
    # newSongs = []
    # print(listOfSongs)
    # for link in listOfSongs:
    #     if 'https' in link:
    #         newSongs.append(link)
    #     urls = []
    #     for song in newSongs:
    #         try:
    #             urls.append(song.split('"')[13])
    #         except:
    #             continue
    #         # print(len(urls))
    #     print(urls)
    for url in hrefs:
        page = requests.get(url)
        soup = BeautifulSoup(page.text,'html.parser')

        lyrics_div = soup.find_all('div', class_='lyric-original')
        head_title = soup.find('h1', class_='head-title')

        # Extract the text from the h1 element
        title_text = head_title.get_text() if head_title else 'Title not found'

        # Print the title text
        # print(title_text)
        # Extract the text from each div
        lyrics = [div.get_text(separator="\n") for div in lyrics_div]
        lyrics = '\n'.join(lyrics)
        lyric.append({"Title":title_text,"Lyrics":lyrics})
        with open('scrape_lyric.txt', 'a') as f:
            f.write(title_text + '\n')
            f.write(lyrics + '\n\n')
        # # If you want to print the lyrics
        # for lyric in lyrics:
        #     print(lyric)
filename = 'scraped_lyrics.json'

# Write JSON data to a file
with open(filename, 'w') as file:
    json.dump(lyric, file, indent=4)