import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai
import pymongo
import os
import json



url = "https://en.wikipedia.org/wiki/Markets_in_Financial_Instruments_Directive_2014#Directive_2014/65/EU_/_Regulation_(EU)_No_600/2014"
response = requests.get(url)

soup = BeautifulSoup(response.content, 'html.parser')

# find all the text on the page
text = soup.get_text()

# find the content div
content_div = soup.find('div', {'class': 'mw-parser-output'})

# remove unwanted elements from div
unwanted_tags = ['sup', 'span', 'table', 'ul', 'ol']
for tag in unwanted_tags:
    for match in content_div.findAll(tag):
        match.extract()


print(content_div.get_text())
article_text = content_div.get_text()


text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 100,
    chunk_overlap  = 20,
    length_function = len,
)


texts = text_splitter.create_documents([article_text])
# Initialize an empty list to store the JSON objects
json_objects = []

# Loop through each text in the texts list
for idx, document in enumerate(texts):
    # Convert the Document object to a string
    text = str(document)
    
    # Create a JSON object with field "mifid2text" and value as the current text
    json_obj = {
        "mifid2text": text
    }
    # Append the JSON object to the list
    json_objects.append(json_obj)

# Define the filename for the JSON file
json_filename = "mifid2_texts.json"

# Write the list of JSON objects to a JSON file
with open(json_filename, "w") as json_file:
    json.dump(json_objects, json_file, indent=4)

print(f"JSON objects stored in {json_filename}")


# MongoDB Atlas configuration
mongo_url = "mongodb+srv://yourusername:psw@yourcluster/test?retryWrites=true&w=majority"
client = pymongo.MongoClient(mongo_url)
db = client["AIRegulation"]
collection = db["mifid2"]

# Load the JSON data from the file
with open(json_filename, "r") as json_file:
    data = json.load(json_file)

# Insert the data into the collection
#collection.insert_many(data)

#print("Data inserted into MongoDB Atlas")

#with open('mifid2withoutembeddings.json', 'w') as json_file:

#    json.dump(texts, json_file, indent=4)
#print(texts[0])
#print(texts[1])

# Generate embeddings for only text[0]
url = "https://api.openai.com/v1/embeddings"
openai_key = "YOUR OPENAI KEY"  # Replace with your OpenAI key.


# for text in texts:
#     response = requests.post(
#         url,
#         json={"input": text.page_content, "model": "text-embedding-ada-002"},
#         headers={
#             "Authorization": f"Bearer {openai_key}",
#             "Content-Type": "application/json",
#         },
#     )
#     embedding=response.json()["data"][0]["embedding"]

#     print(embedding)
