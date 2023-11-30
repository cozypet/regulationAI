# AI Regulation Q&A System
This repository contains a Python script for a Question & Answer system related to AI Regulation. The system utilizes OpenAI's language models, MongoDB for data storage, and various text processing libraries.

For more detailed information, you can check this article: https://artificialcorner.com/mongodb-and-langchain-magic-your-beginners-guide-to-setting-up-a-generative-ai-app-with-your-own-d1f90027d116

<img width="1201" alt="image" src="https://github.com/cozypet/regulationAI/assets/7107896/68b6c1bb-30c1-4f9e-8c04-d1c32c71fd91">

<img width="1210" alt="image" src="https://github.com/cozypet/regulationAI/assets/7107896/41ea036f-8f71-4771-a263-26e60d2e789e">

## Table of Contents
Installation

Usage

Dependencies

Configuration

Code Overview

Contributing

License

### Installation
Clone the repository to your local machine:
```
git clone https://github.com/yourusername/ai-regulation-QA.git
cd ai-regulation-QA
```

1. Ensure you have the necessary dependencies installed. Refer to the Dependencies section for details.

2. Set up your environment variables, especially the OpenAI API key and MongoDB connection details. Refer to the Configuration section for guidance.

3. Run the script:


```python main.py```


### Usage
The script has two main functions:

#### Data Preparation (data_prep):

Fetches data from a specified URL (in this case, a Wikipedia page).

Processes and cleans the text.

Splits the text into smaller chunks.

Calculates embeddings for the chunks.

Stores the data in a MongoDB database.

#### Question & Answer (qna):

Takes a user's question as input.

Embeds the question.

Searches for similar embeddings in the MongoDB database.

Generates a response based on context and the user's question.

### Dependencies

langchain.llms

langchain.PromptTemplate

bs4 (Beautiful Soup)

langchain.text_splitter.RecursiveCharacterTextSplitter

numpy

requests

openai

pymongo

pandas

### Configuration

OpenAI API Key:

Set your OpenAI API key as an environment variable. You can obtain an API key from the OpenAI platform.
MongoDB Connection:

Modify the connect_mongodb function to use your MongoDB URL, database name, and collection name.

### Code Overview

main.py: Contains the main script for data preparation and Q&A functionality.

langchain: A library containing language model-related functionalities.

text_splitter.py: A module for splitting text into smaller chunks.

config.py: Configuration file for setting up API keys and other environment variables.

### Contributing

If you'd like to contribute to this project, feel free to fork the repository and submit a pull request. Please follow the OpenAI code of conduct.

### License
This project is licensed under the MIT License - see the LICENSE file for details.
