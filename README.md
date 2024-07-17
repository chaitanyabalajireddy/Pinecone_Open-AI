# Pinecone_Open-AI


## Project Title: Question Answering Bot

This project implements a Context retrieval system using OpenAI, Pinecone, and Sentence Transformers. It is designed to answer questions by retrieving relevant contexts from a dataset and generating responses based on those contexts.

### Features

- **Document Indexing**: Utilizes Pinecone to index document embeddings for efficient retrieval.
- **Context Extraction**: Extracts contexts from the SQuAD v2 dataset to create a knowledge base.
- **Embedding Generation**: Leverages Sentence Transformers to generate embeddings for document contexts.
- **Querying**: Allows users to input questions and retrieves relevant documents to provide answers.

### Technologies Used

- **OpenAI**: For generating responses to user queries.
- **Pinecone**: For indexing and retrieving document embeddings.
- **Sentence Transformers**: For generating embeddings of text data.
- **Datasets**: For loading and processing the SQuAD v2 dataset.

### Installation

To set up the project, ensure you have the required libraries installed:

```bash
pip install openai pinecone-client transformers datasets sentence-transformers
```

### Getting Started

1. **Initialize Pinecone**: Create a Pinecone index to store document embeddings.
2. **Load the Dataset**: Use the SQuAD v2 dataset for context extraction.
3. **Generate Embeddings**: Encode document contexts to create vector representations.
4. **Query the System**: Input a question to retrieve relevant documents and generate answers.

### Usage

Here’s a detailed explanation of your code step by step:

### 1. **Library Imports**

```python
import openai
import pinecone
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import numpy as np
```

- **`openai`**: For interacting with OpenAI's API to generate text-based answers.
- **`pinecone`**: For creating and managing a vector database for document indexing and retrieval.
- **`SentenceTransformer`**: To generate embeddings for the documents using a pre-trained model.
- **`load_dataset`**: To load datasets (like SQuAD) for training or testing the model.
- **`numpy`**: For numerical operations, especially when handling embeddings.

### 2. **Pinecone Setup**

```python
from pinecone import Pinecone, ServerlessSpec
index_name = "bot"
pc = Pinecone(api_key="your_pinecone_api_key")
pc.create_index(
    name=index_name,
    dimension=384,  # Replace with your model dimensions
    metric="cosine",  # Replace with your model metric
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)
```

- **Creating a Pinecone Instance**: Initializes the Pinecone client with your API key.
- **Creating an Index**: Creates a vector index in Pinecone for storing document embeddings. The `dimension` should match the output size of the embedding model, and `metric` specifies how to measure similarity (cosine similarity in this case).

### 3. **OpenAI API Key Setup**

```python
openai.api_key = "your_openai_api_key"
```

- Sets up your OpenAI API key for making requests to the OpenAI models.

### 4. **Loading the Dataset**

```python
dataset = load_dataset("squad_v2")
```

- Loads the SQuAD v2 dataset, which contains questions and contexts for training or evaluation purposes.

### 5. **Extracting and Processing Contexts**

```python
contexts = []
for item in dataset['train']:
    contexts.append((item['context'], item['id']))  # Keep track of document IDs

# Remove duplicate contexts
contexts = list(set(contexts))
```

- **Extracting Contexts**: Iterates through the dataset to collect the context and its corresponding ID.
- **Removing Duplicates**: Ensures that only unique contexts are retained for indexing.

### 6. **Generating Embeddings**

```python
embeddings = model.encode([context for context, _ in contexts])
```

- Uses the `SentenceTransformer` model to generate embeddings for each context in the dataset. This converts the text into numerical vectors for indexing.

### 7. **Upserting Vectors into Pinecone**

```python
batch_size = 1000
for i in range(0, len(embeddings), batch_size):
    batch_vectors = [
        (f'doc-{i+idx}', embedding.tolist(), {'text': context, 'id': doc_id})
        for idx, (context, doc_id) in enumerate(contexts[i:i+batch_size])
        for i, embedding in enumerate(embeddings[i:i+batch_size])
    ]
    index.upsert(batch_vectors)
```

- **Batching**: Divides the embeddings into smaller batches (of size 1000) to efficiently upsert into Pinecone.
- **Upsert Operation**: Adds the embeddings into the Pinecone index along with their metadata (original text and document ID).

### 8. **Retrieving Documents**

```python
def retrieve_documents(query, top_k=5):
    query_embedding = model.encode([query])
    query_embedding = query_embedding[0].tolist()  # Convert ndarray to list
    result = index.query(vector=query_embedding, top_k=top_k)
    return [match['id'] for match in result['matches']]
```

- **Query Embedding**: Converts the input query into an embedding using the same model.
- **Querying Pinecone**: Searches for the top `k` most similar document embeddings in the index.
- **Return Document IDs**: Extracts and returns the IDs of the matched documents.

### 9. **Constructing a Prompt for OpenAI**

```python
def construct_prompt(query, documents):
    context = "\n\n".join(documents)
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    return prompt
```

- **Building a Prompt**: Creates a prompt string by combining the context from retrieved documents with the original question, formatted for input to OpenAI's model.

### 10. **Generating an Answer**

```python
def generate_answer(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",  # or any preferred model
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()
```

- **Generating Response**: Sends the constructed prompt to OpenAI's model and retrieves the generated answer.

### 11. **Answering Questions**

```python
def answer_question(query):
    document_ids = retrieve_documents(query)
    answers = []
    for doc_id in document_ids:
        fetched_data = index.fetch([doc_id])
        if doc_id in fetched_data['vectors']:
            metadata = fetched_data['vectors'][doc_id].get('metadata', {})
            text = metadata.get('text', 'No text available')
            answers.append(text)
        else:
            answers.append('No text available for this document')
    return answers
```

- **Retrieve Document IDs**: Uses the `retrieve_documents` function to get relevant document IDs.
- **Fetch Document Metadata**: Retrieves the full document text using the fetched IDs and constructs a list of answers.

### 12. **Putting It All Together**

```python
query = "When did Beyonce start becoming popular?"
answer = answer_question(query)
print(answer)
```

- **Query Execution**: Executes a sample query and prints out the retrieved answers. In this case, it outputs an answer related to Beyonce's popularity.

### Summary of Output

The expected output from the example query is **"in the late 1990s"**, indicating that the system has successfully retrieved relevant documents and generated an appropriate answer using the OpenAI model.

