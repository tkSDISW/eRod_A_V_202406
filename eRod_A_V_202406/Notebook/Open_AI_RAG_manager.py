import sys
from pathlib import Path
from openai import OpenAI
from IPython.display import display, Markdown
from IPython.core.display import HTML
from ipywidgets import widgets
import numpy as np
import json
import os
from datetime import datetime

import asyncio
from ipywidgets import widgets
from IPython.display import display

def get_api_key():
    """Retrieve the OpenAI API key from a hidden file."""
    home_dir = Path.home()
    key_file = home_dir / ".secrets" / "openai_api_key.txt"

    if not key_file.exists():
        raise FileNotFoundError(
            f"API Key file not found at {key_file}. "
            "Please ensure the key is saved correctly."
        )

    with key_file.open("r") as f:
        api_key = f.read().strip()

    if not api_key:
        raise ValueError("API Key file is empty. Provide a valid API key.")

    return api_key


class EmbeddingManager :
    def __init__(self):
        """Initialize the analyzer with YAML content."""
        try:
            self.api_key = get_api_key()
            print("OpenAI API Key retrieved successfully.")
        except (FileNotFoundError, ValueError) as e:
            print(e)
            sys.exit(1)  # Exit with an error code

       
        self.client = OpenAI(
            api_key=self.api_key,  # This is the default and can be omitted
            )
        self.model = "text-embedding-3-small"
        self.model_file = ''
        self.embedding_file =''
    
        
    def save_embeddings(self ):
        """Save embeddings to a file."""
        with open(self.embedding_file, "w") as f:
            json.dump(self.embeddings, f)

    def set_files(self, model_file, embedding_file) :
        """Set model file and ebedding fie."""
        self.model_file = model_file
        self.embedding_file = embedding_file


    def is_embedding_up_to_date(self):
        """Check if the embedding file is up-to-date with the model file."""
        if not os.path.exists(self.embedding_file):
            # Return False if the embedding file is not present
            print("Embedding file not found.")
            return False
        model_time = os.path.getmtime(self.model_file)
        embedding_time = os.path.getmtime(self.embedding_file)
        #print("embedding time:",embedding_time, "model time:", model_time) 
        return embedding_time > model_time
    
    def generate_object_embeddings(self, objects):
        for obj in objects:
            # Combine metadata into a single text string
            metadata_text = f"{obj['name']} {obj['description']} {obj['type']}"
            metadata_text =  metadata_text.replace("\n", " ")
            obj["embedding"] = self.client.embeddings.create(input = [metadata_text], model=self.model).data[0].embedding
        self.embeddings = objects
        return objects


    

    
    def create_model_embeddings(self, model) :
        def get_object_info(object) :
            object_info = {
                    "uuid": object.uuid,
                    "name": object.name,
                    "description": object.description,
                    "type": type(object).__name__
                }
            return object_info
        
        if self.is_embedding_up_to_date() :
            print("Loading embeddings")
            self.load_embeddings()
        else : 
            print("Creating Embeddings")
            object_data = []
            #OA
            for component in model.oa.all_entities:  # Assuming Logical Architecture
                object_info = get_object_info(component)
                object_data.append(object_info)
            for obj in model.oa.all_activities:  # Assuming Logical Architecture
                object_info = get_object_info(obj)
                object_data.append(object_info)
            for obj in model.oa.all_capabilities:  # Assuming Logical Architecture
                object_info = get_object_info(obj)
                object_data.append(object_info)
            for obj in model.oa.all_entity_exchanges:  # Assuming Logical Architecture
                object_info = get_object_info(obj)
                object_data.append(object_info)
            for obj in model.oa.all_processes:  # Assuming Logical Architecture
                object_info = get_object_info(obj)
                object_data.append(object_info)
            #SA
            for component in model.sa.all_components:  # Assuming Logical Architecture
                object_info = get_object_info(component)
                object_data.append(object_info)
            for obj in model.sa.all_capabilities:  # Assuming Logical Architecture
                object_info = get_object_info(obj)
                object_data.append(object_info)
            for obj in model.sa.all_function_exchanges:  # Assuming Logical Architecture
                object_info = get_object_info(obj)
                object_data.append(object_info)
            for obj in model.sa.all_functions:  # Assuming Logical Architecture
                object_info = get_object_info(obj)
                object_data.append(object_info)
            for obj in model.sa.all_missions:  # Assuming Logical Architecture
                object_info = get_object_info(obj)
                object_data.append(object_info)
            for obj in model.sa.all_functional_chains:  # Assuming Logical Architecture
                object_info = get_object_info(obj)
                object_data.append(object_info)            #LA
            for obj in model.la.all_capabilities:  # Assuming Logical Architecture
                object_info = get_object_info(obj)
                object_data.append(object_info)
            for component in model.la.all_components:  # Assuming Logical Architecture
                object_info = get_object_info(component)
                object_data.append(object_info)
            for obj in model.la.all_functions:  # Assuming Logical Architecture
                object_info = get_object_info(obj)
                object_data.append(object_info)
            for obj in model.la.all_functional_chains:  # Assuming Logical Architecture
                object_info = get_object_info(obj)
            for obj in model.la.all_component_exchanges:  # Assuming Logical Architecture
                object_info = get_object_info(obj)
            #PA
            for component in model.pa.all_components:  # Assuming Logical Architecture
                object_info = get_object_info(component)
                object_data.append(object_info)
            for obj in model.pa.all_functions:  # Assuming Logical Architecture
                object_info = get_object_info(obj)
                object_data.append(object_info)
            for obj in model.la.all_functional_chains:  # Assuming Logical Architecture
                object_info = get_object_info(obj)
            for obj in model.pa.all_capabilities:  # Assuming Logical Architecture
                object_info = get_object_info(obj)
            for obj in model.pa.all_component_exchanges:  # Assuming Logical Architecture
                object_info = get_object_info(obj)
            for obj in model.pa.all_physical_exchanges:  # Assuming Logical Architecture
                object_info = get_object_info(obj)
            model_embeddings = self.generate_object_embeddings(object_data)
            print("Saving embeddings")
            self.save_embeddings()

    
    
    def load_embeddings(self):
        """Load embeddings from a file."""
        with open(self.embedding_file, "r") as f:
            self.embeddings = json.load(f)

    # Define a cosine similarity function
    def cosine_similarity(self, vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    # Find similar objects based on the query
    def find_similar_objects(self, query,  top_n=10):
        query_embedding = self.client.embeddings.create(input = [query], model=self.model).data[0].embedding
        similarities = []
        
        for obj in self.embeddings:
            similarity = self.cosine_similarity(query_embedding, obj["embedding"])
            similarities.append((obj, similarity))
        
        # Sort by similarity in descending order
        similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
        return similarities[:top_n]
    
    




class ChatGPTAnalyzer:
    def __init__(self, yaml_content):
        """Initialize the analyzer with YAML content."""
        try:
            self.api_key = get_api_key()
            print("OpenAI API Key retrieved successfully.")
        except (FileNotFoundError, ValueError) as e:
            print(e)
            sys.exit(1)  # Exit with an error code

        self.yaml_content = yaml_content
        self.messages = [
            {
                "role": "system",
                "content": f"You are an expert in analyzing YAML files for system design. Here is the YAML file:\n---\n{self.yaml_content}\n---",
            }
        ]

    def _construct_prompt(self, user_prompt):
        """Construct the full prompt with YAML content."""
        return f"""
The following is a YAML file describing a system design component in Capella:
---
{self.yaml_content}
---
{user_prompt}
Please format the response in .html format.
"""

    def add_prompt(self, user_prompt):
        """Add a user prompt to the conversation."""
        user_prompt = user_prompt + "Format the response in .html format."
        self.messages.append({"role": "user", "content": user_prompt})

    def get_response(self):
        """Send messages to ChatGPT and get a response."""
        try:
            client = OpenAI(
            api_key=self.api_key,  # This is the default and can be omitted
            )
            response = client.chat.completions.create(
                messages=self.messages,
                model="gpt-4o",
            )
            assistant_message =  response.choices[0].message.content
            self.messages.append({"role": "assistant", "content": assistant_message})
            return assistant_message
            
        except Exception as e:
            return f"Error communicating with OpenAI API: {e}"






def interactive_chat(analyzer):
    """
    Asynchronous interactive chat loop.
    """
    user_prompt = ""
    print("Starting interactive chat...")
    while True:
        # Get user input
        user_prompt = input("Enter your prompt (or type 'exit' to quit):")
        
        # Exit condition
        if user_prompt.lower() == "exit":
            print("Exiting...")
            break

        # Add the prompt to the analyzer and get a response
        analyzer.add_prompt(user_prompt)
        response = analyzer.get_response()

        # Display the response
        display(Markdown(f"**Your Prompt:**\n\n{user_prompt}\n"))
        if "<table" in response or "<html" in response:
            display(HTML(response))  # Render HTML
        else:
            display(Markdown(f"**ChatGPT Response:**\n\n{response}\n"))
