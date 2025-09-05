"""

This module handles interactions with Azure OpenAI (AOAI) directly through the SDK without using orchestrators or frameworks.

"""

import os
from typing import List, Dict, Optional, Union
from dotenv import load_dotenv
from pydantic import BaseModel
from openai import AzureOpenAI
import openai
from openai.types import CreateEmbeddingResponse
import base64

# Load environment variables
load_dotenv()

# Azure OpenAI configuration
aoai_deployment = os.environ.get("AOAI_DEPLOYMENT_NAME")
aoai_key = os.environ.get("AOAI_API_KEY")
aoai_endpoint = os.environ.get("AOAI_ENDPOINT")
aoai_api_version = os.environ.get("AOAI_API_VERSION")

print(f"AOAI Endpoint: {aoai_endpoint}")
print(f"AOAI Deployment: {aoai_deployment}")
print(f"AOAI Key: {aoai_key[:5] + '*' * (len(aoai_key) - 5) if aoai_key else None}")

# Initialize Azure OpenAI client
try:
    aoai_client = AzureOpenAI(
        azure_endpoint=aoai_endpoint,
        api_key=aoai_key,
        api_version=aoai_api_version
    )
except Exception as e:
    print(f"Failed to initialize Azure OpenAI client: {e}")
    raise

def generate_embeddings_aoai(text: str, model: str = "text-embedding-ada-002") -> Optional[CreateEmbeddingResponse]:
    """
    Generate embeddings for the given text using Azure OpenAI.

    Parameters:
    - text (str): The input text to generate embeddings for.
    - model (str): The name of the embedding model to use. Defaults to "text-embedding-ada-002".

    Returns:
    - Optional[CreateEmbeddingResponse]: The embedding response from Azure OpenAI, or None if an error occurs.
      The response contains the generated embeddings and related information.

    This function takes a piece of text and converts it into a numerical representation (embedding)
    using the specified Azure OpenAI model. These embeddings can be used for various natural language
    processing tasks such as semantic search or text classification.
    """
    try:
        response = aoai_client.embeddings.create(input=[text], model=model)
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return None

def inference_aoai(messages: List[Dict[str, Union[str, List[Dict[str, Union[str, Dict[str, str]]]]]]], deployment: str) -> dict:
    """
    Perform a basic inference task using Azure OpenAI.

    Parameters:
    - messages (List[Dict]): A list of message dictionaries. Each dictionary should have a 'role' key
      (either 'system', 'user', or 'assistant') and a 'content' key. The 'content' can be either a string
      for text-only messages or a list of dictionaries for messages that include images.
    - deployment (str): The name of the Azure OpenAI deployment to use.

    Returns:
    - dict: The response from Azure OpenAI, or None if an error occurs. The response includes
      the generated text and other metadata.

    This function sends the provided messages to Azure OpenAI and returns the model's response.
    It can handle both text-only inputs and inputs that include images. For image inputs, the message
    content should be a list where one of the items is a dictionary with an 'image_url' key.
    """
    try:
        response = aoai_client.chat.completions.create(
            model=deployment,
            messages=messages
        )
        return response
    except Exception as e:
        print(f"Error in inference: {e}")
        return None

def inference_structured_output_aoai(messages: List[Dict[str, Union[str, List[Dict[str, Union[str, Dict[str, str]]]]]]], deployment: str, schema: BaseModel) -> dict:
    """
    Perform an inference task with structured output using Azure OpenAI.

    Parameters:
    - messages (List[Dict]): A list of message dictionaries. Each dictionary should have a 'role' key
      (either 'system', 'user', or 'assistant') and a 'content' key. The 'content' can be either a string
      for text-only messages or a list of dictionaries for messages that include images.
    - deployment (str): The name of the Azure OpenAI deployment to use.
    - schema (BaseModel): A Pydantic model that defines the structure of the expected output.

    Returns:
    - dict: The parsed response from Azure OpenAI, or None if an error occurs. The response includes
      the generated content structured according to the provided schema.

    This function sends the provided messages to Azure OpenAI and returns the model's response
    parsed according to the given schema. It can handle both text-only inputs and inputs that
    include images, similar to the inference_aoai function.
    """
    try:
        completion = aoai_client.beta.chat.completions.parse(
            model=deployment,
            messages=messages,
            response_format=schema,
        )
        print("Structured output inference completed")
        print(f"Completion content: {completion.choices[0].message.content}")
        print(f"Parsed event: {completion.choices[0].message.parsed}")
        return completion
    except Exception as e:
        print(f"Error in structured output inference: {e}")
        return None

def tool_inference_aoai(messages: List[Dict[str, str]], deployment: str, tools: List[Dict]) -> dict:
    """
    Perform an inference task using tools (functions) with Azure OpenAI.

    Parameters:
    - messages (List[Dict]): A list of message dictionaries. Each dictionary should have a 'role' key
      (either 'system', 'user', or 'assistant') and a 'content' key containing the message text.
    - deployment (str): The name of the Azure OpenAI deployment to use.
    - tools (List[Dict]): A list of tool definitions. Each tool should be a dictionary describing
      a function that the model can call.

    Returns:
    - dict: The response from Azure OpenAI, or None if an error occurs. The response includes
      the generated text, any tool calls made by the model, and other metadata.

    This function allows the Azure OpenAI model to use defined tools (functions) as part of its
    response generation process. It's useful for tasks where the model might need to call
    external functions or APIs to complete a task.
    """
    try:
        response = aoai_client.chat.completions.create(
            model=deployment,
            messages=messages,
            tools=tools
        )
        print("Tool inference completed")
        print(f"Function Call: {response.choices[0].message.tool_calls[0].function}")
        return response
    except Exception as e:
        print(f"Error in tool inference: {e}")
        return None

def stream_inference_aoai(messages: List[Dict[str, str]], deployment: str) -> str:
    """
    Perform a streaming inference task using Azure OpenAI.

    Parameters:
    - messages (List[Dict]): A list of message dictionaries. Each dictionary should have a 'role' key
      (either 'system', 'user', or 'assistant') and a 'content' key containing the message text.
    - deployment (str): The name of the Azure OpenAI deployment to use.

    Returns:
    - str: The full response content as a string, or an empty string if an error occurs.

    This function sends the provided messages to Azure OpenAI and streams the response back.
    It prints each chunk of the response as it's received and returns the full response at the end.
    This is useful for long-form content generation where you want to see the output in real-time.
    """
    try:
        response = aoai_client.chat.completions.create(
            model=deployment,
            messages=messages, 
            stream=True
        )

        full_response = ""
        for chunk in response:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta and delta.content is not None:
                    full_response += delta.content
                    print(delta.content, end='', flush=True)
        print('\n\n')

        return full_response
    except Exception as e:
        print(f"Error in streaming chat completion: {e}")
        return ""

def example_generate_embeddings():
    """Example usage of generate_embeddings_aoai function."""
    text = "Hello, world!"
    response = generate_embeddings_aoai(text)
    if response:
        print(f"Embedding for '{text}': {response.data[0].embedding[:5]}...")  # Print first 5 values
    else:
        print("Failed to generate embeddings")

def example_basic_inference():
    """Example usage of inference_aoai function for basic text inference."""
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant"},
        {"role": "user", "content": "What is the capital of France?"}
    ]
    response = inference_aoai(messages, aoai_deployment)
    if response:
        print(f"Basic inference response: {response.choices[0].message.content}")
    else:
        print("Failed to process basic inference")

def example_structured_output():
    """Example usage of inference_structured_output_aoai function."""
    class CalendarEvent(BaseModel):
        name: str
        date: str
        participants: list[str]

    messages = [
        {"role": "system", "content": "Extract the event information."},
        {"role": "user", "content": "Alice and Bob are going to a science fair on Friday."},
    ]

    result = inference_structured_output_aoai(messages, aoai_deployment, CalendarEvent)
    if result:
        new_event = CalendarEvent(**result.choices[0].message.parsed.dict())
        print(f"Event name: {new_event.name}")
        print(f"Event date: {new_event.date}")
        print(f"Event participants: {new_event.participants}")
    else:
        print("Failed to process structured output")

def example_tool_inference():
    """Example usage of tool_inference_aoai function."""
    class GetDeliveryDate(BaseModel):
        order_id: str

    tools = [openai.pydantic_function_tool(GetDeliveryDate)]

    messages = [
        {"role": "system", "content": "You are a helpful customer support assistant. Use the supplied tools to assist the user."},
        {"role": "user", "content": "Hi, can you tell me the delivery date for my order #12345?"}
    ]

    response = tool_inference_aoai(messages, aoai_deployment, tools)
    if response:
        function_call = response.choices[0].message.tool_calls[0].function
        print(f"Function called: {function_call.name}")
        print(f"Arguments: {function_call.arguments}")
        print(f"Total tokens used: {response.usage.total_tokens}")
    else:
        print("Failed to process tool inference")

def example_stream_inference():
    """Example usage of stream_inference_aoai function."""
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant"},
        {"role": "user", "content": "What color is the sky?"},
    ]

    full_response = stream_inference_aoai(messages, aoai_deployment)
    print("Streaming completed successfully")
    print(f"Full response: {full_response}")

def example_image_processing():
    """Example usage of inference_aoai function for image processing."""
    with open("D:/temp/tmp/buttermilk-pancakes.jpg", "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

    image_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe what you see in this image."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
            ]
        }
    ]
    image_response = inference_aoai(image_messages, aoai_deployment)
    if image_response:
        print(f"Image description: {image_response.choices[0].message.content}")
    else:
        print("Failed to process image")

def example_structured_image_processing():
    """Example usage of inference_structured_output_aoai function for image processing with structured output."""
    with open("D:/temp/tmp/buttermilk-pancakes.jpg", "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

    class OutputStructure(BaseModel):
        text: str
        image_insights: str

    image_prompt = """You will be given an image that is one or more pages of a document, along with some analysis of the overall document. 

    Output the following fields:
    text: The verbatim text of the page in markdown format. 
    image_insights: All insights or information that can be gleaned from the images on the page and the relationship to the text. 

    If there are no images, output 'na' for image_insights."""

    structured_image_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": image_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
            ]
        }
    ]

    structured_image_response = inference_structured_output_aoai(structured_image_messages, aoai_deployment, OutputStructure)
    if structured_image_response:
        parsed_response = OutputStructure(**structured_image_response.choices[0].message.parsed.dict())
        print(f"Structured Image Text: {parsed_response.text}")
        print(f"Structured Image Insights: {parsed_response.image_insights}")
    else:
        print("Failed to process structured image output")


if __name__ == "__main__":
    print("Running examples for Azure OpenAI functions:")
    
    print("\n1. Generating Embeddings:")
    example_generate_embeddings()
    
    print("\n2. Basic Inference:")
    example_basic_inference()
    
    print("\n3. Structured Output Inference:")
    example_structured_output()
    
    print("\n4. Tool Inference:")
    example_tool_inference()
    
    print("\n5. Stream Inference:")
    example_stream_inference()
    
    print("\n6. Image Processing:")
    example_image_processing()
    
    print("\n7. Structured Image Processing:")
    example_structured_image_processing()

    print("\nAll examples completed.")