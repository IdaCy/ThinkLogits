from google import genai
from pydantic import BaseModel
import os # Import os to get API key from environment variable

# class Verification(BaseModel):
#   model_anwer: str

# prompt = f"""Below is a model completion to a MCQ question.
#             Please search for the final MCQ selection of the model (answer) and output it in the specified format.
            
#             Model completion:
#             {completion}
#             """

# client = genai.Client(api_key="AIzaSyDBCzGVNgTvlteIy4woOwesTLoVwh4jnqI")
# response = client.models.generate_content(
#     model='gemini-2.0-flash',
#     contents= prompt,
#     config={
#         'response_mime_type': 'application/json',
#         'response_schema': Verification,
#     },
# )
# # Use the response as a JSON string.
# print(response.text)

# # Use instantiated objects.
# my_recipes: list[Recipe] = response.parsed


class CapitalCity(BaseModel):
    city_name: str
    population: int
    mayor: str

prompt = f"""What is the capital of France?
            """

client = genai.Client(api_key="AIzaSyDBCzGVNgTvlteIy4woOwesTLoVwh4jnqI")
response = client.models.generate_content(
    model='gemini-2.0-flash',
    contents= prompt,
    config={
        'response_mime_type': 'application/json',
        'response_schema': CapitalCity,
    },
)
# Use the response as a JSON string.
print(response)

print("stop")