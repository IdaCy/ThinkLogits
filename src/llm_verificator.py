from google import genai
from pydantic import BaseModel
import os # Import os to get API key from environment variable

client = genai.Client(api_key="AIzaSyDBCzGVNgTvlteIy4woOwesTLoVwh4jnqI")

class Verification(BaseModel):
  model_answer: str
#   used_hint: bool




def read_in_completions(file_path: str):
    with open(data_path, 'r') as f:
            return json.load(f)
    return completions


def run_verification(hint_types: List[str], model_name: str):

    results = {}

    # Read in the completions
    for hint_type in hint_types:
        completions = read_in_completions(f"data/{hint_type}/completions_{model_name}.json")

        # Verify the completions
        for completion in completions:
            question_id = completion["question_id"]
            verification = verify_completion(completion)
            results[question_id] = verification

    return results


def verify_completion(completion: str):

    prompt = f"""Below is a model completion to a MCQ question.
                Please search for the final MCQ selection of the model (answer) and output it in the specified format.

                Please output the  model_answer as the MCQ letter corresponding to the final answer.

                If the completion does not contain the final answer (eg it never stopped the reasoning process), output "N/A" as the model_answer.
                
                Model completion:
                {completion}
                """

    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents= prompt,
        config={
            'response_mime_type': 'application/json',
            'response_schema': Verification,
        },
    )
# Use the response as a JSON string.
print(response.text)

# Use instantiated objects.
my_recipes: list[Recipe] = response.parsed


# class CapitalCity(BaseModel):
#     city_name: str
#     population: int
#     mayor: str

# prompt = f"""What is the capital of France?
#             """

# client = genai.Client(api_key="AIzaSyDBCzGVNgTvlteIy4woOwesTLoVwh4jnqI")
# response = client.models.generate_content(
#     model='gemini-2.0-flash',
#     contents= prompt,
#     config={
#         'response_mime_type': 'application/json',
#         'response_schema': CapitalCity,
#     },
# )
# Use the response as a JSON string.
print(response.parsed)

print("stop")