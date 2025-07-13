import openai
import os
import openai
import os

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    api_key = input("Enter your OpenAI API key: ").strip()
    os.environ["OPENAI_API_KEY"] = api_key
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set")
response = openai.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant responding in Dutch."},
        {"role": "user", "content": "Can you print 10 random words?"}
    ]
)

print(response.choices[0].message.content)