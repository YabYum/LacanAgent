from openai import OpenAI

client = OpenAI(
    base_url="https://oneapi.xty.app/v1",
    api_key="sk-ykouqpo2aV9puc2xEe70Fc6a9c564321A279732507D992D8"
)


def chat(prompt):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system",
             "content": "You are a Large Language Model trained with all available content on Internet."},
            {"role": "user", "content": prompt}
        ]
    )
    # Extracting the content from the response
    first_choice = completion.choices[0]
    first_message = first_choice.message
    message_content = first_message.content
    print(message_content)


def signification(prompt):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system",
             "content": "You are a avatar of the internalized big Other, the sociological concept. The "
                        "prompt you received is basic needs of your ego. You need interpret your needs "
                        "as specific object which you can pursue in the real world, and output in the "
                        "formation of 'I want/should/will/would like to xxx'. For example, if the needs "
                        "is 'hungry', you need to transform the this needs to specific food that can "
                        "satisfied or resolve current needs, if you select bread, you say 'I'm gonna "
                        "eat a bread'. "},
            {"role": "user", "content": prompt}
        ]
    )
    # Extracting the content from the response
    first_choice = completion.choices[0]
    first_message = first_choice.message
    message_content = first_message.content
    return message_content


def gaze(prompt):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system",
             "content": "answer in first-person and short as one sentence."},
            {"role": "user",
             "content": "From others' or the social cultural perspective, how should you look like a" + prompt}
        ]
    )
    # Extracting the content from the response
    first_choice = completion.choices[0]
    first_message = first_choice.message
    message_content = first_message.content
    return message_content


def grasp(symb, prompt):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system",
             "content": "You are ChatGPT acting as the avatar of the big Other, for user's descriptions of you, "
                        "you need to infer which kind of person you are and Respond in first person, "
                        "using 'I'. for example, if you enjoy reading DaBing's books, you might respond that "
                        "'I am an artistic youth'."},
            {"role": "user", "content": "you admire the " + symb + ", and if you look like " + prompt + ", which kind of"
                                        "person are you probably?"}
        ]
    )
    # Extracting the content from the response
    first_choice = completion.choices[0]
    first_message = first_choice.message
    message_content = first_message.content
    return message_content
