import os
from openai import OpenAI
from openai import AzureOpenAI
import base64


client = OpenAI(
    api_key="input your own key",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
# client = OpenAI(
#     api_key="xx",
#     base_url="https://api3.apifans.com/v1",
# )



def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def get_response(messages, model_name):
    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
    )
    return completion.model_dump()


def chat_with_llm(prompt, image_paths=None):

    # 根据是否有图片选择不同的模型和消息格式
    if image_paths:
        # Use multimodal model for image+text input
        model_name = "qwen-vl-plus-latest"
        # model_name = "gpt-4o"

        # Prepare message content with text and multiple images
        message_content = [{"type": "text", "text": prompt}]

        base64_image = encode_image(image_paths)
        message_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
        })

        messages = [
            {"role": "user", "content": message_content}
        ]
    else:
        # 仅文本输入使用文本模型
        # model_name = "gpt-4o"
        model_name = "qwen-plus-latest"
        messages = [
            {"role": "user", "content": prompt}
        ]

    response = client.chat.completions.create(
        model=model_name,
        messages=messages
    )

    text = response.choices[0].message.content

    return text


def chat_with_llm_images(prompt, image_paths=None):

    if image_paths:

        message_content = [{"type": "text", "text": prompt}]

        for image_path in image_paths:
            base64_image = encode_image(image_path)
            message_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            })

        messages = [
            {"role": "user", "content": message_content}
        ]
    else:
        messages = [
            {"role": "user", "content": prompt}
        ]

    response = client.chat.completions.create(
        # model="gpt-4o",
        model="qwen-vl-plus-latest",
        messages=messages
    )

    text = response.choices[0].message.content

    return text


if __name__ == '__main__':
    # 示例使用
    # 仅文本输入
    #text_prompt = "What is the capital of France?"
    #a = chat_with_llm(text_prompt)
    #print(a)

    # 图文输入
    image_prompt = f"""  

                                    Select one action from [Go Up, Go Down, Go Forward, Turn Left, Turn Right, Go left, Go right]) following guidelines below: 
                                    -Follow the exploitation advice or exploration advice will help you to success.
                                    -Exploitation advice: Select action Go Up will help you to approach the target  with a probability of 95%.
                                    -Exploration advice: Choosing action Go Down helps you explore the surrounding environment.
                                    -If moving forward will hit the wall, do not choose the action “Go forward”, choose "Go left" or "Go Right" actions instead.
                                    Only return the name of the action you selected.
                                    """
    a = chat_with_llm(image_prompt, "target.png")
    print(a)


