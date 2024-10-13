import google.generativeai as genai
import PIL.Image




chat = None

def Start_a_Chat():
    genai.configure(api_key='AIzaSyC5AP9fgrxLQxbjf8mua-a7xS-LxV6kIeY')
    global  chat
    import Data_Recommendation
    model = genai.GenerativeModel('gemini-1.5-flash')
    chat = model.start_chat(history=[])
    # json_data = Data_Recommendation.data[:300].to_json(orient='records')

    # initial_prompt = f"Here's some data:\n  {json_data} \n Please remember the data as u are the centeral management of entire store ."
    # chat.send_message(initial_prompt)

def new_chat():
    chat = model.start_chat(history=[])

def images(path):
    img = PIL.Image.open(path)
    prompt2 = "Identify the clothing items in the image and provide a good user written prompt who is looking for the cloth shown in the image. Be professional as you are an expert in clothing."
    response = chat.send_message([prompt2,img])
    print("Model Response Image : "+response.text)
    return response.text

def give_indices(user_prompt):
    prompt1 = user_prompt
    print(user_prompt)
    prompt2 = "Given these above requirements provide me with product id that matches the user need "
    prompt3="The Output format should list of product id For ex [1,2,3,4,5,8], If there is no product then give most matching product ids based on name or description list The output should be just a list nothing else"
    prompt=(f"User For men :{prompt1} , "
            f"{prompt2} ,{prompt3}")
    response = chat.send_message(prompt)
    print("Model Response : "+response.text)
    li = response.text.split(',')[1:-1]
    return li

def getLLMResponse(user_prompt,data,score):
    prompt = f"New Question. You are an expert in clothing. Recommend the best products based on this search query: {user_prompt}. Here are the top products: {data}. Be professional and Do not return product data in response but just give a gist of product type. Do not make user feel uncomfortable with your response. If {score} is less than 0.3, only then ask user in a nice way to ask query related to clothing more specifically for personalized recommendations. Your current query is not directly related to clothing. If user query is friendly like \"Hi or Hello , How are you?\" then respond it in a nice way asking \"How can I assist you today?\". Do not help user with query which is not related to clothing."
    response = chat.send_message(prompt)
    print(response.text)
    return response.text