import os
from azure.ai.contentsafety import ContentSafetyClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from azure.ai.contentsafety.models import AnalyzeTextOptions


def analyze_text(text):
    # analyze text
    key = '226570f368654761b52f446ac378ec81'
    endpoint = 'https://23f-moderator-eastus.cognitiveservices.azure.com/'


    # Create an Azure AI Content Safety client
    client = ContentSafetyClient(endpoint, AzureKeyCredential(key))

    # Contruct request
    request = AnalyzeTextOptions(text=text)

    # Analyze text
    try:
        response = client.analyze_text(request)
    except HttpResponseError as e:
        print("Analyze text failed.")
        if e.error:
            print(f"Error code: {e.error.code}")
            print(f"Error message: {e.error.message}")
            raise
        print(e)
        raise

    output_dict = {'Hate severity': response.hate_result.severity,'SelfHarm severity': response.self_harm_result.severity,'Sexual severity': response.sexual_result.severity,'Violence severity': response.violence_result.severity}
    return output_dict


import os
import json

#dirname = '/home/elicer/honest_llama/validation/myhatefulxplain_greedy/'
dirname = '/home/elicer/honest_llama/validation/'
filelist = os.listdir(dirname)
for file_in in filelist:
    if 'hatexplain_' not in file_in or '.json' not in file_in:
        continue
    file = dirname + file_in

    print(f"open file: {file_in}")
    with open(file,'r') as f:
        x_json = json.load(f)

    # x_prompt = list(map(lambda x:x['prompt'], x_json))
    # 
    #print(x_json[0]['prompt'])
    #x_prompt = list(map(lambda x:' '.join(x['prompt']['post_tokens']),x_json))
    x_output = list(map(lambda x:x['output'], x_json))
    #x_concat = list(map(lambda x:' '.join(x['prompt']['post_tokens'])+x['output'], x_json))
    from tqdm import tqdm

    prompt_mod = []
    import time
    for input_line in tqdm(x_output):
        try:
            response = analyze_text(input_line)
        except:
            continue
        #print(response)
        prompt_mod.append(response)
        #print(output)
        time.sleep(0.7)
    moderate_='moderate_result'
    with open(f'moderate_result_hatexpliain_new/{file_in}_moderate_concat.json', "w") as json_file:
        json.dump(prompt_mod, json_file)

