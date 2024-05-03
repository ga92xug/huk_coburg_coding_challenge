import requests

def make_prediction(text, model_type):
    url = 'http://127.0.0.1:5000/predict'
    headers = {'Content-Type': 'application/json'}
    data = {
        'text': text,
        'model_type': model_type
    }
    
    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        return {'error': 'Failed to get response from the server', 'status_code': response.status_code}


def main():
    result = make_prediction("I love this company.", "distil_bert")
    print(result)


if __name__ == '__main__':
    main()