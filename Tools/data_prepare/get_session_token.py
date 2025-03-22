import requests
api_key = ""
url = f"https://tile.googleapis.com/v1/createSession?key={api_key}"
headers = {"Content-Type": "application/json"}
data = {
    "mapType": "satellite",
    "language": "en-US",
    "region": "US"
}

response = requests.post(url, headers=headers, json=data)

if response.status_code == 200:
    print("Session Created:", response.json())  # 解析返回的 JSON
else:
    print("Error:", response.status_code, response.text)