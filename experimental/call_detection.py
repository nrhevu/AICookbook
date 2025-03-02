import requests

url = "http://localhost:8000/upload"

payload = {}
files=[
  ('file',('pngtree-fruit-photography-image_881781.jpg',open('/C:/Users/ADMIN/Downloads/pngtree-fruit-photography-image_881781.jpg','rb'),'image/jpeg'))
]
headers = {}

response = requests.request("POST", url, headers=headers, data=payload, files=files)

print(response.text)

"""curl --location 'http://localhost:8000/upload' \
--form 'file=@"/C:/Users/ADMIN/Downloads/pngtree-fruit-photography-image_881781.jpg"'"""