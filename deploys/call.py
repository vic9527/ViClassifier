# import call_httpx as call
import call_requests as call

post_url = "http://127.0.0.1:8888"
post_data = {"image": 112, "name": "sdf"}

response = call.interface(post_url, post_data)
print('text: ', response.text)
