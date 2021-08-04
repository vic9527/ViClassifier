def interface(url, data):
    import requests
    import json

    head = {"Content-Type": "application/json; charset=UTF-8"}

    # dumps：将python字典对象解码为json数据
    post_json = json.dumps(data)

    # 访问服务
    return requests.post(url=url, data=post_json, headers=head)




if __name__ == '__main__':
    post_url = "http://127.0.0.1:8888"
    post_data = {"image": 112, "name": 1}

    response = interface(post_url, post_data)

    print('status_code: ', response.status_code)  # 打印状态码
    # print('url: ', response.url)          # 打印请求url
    # print('headers: ', response.headers)      # 打印头信息
    # print('cookies: ', response.cookies)      # 打印cookie信息
    print('text: ', response.text)  # 以文本形式打印网页源码
    # print('content: ', response.content) #以字节流形式打印
