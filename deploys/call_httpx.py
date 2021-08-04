"""
比requests更强大python库，让你的爬虫效率提高一倍
https://mp.weixin.qq.com/s/jqGx-4t4ytDDnXxDkzbPqw

HTTPX 基础教程
https://zhuanlan.zhihu.com/p/103824900
"""


def interface(url, data):
    import httpx
    head = {"Content-Type": "application/json; charset=UTF-8"}

    return httpx.request('POST', url, json=data, headers=head)


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
