"""
Flask获取请求参数
https://www.cnblogs.com/wsg-python/articles/9988295.html

Flask Response响应(flask中设置响应信息的方法，返回json数据的方法)
https://www.cnblogs.com/leijiangtao/p/4162829.html

"""

from flask import request, Flask, jsonify
# import json
import app

APP = Flask(__name__)


@APP.route("/", methods=['POST'])
def main():
    code, data = 50000, None
    try:
        # 如果是请求体的数据不是表单格式的（如json格式），可以通过request.data获取。
        # json.dumps(字典）  将python的字典转换为json字符串
        # json.loads(字符串）  将字符串转换为python中的字典
        # request_dict = json.loads(request.data)

        # request.json首先判断Content-Type是不是application/json，如果是，做json.load；如果不是，直接返回None。
        request_dict = request.json
        # print(request_dict)

        data = app.process(request_dict)

    except Exception as err:
        code = 40000
        print('error: ', err)
        pass
    else:
        code = 20000

    result = {
        "code": str(code),
        "data": str(data)
    }

    # from flask import make_response
    # response = make_response(result)
    # response.status = "200 OK"  # 状态码
    # response.headers['Content-Type'] = "application/json"
    # return response

    # jsonify帮助转为json数据，并设置响应头 Content-Type 为application/json
    # jsonify除了将字典转换成json对象，还将对象包装成了一个Response对象
    return jsonify(result)


if __name__ == "__main__":
    APP.run(debug=False, threaded=False, host='0.0.0.0', port=5000)  # 端口为5000
