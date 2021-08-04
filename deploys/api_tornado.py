"""
python3 tornado 设计 RESTful API
https://blog.csdn.net/dangsh_/article/details/79195230

https://tornado-zh-cn.readthedocs.io/zh_CN/latest/httpserver.html
# listen: simple single-process:
server = HTTPServer(app)
server.listen(8888)
IOLoop.current().start()
In many cases, tornado.web.Application.listen can be used to avoid the need to explicitly create the HTTPServer.

# bind/start: simple multi-process:
server = HTTPServer(app)
server.bind(8888)
server.start(0)  # Forks multiple sub-processes
IOLoop.current().start()

tornado之获取参数
https://www.cnblogs.com/quzq/p/10975766.html

tornado 中读取访问参数（get和post方式）
https://blog.csdn.net/weixin_43883022/article/details/102832114
"""


import tornado.web
import tornado.escape
import tornado.ioloop
import app


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Hello , SELECT\n")

    def post(self):
        # self.write("hello , ADD\n")
        code, data = 50000, None
        try:
            data = tornado.escape.json_decode(self.request.body)
            print(data)
            # data = app.process(request_dict)

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

        self.write(result)

    def put(self):
        self.write("hello , UPDATE\n")

    def delete(self):
        self.write("hello , DELETE\n")

def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
    ])

if __name__ == "__main__":
    # # import os
    #
    # settings = {
    #     'debug': True,
    #     # 'static_path': os.path.join(os.path.dirname(__file__), "static"),
    #     # 'template_path': os.path.join(os.path.dirname(__file__), "template")
    # }
    #
    # application = tornado.web.Application([(r"/", MainHandler)], **settings)
    # application.listen(8888)  # 默认端口 8888
    # print("Setting OK!")
    # tornado.ioloop.IOLoop.instance().start()

    APP = make_app()
    APP.listen(8888)
    print("Setting OK!")
    tornado.ioloop.IOLoop.current().start()

