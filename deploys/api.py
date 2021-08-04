import api_flask
api_flask.APP.run(debug=False, threaded=False, host='0.0.0.0', port=5000)  # 端口为5000

# import uvicorn
# uvicorn.run("api_fastapi:APP", host="0.0.0.0", port=8000)  # 端口为8000
#
# import tornado.ioloop
# import api_tornado
# application = api_tornado.make_app()
# application.listen(8888)
# print("Setting OK!")
# tornado.ioloop.IOLoop.current().start()


