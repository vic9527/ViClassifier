"""
FastAPI：Python 世界里最受欢迎的异步框架
https://www.cnblogs.com/traditional/p/14733610.html

Python Web 框架之FastAPI
https://www.jianshu.com/p/d01d3f25a2af

请不要把 Flask 和 FastAPI 放到一起比较
https://greyli.com/flask-fastapi/

"""


from fastapi import FastAPI
from pydantic import BaseModel
import app


APP = FastAPI()


class PostData(BaseModel):
    name: str
    image: str


@APP.post('/')
def main(post_data: PostData):
    code, data = 50000, None
    try:
        # print(post_data.image, post_data.name)
        request_dict = {'image': post_data.image, 'name': post_data.name}
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

    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_fastapi:APP", host="0.0.0.0", port=8000)  # 端口为8000
