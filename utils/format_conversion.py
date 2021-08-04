"""
python PIL/cv2/base64相互转换
https://blog.csdn.net/haveanybody/article/details/86494063
"""


def base64_read(img_path):
    import base64
    with open(img_path, "rb") as f:
        # base64_str = base64.b64encode(f.read())
        base64_str = base64.b64encode(f.read()).decode()
    return base64_str


def base64_to_cv2(b64str):
    # base64转cv2
    import base64
    import numpy
    import cv2
    img_string = base64.b64decode(b64str.encode('utf8'))
    nparr = numpy.fromstring(img_string, numpy.uint8)
    cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return cv2_img


def base64_to_pil(b64str):
    # base64转pil
    from io import BytesIO
    from PIL import Image
    import base64
    img_string = base64.b64decode(b64str.encode('utf8'))
    img_io = BytesIO(img_string)
    pil_img = Image.open(img_io)
    return pil_img


def cv2_to_pil(cv2_img):
    # cv2转pil
    import cv2
    from PIL import Image
    # cv2_img = cv2.imread("cv2.jpg")  # 默认BGR
    return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))


def pil_to_cv2(pil_img):
    # pil转cv2
    import cv2
    import numpy
    # pil_img = Image.open("pil.jpg")  # 默认RGB
    # pil_img = Image.open("pil.jpg").convert('RGB')
    return cv2.cvtColor(numpy.asarray(pil_img), cv2.COLOR_RGB2BGR)


def cv2_to_base64(cv2_img):
    # cv2转base64
    import base64
    import cv2
    cv2str = cv2.imencode('.jpg', cv2_img)[1].tostring()
    b64str = base64.b64encode(cv2str)
    return b64str

# def cv2_to_base64(cv2_img):
#     image_code = cv2.imencode('.jpg', cv2_img)[1]
#     b64str = str(base64.b64encode(image_code))[2:-1]
#     return b64str


def pil_to_base64(pil_img):
    # pil转base64
    import base64
    from io import BytesIO
    img_buffer = BytesIO()
    pil_img.save(img_buffer, format='JPEG')
    byte_data = img_buffer.getvalue()
    b64str = base64.b64encode(byte_data)
    return b64str


def url_to_cv2(url):
    # URL到图片
    import numpy as np
    import urllib
    import cv2
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    response = urllib.request.urlopen(url)
    # bytearray将数据转换成（返回）一个新的字节数组
    # asarray 复制数据，将结构化数据转换成ndarray
    img_array = np.array(bytearray(response.read()), dtype=np.uint8)
    # cv2.imdecode()函数将数据解码成Opencv图像格式
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    # return the image
    return image

