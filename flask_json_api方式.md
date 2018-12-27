### flask服务器端

引入文件：

```
#coding=utf-8
from flask import Flask,request
from util import image_to_base64,base64_to_image
from flask_cors import *
import json
project_root_path = os.path.join(os.path.dirname(__file__), '../')
app = Flask(__name__)
CORS(app)
```

路由函数：

```
@app.route('/cv/v1/face_detect',methods=['GET','POST'])
def face_detect():
    img_base64 = request.json['img_base64']
    img_PIL = base64_to_image(img_base64)
    # 此处为逻辑处理过程
    
    #返回内容，返回json
    jsonresult = {}
    jsonresult['base64'] = image_to_base64(img_PIL)
    dataPModel = {"msg": "face detect success", "code": 200, "result": jsonresult}
    return json.dumps(dataPModel)
```

main函数起服务：

```
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8502,use_reloader=False)
```

### base64与PIL转换函数

```
def image_to_base64(img):
    output_buffer = BytesIO()
    img.save(output_buffer, format='JPEG')
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data)
    return base64_str.decode()

def base64_to_image(base64_str):
    base64_data = re.sub('^data:image/.+;base64,', '', base64_str)
    byte_data = base64.b64decode(base64_data)
    image_data = BytesIO(byte_data)
    img = Image.open(image_data)
    return img
```

### 测试程序

当服务启动完成，需要测试api是否调试正确，测试程序如下：

```
from io import BytesIO
from PIL import Image
import base64
import json
import re
import requests
from time import time

#发送代码
def json_send(dataPModel,url):
    headers = {"Content-type": "application/json", "Accept": "text/plain", "charset": "UTF-8"}
    response = requests.post(url=url, headers=headers, data=json.dumps(dataPModel))
    return json.loads(response.text)

if __name__ == "__main__":
    url = 'http://192.168.102.198:8502/cv/v1/face_detect'
    img = Image.open('3.jpg')
    # print(img.size)
    img_base64 = image_to_base64(img)
    dataPModel = {"img_base64": img_base64}
    a = time()
    for i in range(100):
        result = json_send(dataPModel,url)
        base64_to_image(result['result']['base64'])

    print(time()-a)
```

