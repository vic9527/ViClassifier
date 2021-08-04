<div align=center>
<!--  <img src="./logo.png" width="25%" />  -->
<img src="https://www.whing.cn/about/tmps/logo2.png" width="100%" />
</div>


**English | [简体中文](README_cn.md)**

# ViClassifier 

ViClassifier (Vision Classifier), a simple and easy-to-use classification toolbox based on pytorch.

(development based on other frameworks is planned)

# DevLogs

[2021/08/04] Update the first version (v0.0.1) of the code, which may be rough and is constantly being optimized.

# Requirements

Please install the following required modules.

> pip install [torch](https://pytorch.org/get-started/locally/)
>
> pip install opencv-python
>
> pip install tqdm

The following modules are optional and can be selectively installed during deployment.

> pip install flask
> 
> pip install fastapi
> 
> pip install tornado
> 
> pip install requests
> 
> pip install httpx



# Installation

'setup.py' will be added to install the 'whl file' in the future.

but now, you could git it to your project directory.

```
git clone https://github.com/vic9527/viclassifier.git
```

# Examples

**Quick to use:**

```
import viclassifier as vic

image_path = r'test.jpg' 
idx_to_class = {0: "bad", 1: "good"}

model = vic.main.VicModel('model.pth')
result = model.infer_image(image_path, idx_to_class)
print(result)
```

# How to Train

Coming soon......

# How to Deploy

Coming soon......

# Rewards & Thanks

Could you treat me a cup of tea？

<div align=left>
<!--  <img src="./reward-wx.png" width="35%" />  -->
<img src="https://www.whing.cn/about/tmps/reward-wx.png" width="30%" />
</div>