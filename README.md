<div align=center>
<!--  <img src="./logo.png" width="25%" />  -->
</div>

![](./logo2.png)

**English | [简体中文](README_cn.md)**

# ViClassifier 

ViClassifier (Vision Classifier), a simple and easy-to-use classification toolbox based on pytorch.

(development based on other frameworks is planned)

# DevLogs

[2021/08/04] Update the first version (v0.0.1) of the code, which may be rough and is constantly being optimized.

# Requirements

Please install the following required modules.

> pip install [torch](https://pytorch.org/get-started/locally/) opencv-python tqdm
>

The following modules are optional and can be selectively installed during deployment.

> pip install flask fastapi tornado requests httpx
> 



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

# idx_to_class = {0: "bad", 1: "good"}

model_path = r'model.pth'
image_path = r'test.jpg' 

model = vic.main.VicModel(model_path)
result = model.infer_image(image_path)

# print(idx_to_class[result[0]], result[1])
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
</div>

![](reward-wx.png)
