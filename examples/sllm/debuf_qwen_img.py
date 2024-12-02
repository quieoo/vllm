from PIL import Image
from qwen_vl_utils import process_vision_info

# 定义本地图像路径列表
IMAGE_PATHS = [
    "/public/home/shenzhaoyan002/zhu/data/duck.jpg",  # 替换为实际文件路径
    "/public/home/shenzhaoyan002/zhu/data/lion.jpg",  # 替换为实际文件路径
]
QUESTION = "What is the content of each image?"

# 构建消息
placeholders = [{"type": "image", "image": path} for path in IMAGE_PATHS]
messages = [{
    "role": "system",
    "content": "You are a helpful assistant."
}, {
    "role": "user",
    "content": [
        *placeholders,
        {
            "type": "text",
            "text": QUESTION
        },
    ],
}]

if process_vision_info:
    max_model_len=32768
else:
    max_model_len=4096

# max_model_len=32768 if process_vision_info is None else 4096
print(f"max model len {max_model_len}")

# 调用 process_vision_info
print(f"Input to process_vision_info: {messages}")
try:
    image_data, _ = process_vision_info(messages)
    print(f"Output from process_vision_info: {image_data}")
except Exception as e:
    print(f"Error in process_vision_info: {e}")
    raise

