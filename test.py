import torch
import torchvision.transforms as T
from PIL import Image

# from transformers import pipeline
# from transformers.image_utils import load_image

repo_dir = "/iag_ad_01/ad/yuanweizhong/huzeyu/dinov3"
device = torch.device('cuda')
model = torch.hub.load(repo_dir,'dinov3_vits16',source='local',weights="/yuanweizhong-tos-volc-engine/dinov3_models/dinov3_vits16_pretrain_lvd1689m-08c60483.pth").to(device)
# model = torch.hub.load("Fanqi-Lin-IR/dinov3_vitl16","dinov3_vitl16")
# state_dict = torch.load("/yuanweizhong-tos-volc-engine/dinov3_models/dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth",map_location = "cpu")


# 2. 定义图像的预处理步骤
transform = T.Compose([
    T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

# 3. 加载并预处理您的图像
# 请将 "path/to/your/image.jpg" 替换为您的图片路径
img = Image.open("/iag_ad_01/ad/yuanweizhong/datasets/senseauto/2024_09_08_07_53_23_pathway_pilotGtParser/camera/center_camera_fov30/1725782003299999850.jpg")
img_tensor = transform(img).unsqueeze(0).to(device)

# 4. 模型推理
with torch.no_grad():
    features = model(img_tensor)
    # DINOv3 的输出是一个特征字典，通常我们关心的是特征向量
    # features_dict = model.forward_features(img_tensor)
    # features = features_dict['x_norm_patchtokens'] # 或者 'x_norm_clstoken'

# 5. 查看输出
# 输出的 features 是图像的特征向量，可以用于下游任务，如分类、聚类、图像检索等
print(features.shape)

# url = "/iag_ad_01/ad/yuanweizhong/datasets/senseauto/2024_09_08_07_53_23_pathway_pilotGtParser/camera/center_camera_fov30/1725782003299999850.jpg"
# image = load_image(url)

# feature_extractor = pipeline(
#     model="Fanqi-Lin-IR/dinov3_vitl16",
#     task="image-feature-extraction", 
# )
# features = feature_extractor(image)