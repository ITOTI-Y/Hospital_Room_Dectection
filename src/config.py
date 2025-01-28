import pathlib
import torch
from transformers import SwinConfig, Mask2FormerConfig

class Image_Processor_Config:
    def __init__(self):
        self.DO_RESIZE = True
        self.IMAGE_SIZE = (768, 768)
        self.IGNORE_INDEX = 0
        self.NUM_LABELS = len(COLOR_MAP)

class Train_Config:
    def __init__(self):
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.IMAGE_DIR = pathlib.Path('./data/image')
        self.LABEL_DIR = pathlib.Path('./data/label')
        self.SAVE_MODEL_PATH = pathlib.Path('./models/best_model.pth')
        self.SAVE_CHECKPOINT_PATH = pathlib.Path('./models/checkpoint.pth')

        # LOG CONFIG
        self.LOG_DIR = pathlib.Path('/root/tf-logs')
        self.LOG_INTERVAL = 10
        
        # TRAIN CONFIG
        self.TRAIN_SIZE = 0.8
        self.VAL_SIZE = 0.2
        self.SEED = 1
        self.NUM_WORKERS = 8
        self.BATCH_SIZE = 1 # 暂不支持多张图片

        self.LR = 1e-4
        self.LR_FACTOR = 0.1
        self.LR_PATIENCE = 5
        self.LR_VERBOSE = False
        self.LR_MODE = 'min'
        self.WEIGHT_DECAY = 1e-4
        
        self.PATIENCE = 2000
        self.EPOCHS = 1000
        self.SHUFFLE = True

class Swin_Model_Config:
    def __init__(self):
        self.IMAGE_SIZE = Image_Processor_Config().IMAGE_SIZE[0]
        self.PATCH_SIZE = 4
        self.NUM_CHANS = 3
        self.EMBED_DIM = 96
        self.DEPTHS = [2,2,6,2]
        self.NUM_HEADS = [3,6,12,24]
        self.WINDOW_SIZE = 7
        self.OUT_INDICES = [0,1,2]

    def get_config(self):
        config = SwinConfig(
            image_size=self.IMAGE_SIZE,
            patch_size=self.PATCH_SIZE,
            num_channels=self.NUM_CHANS,
            embed_dim=self.EMBED_DIM,
            depths=self.DEPTHS,
            num_heads=self.NUM_HEADS,
            window_size=self.WINDOW_SIZE,
            out_indices=self.OUT_INDICES,
        )
        return config
    
class Mask2Former_Model_Config:
    def __init__(self):
        self.BACKBONE_CONFIG = Swin_Model_Config().get_config()
    
    def get_config(self):
        config = Mask2FormerConfig().from_backbone_config(self.BACKBONE_CONFIG)
        config.num_labels = len(COLOR_MAP)
        return config

class Loss_Config:
    def __init__(self):
        self.SMOOTH = 1e-5
        self.CE_WEIGHT = 1
        self.DICE_WEIGHT = 0.5

class Network_Config:
    def __init__(self):
        self.AREA_THRESHOLD = 60
        self.SKEWNESS = 20
        self.CONNECTION_TYPES = ['Door']
        self.BAN_TYPES = ['Wall', 'Outside', 'Corridor']
        self.ROOM_TYPES = [v for k,v in COLOR_MAP.items() if v not in self.BAN_TYPES]
        self.TRANSPORTATION_TYPES = []
        self.PEDESTRIAN_TYPES = ['Corridor','Outside']
        self.GRID_SIZE = 50

COLOR_MAP = {
    (156, 40, 177): 'Outside',
    (245, 67, 55): 'Pharmacy', 
    (0, 151, 136): 'Reception desk', 
    (103, 58, 183): 'Emergency Department', 
    (33, 150, 243): 'Outpatient Treatment', 
    (3, 169, 245): 'Radiology', 
    (0, 188, 213): 'Pediatric Unit', 
    (207, 216, 221): 'Corridor', 
    (117, 117, 117): 'Stairs', 
    (189, 189, 189): 'Elevator', 
    (158, 158, 158): 'Escalator', 
    (76, 176, 80): 'Landscape Space', 
    (255, 235, 60): 'Wall', 
    (121, 85, 71): 'Door', 
    (140, 195, 75): 'Endoscopy Center', 
    (205, 220, 57): 'Testing Center', 
    (254, 193, 7): 'Gastroenterology', 
    (255, 152, 1): 'Endocrinology', 
    (254, 87, 33): 'Respiratory Medicine', 
    (170, 238, 91): 'Cardiovascular Medicine', 
    (239, 199, 78): 'Ophthalmology', 
    (254, 186, 87): 'Chinese medicine', 
    (251, 133, 97): 'Oral Medicine', 
    (198, 254, 131): 'Otorhinolaryngology', 
    (124, 165, 185): 'Ultrasound Department', 
    (173, 133, 10): 'Functional Inspection Department', 
    (255, 212, 149): 'Rehabilitation Medicine Department', 
    (250, 145, 139): 'Orthopedics', 
    (255, 128, 171): 'Nephrology', 
    (33, 250, 231): 'Physical Medicine and Rehabilitation', 
    (141, 78, 255): 'Hematology', 
    (82, 108, 255): 'Dermatology', 
    (226, 58, 255): 'Gynecology', 
    (100, 139, 56): 'Obstetrics', 
    (179, 115, 189): 'Courtyard', 
    (233, 136, 249): 'Rheumatology Department', 
    (63, 100, 23): 'Neurology', 
    (182, 199, 9): 'Neurosurgery', 
    (241, 222, 166): 'Thoracic Surgery', 
    (222, 173, 229): 'Colorectal Surgery', 
    (165, 45, 36): 'Urology', 
    (187, 24, 81): 'General Surgery', 
    (7, 91, 83): 'Special Outpatient Clinic', 
    (150, 133, 178): 'Dialysis Center', 
    (114, 124, 177): 'Integrated Traditional Chinese and Western Medicine Department', 
    (195, 127, 122): 'General Outpatient Clinic', 
    (48, 122, 113): 'Reproductive Medicine Department', 
    (112, 40, 236): 'Oncology', 
    (142, 158, 246): 'Gastrointestinal Surgery', 
    (241, 190, 185): 'Family Planning Department', 
    (187, 146, 160): 'Occupational Disease Department', 
    (70, 195, 179): 'Psychology Department', 
    (187, 151, 247): 'Beauty Department',
    (255, 255, 255): 'Empty Room'}

# COLOR_LABEL = {
#     'f54337': 'Pharmacy', # 药房
#     # 'ea1e63': 'Chinese Pharmacy',
#     '009788': 'Reception desk', # 挂号处
#     '673ab7': 'Emergency Department',
#     # '3f51b5': 'Infectious Disease Clinic',
#     '2196f3': 'Outpatient Treatment', # 门诊治疗
#     '03a9f5': 'Radiology', # 放射科
#     '00bcd5':'Pediatric Unit', # 儿科
#     'cfd8dd': 'Corridor', # 走廊
#     '757575': 'Stairs', # 楼梯
#     'bdbdbd': 'Elevator', # 电梯
#     '9e9e9e': 'Escalator', # 扶梯
#     '4cb050': 'Landscape Space', # 景观空间
#     'ffeb3c': 'Wall', # 墙
#     '795547': 'Door', # 门
#     '9c28b1': 'Outside', # 室外
#     '8cc34b': 'Endoscopy Center', # 内镜中心
#     'cddc39': 'Testing Center', # 检验中心
#     'fec107': 'Gastroenterology', # 肠胃科
#     'ff9801': 'Endocrinology', # 内分泌科
#     'fe5721': 'Respiratory Medicine', # 呼吸内科
#     'aaee5b': 'Cardiovascular Medicine', # 心血管内科
#     # '607d8b': 'Health Examination Department',
#     'efc74e': 'Ophthalmology', # 眼科
#     'feba57': 'Chinese medicine', # 中医科
#     'fb8561': 'Oral Medicine', # 口腔科
#     'c6fe83': 'Otorhinolaryngology', # 耳鼻喉科
#     '7ca5b9': 'Ultrasound Department', # 超声科
#     'ad850a': 'Functional Inspection Department', # 功能检查科
#     'ffd495': 'Rehabilitation Medicine Department', # 康复医学科
#     'fa918b': 'Orthopedics', # 骨科
#     'ff80ab': 'Nephrology', # 肾内科
#     '21fae7': 'Physical Medicine and Rehabilitation', # 理疗科
#     '8d4eff': 'Hematology', # 血液科
#     '526cff': 'Dermatology', # 皮肤科
#     'e23aff': 'Gynecology', # 妇科
#     '648b38': 'Obstetrics', # 产科
#     'b373bd': 'Courtyard', # 中庭
#     'e988f9': 'Rheumatology Department', # 风湿免疫科
#     '3f6417': 'Neurology', # 神经内科
#     'b6c709': 'Neurosurgery', # 神经外科
#     'f1dea6': 'Thoracic Surgery', # 胸外科
#     'deade5': 'Colorectal Surgery', # 肛肠外科
#     'a52d24': 'Urology', # 泌尿外科
#     'bb1851': 'General Surgery', # 普外科
#     '075b53': 'Special Outpatient Clinic', # 专家门诊
#     '9685b2': 'Dialysis Center', # 透析中心
#     '727cb1': 'Integrated Traditional Chinese and Western Medicine Department', # 中西医结合科
#     'c37f7a': 'General Outpatient Clinic', # 全科门诊
#     '307a71': 'Reproductive Medicine Department', # 生殖医学科
#     '7028ec': 'Oncology', # 肿瘤科
#     '8e9ef6': 'Gastrointestinal Surgery', # 肝胆外科
#     'f1beb9': 'Family Planning Department', # 计划生育科
#     'bb92a0': 'Occupational Disease Department', # 职业病科
#     '46c3b3': 'Psychology Department', # 心理科
#     'bb97f7': 'Beauty Department', # 美容科
# }

