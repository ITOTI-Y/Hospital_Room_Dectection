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
        self.AREA_THRESHOLD = 60 # 面积阈值
        self.SKEWNESS = 20 # 偏度
        self.CONNECTION_TYPES = ['门']
        self.BAN_TYPES = ['墙', '室外', '走廊', '电梯', '扶梯', '楼梯', '空房间', '绿化', '中庭'] # 禁止区域
        self.ROOM_TYPES = [v['name'] for k, v in COLOR_MAP.items() if v['name'] not in self.BAN_TYPES and v['name'] not in self.CONNECTION_TYPES] # 房间区域
        self.VERTICAL_TYPES = ['电梯', '扶梯', '楼梯'] # 垂直交通区域
        self.PEDESTRIAN_TYPES = ['走廊'] # 人行区域
        self.OUTSIDE_TYPES = ['室外'] # 室外区域
        self.GRID_SIZE = 40 # 网格大小
        self.OUTSIDE_ID = -1 # 室外区域ID
        self.OUTSIDE_TIMES = 2 # 室外网格间距倍数，同时影响室外时间
        self.BACKGROUND_ID = 0 # 背景区域ID
        self.PEDESTRIAN_ID = 255 # 人行区域ID
        self.PEDESTRAIN_TIME = 1 # 人行时间
        self.CONNECTION_TIME = 1 # 连接时间 - 门

COLOR_MAP = {
    (244, 67, 54): {'name':'药房', 'time': 1},
    (0, 150, 136): {'name':'挂号处', 'time': 1},
    (103, 58, 183): {'name':'急诊科', 'time': 1},
    (145, 102, 86): {'name': '中心供应室', 'time': 1},
    (33, 150, 243): {'name': '门诊治疗', 'time': 1},
    (3, 169, 244): {'name': '放射科', 'time': 1},
    (0, 188, 212): {'name': '儿科单元', 'time': 1},
    (207, 216, 220): {'name': '走廊', 'time': 1},
    (117, 117, 117): {'name': '楼梯', 'time': 1},
    (189, 189, 189): {'name': '电梯', 'time': 1},
    (158, 158, 158): {'name': '扶梯', 'time': 1},
    (76, 175, 80): {'name': '绿化', 'time': 1},
    (255, 235, 59): {'name': '墙', 'time': 1},
    (121, 85, 72): {'name': '门', 'time': 1},
    (156, 39, 176): {'name': '室外', 'time': 1},
    (139, 195, 74): {'name': '内镜中心', 'time': 1},
    (205, 220, 57): {'name': '检验中心', 'time': 1},
    (255, 193, 7): {'name': '消化内科', 'time': 1},
    (255, 152, 0): {'name': '内分泌科', 'time': 1},
    (254, 87, 34): {'name': '呼吸科', 'time': 1},
    (169, 238, 90): {'name': '心血管内科', 'time': 1},
    (88, 67, 60): {'name': '采血处', 'time': 1},
    (239, 199, 78): {'name': '眼科', 'time': 1},
    (253, 186, 87): {'name': '中医科', 'time': 1},
    (250, 133, 96): {'name': '口腔科', 'time': 1},
    (197, 254, 130): {'name': '耳鼻喉科', 'time': 1},
    (124, 165, 185): {'name': '超声科', 'time': 1},
    (173, 133, 11): {'name': '功能检查科', 'time': 1},
    (119, 90, 10): {'name': '病理科', 'time': 1},
    (250, 146, 138): {'name': '骨科', 'time': 1},
    (255, 128, 171): {'name': '肾内科', 'time': 1},
    (33, 250, 230): {'name': '康复医学科', 'time': 1},
    (141, 78, 255): {'name': '血液科', 'time': 1},
    (82, 108, 255): {'name': '皮肤科', 'time': 1},
    (226, 58, 255): {'name': '妇科', 'time': 1},
    (100, 139, 55): {'name': '产科', 'time': 1},
    (113, 134, 91): {'name': '手术室', 'time': 1},
    (175, 207, 142): {'name': '门诊手术室', 'time': 1},
    (179, 116, 190): {'name': '中庭', 'time': 1},
    (232, 137, 248): {'name': '风湿免疫科', 'time': 1},
    (63, 100, 23): {'name': '神经内科', 'time': 1},
    (182, 198, 9): {'name': '神经外科', 'time': 1},
    (240, 222, 165): {'name': '胸外科', 'time': 1},
    (221, 173, 229): {'name': '结直肠外科', 'time': 1},
    (166, 45, 36): {'name': '泌尿外科', 'time': 1},
    (187, 24, 80): {'name': '普外科', 'time': 1},
    (7, 91, 82): {'name': '特需门诊', 'time': 1},
    (150, 133, 179): {'name': '透析中心', 'time': 1},
    (115, 124, 177): {'name': '中西医结合科', 'time': 1},
    (195, 127, 122): {'name': '全科门诊', 'time': 1},
    (48, 122, 113): {'name': '生殖医学科', 'time': 1},
    (112, 40, 236): {'name': '肿瘤科', 'time': 1},
    (142, 157, 246): {'name': '胃肠外科', 'time': 1},
    (241, 190, 186): {'name': '计划生育科', 'time': 1},
    (186, 146, 160): {'name': '职业病科', 'time': 1},
    (71, 195, 180): {'name': '心理科', 'time': 1},
    (187, 152, 247): {'name': '美容科', 'time': 1},
    (254, 210, 145): {'name': '介入科', 'time': 1},
    (255, 255, 255): {'name': '空房间', 'time': 1}
}

# COLOR_MAP = {
#     (244, 67, 54): 'Pharmacy',
#     (0, 150, 136): 'Reception Desk',
#     (103, 58, 183): 'Emergency Department',
#     (145, 102, 86): 'Central Supply',
#     (33, 150, 243): 'Outpatient Treatment',
#     (3, 169, 244): 'Radiology',
#     (0, 188, 212): 'Pediatric Unit',
#     (207, 216, 220): 'Corridor',
#     (117, 117, 117): 'Stairs',
#     (189, 189, 189): 'Elevator',
#     (158, 158, 158): 'Escalator',
#     (76, 175, 80): 'Landscape Space',
#     (255, 235, 59): 'Wall',
#     (121, 85, 72): 'Door',
#     (156, 39, 176): 'Outside',
#     (139, 195, 74): 'Endoscopy Center',
#     (205, 220, 57): 'Testing Center',
#     (255, 193, 7): 'Gastroenterology',
#     (255, 152, 0): 'Endocrinology',
#     (254, 87, 34): 'Respiratory Medicine',
#     (169, 238, 90): 'Cardiovascular Medicine',
#     (88, 67, 60): 'Phlebotomy',
#     (239, 199, 78): 'Ophthalmology',
#     (253, 186, 87): 'Chinese Medicine',
#     (250, 133, 96): 'Oral Medicine',
#     (197, 254, 130): 'Otorhinolaryngology',
#     (124, 165, 185): 'Ultrasound Department',
#     (173, 133, 11): 'Functional Inspection Department',
#     (119, 90, 10): 'Pathology',
#     (250, 146, 138): 'Orthopedics',
#     (255, 128, 171): 'Nephrology',
#     (33, 250, 230): 'Physical Medicine and Rehabilitation',
#     (141, 78, 255): 'Hematology',
#     (82, 108, 255): 'Dermatology',
#     (226, 58, 255): 'Gynecology',
#     (100, 139, 55): 'Obstetrics',
#     (113, 134, 91): 'Operating Room',
#     (175, 207, 142): 'Outpatient Surgery',
#     (179, 116, 190): 'Courtyard',
#     (232, 137, 248): 'Rheumatology Department',
#     (63, 100, 23): 'Neurology',
#     (182, 198, 9): 'Neurosurgery',
#     (240, 222, 165): 'Thoracic Surgery',
#     (221, 173, 229): 'Colorectal Surgery',
#     (166, 45, 36): 'Urology',
#     (187, 24, 80): 'General Surgery',
#     (7, 91, 82): 'Special Outpatient Clinic',
#     (150, 133, 179): 'Dialysis Center',
#     (115, 124, 177): 'Integrated Traditional Chinese and Western Medicine Department',
#     (195, 127, 122): 'General Outpatient Clinic',
#     (48, 122, 113): 'Reproductive Medicine Department',
#     (112, 40, 236): 'Oncology',
#     (142, 157, 246): 'Gastrointestinal Surgery',
#     (241, 190, 186): 'Family Planning Department',
#     (186, 146, 160): 'Occupational Disease Department',
#     (71, 195, 180): 'Psychology Department',
#     (187, 152, 247): 'Beauty Department',
#     (254, 210, 145): 'Interventional Department',
#     (255, 255, 255): 'Empty Room'
# }