import pathlib
import torch


class Train_Config:
    def __init__(self):
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.IMAGE_DIR = pathlib.Path('./data/image')
        self.LABEL_DIR = pathlib.Path('./data/label')
        self.IMAGE_SIZE = (256,256)
        self.BATCH_SIZE = 2
        self.EPOCHS = 10
        self.SHUFFLE = True

COLOR_MAP = {
    (55, 67, 245): 'Pharmacy', 
    (136, 151, 0): 'Reception desk', 
    (183, 58, 103): 'Emergency Department', 
    (243, 150, 33): 'Outpatient Treatment', 
    (245, 169, 3): 'Radiology', 
    (213, 188, 0): 'Pediatric Unit', 
    (221, 216, 207): 'Corridor', 
    (117, 117, 117): 'Stairs', 
    (189, 189, 189): 'Elevator', 
    (158, 158, 158): 'Escalator', 
    (80, 176, 76): 'Landscape Space', 
    (60, 235, 255): 'Wall', 
    (71, 85, 121): 'Door', 
    (177, 40, 156): 'Outside', 
    (75, 195, 140): 'Endoscopy Center', 
    (57, 220, 205): 'Testing Center', 
    (7, 193, 254): 'Gastroenterology', 
    (1, 152, 255): 'Endocrinology', 
    (33, 87, 254): 'Respiratory Medicine', 
    (91, 238, 170): 'Cardiovascular Medicine', 
    (78, 199, 239): 'Ophthalmology', 
    (87, 186, 254): 'Chinese medicine', 
    (97, 133, 251): 'Oral Medicine', 
    (131, 254, 198): 'Otorhinolaryngology', 
    (185, 165, 124): 'Ultrasound Department', 
    (10, 133, 173): 'Functional Inspection Department', 
    (149, 212, 255): 'Rehabilitation Medicine Department', 
    (139, 145, 250): 'Orthopedics', 
    (171, 128, 255): 'Nephrology', 
    (231, 250, 33): 'Physical Medicine and Rehabilitation', 
    (255, 78, 141): 'Hematology', 
    (255, 108, 82): 'Dermatology', 
    (255, 58, 226): 'Gynecology', 
    (56, 139, 100): 'Obstetrics', 
    (189, 115, 179): 'Courtyard', 
    (249, 136, 233): 'Rheumatology Department', 
    (23, 100, 63): 'Neurology', 
    (9, 199, 182): 'Neurosurgery', 
    (166, 222, 241): 'Thoracic Surgery', 
    (229, 173, 222): 'Colorectal Surgery', 
    (36, 45, 165): 'Urology', 
    (81, 24, 187): 'General Surgery', 
    (83, 91, 7): 'Special Outpatient Clinic', 
    (178, 133, 150): 'Dialysis Center', 
    (177, 124, 114): 'Integrated Traditional Chinese and Western Medicine Department', 
    (122, 127, 195): 'General Outpatient Clinic', 
    (113, 122, 48): 'Reproductive Medicine Department', 
    (236, 40, 112): 'Oncology', 
    (246, 158, 142): 'Gastrointestinal Surgery', 
    (185, 190, 241): 'Family Planning Department', 
    (160, 146, 187): 'Occupational Disease Department', 
    (179, 195, 70): 'Psychology Department', 
    (247, 151, 187): 'Beauty Department'}

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

