import src
from src import Node
import torch
import os
import networkx as nx
import json

if __name__ == "__main__":
    # torch.manual_seed(CONFIG.SEED)

    # 训练模型
    # dataset = src.Mask2FormerDataset()
    # train_dataset, val_dataset = random_split(dataset=dataset, lengths=[CONFIG.TRAIN_SIZE, CONFIG.VAL_SIZE])
    # src.Train(train_dataset,val_dataset).train()

    # 可视化数据集
    # src.utils.visualize_dataset(dataset)

    # 预测
    # predict = src.Mask2FormerPredict(model_path='./models/best_model.pth')
    # predict.run(image_path='./data/image/01_1f.png', save_path='./result/predict.jpg')

    # predict = Predict(model_path='./models/best_model.pth', image_path='./data/val_image/1f.jpg')
    # predict.run()

    # 节点图
    image_paths = [os.path.join('./data/label', image) for image in os.listdir('./data/label')]
    super_network = src.SuperNetwork(tolerance=30) # tolerance: 表示两层节点之间的识别距离
    super_network.run(image_paths=image_paths, zlevels = [-10, 0, 10, 20, 30])
    super_network.plot_plotly(save=True)
    src.calculate_room_travel_times(super_network.super_graph)
    pass