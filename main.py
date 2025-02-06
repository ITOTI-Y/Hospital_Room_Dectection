import src
import torch

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
    node_graph = src.Network(image_path='./data/label/1F-meng.png')
    node_graph.plot()
