import src
import os
import src.network

if __name__ == "__main__":
    # 单层节点图
    # network = src.Network()
    # network.run(image_path = './data/label/1F-meng.png', zlevel=0)
    # network.plot_plotly(save=True)

    # 多层节点图
    super_network = src.SuperNetwork(tolerance=30) # tolerance: 表示两层节点之间的识别距离
    image_paths = [os.path.join('./data/label', image) for image in os.listdir('./data/label')]
    super_network.run(image_paths=image_paths, zlevels = [-10, 0, 10, 20, 30])
    super_network.plot_plotly(save=True)

    # 计算节点之间的距离，输出csv文件
    src.calculate_room_travel_times(super_network.super_graph)