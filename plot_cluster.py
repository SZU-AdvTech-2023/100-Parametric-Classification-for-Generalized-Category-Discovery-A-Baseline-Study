import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'SimHei'
S_data = pd.read_csv('feature_SimGCD.csv')
S_data.set_index('Unnamed: 0', inplace=True)
print(S_data)
Label = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车']
# 设置散点形状
maker = 'o'
# 设置散点颜色
colors = ['#e38c7a', '#656667', '#99a4bc', 'cyan', 'blue', 'lime', 'r', 'violet', 'm', 'peru', 'olivedrab',
          'hotpink']

fig = plt.figure(figsize=(10, 10))
for index in range(10):  # 假设总共有三个类别，类别的表示为0,1,2
    X = S_data.loc[S_data['label'] == index]['x']
    Y = S_data.loc[S_data['label'] == index]['y']
    plt.scatter(X, Y, cmap='brg', s=100, marker=maker, c=colors[index], edgecolors=colors[index], alpha=0.65,
                label=Label[index])

    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.legend()

plt.title('SimGCD', fontsize=26, fontweight='normal', pad=20)
plt.show()
