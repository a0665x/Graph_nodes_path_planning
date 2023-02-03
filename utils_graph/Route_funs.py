import cv2
import pandas as pd
import json
from pylab import show
import networkx as nx
import matplotlib.pyplot as plt


def load_and_check_nodes_map(background_map_path='./imgs/map_1.png', nodes_txt_path='./nodes/map_1./points.txt',
                             drew_nodes=True):
    image = cv2.imread(background_map_path)
    nodes_pos = []
    f = open(nodes_txt_path)
    for line in f.readlines():
        line = line.strip('\n')
        line_ = line[1:-1].split(',')
        nodes_pos.append((int(line_[0]), int(line_[1])))
    f.close()
    if drew_nodes == True:
        for idx, p in enumerate(nodes_pos):
            image = cv2.circle(image, (int(p[0]), int(p[1])), radius=10, color=(230, 170, 110), thickness=-1)
            cv2.putText(image, f'{idx}', (int(p[0]) - 6, int(p[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (140, 0, 255), 1,
                        cv2.LINE_AA)
    return image, nodes_pos


class Route:
    def __init__(self, g, time_df, weight_df, threshold, order_list, pos_list):  # = orders  # = position
        self.g = g
        self.threshold = threshold
        self.orders = order_list
        self.position = pos_list
        self.time_df = time_df
        self.weight_df = weight_df
        self.g_values = self.df2graph(self.g, self.time_df, self.weight_df)
        self.draw_graph = self.draw_graph(self.g, self.g_values, self.threshold, self.orders, self.position)

    def df2graph(self, g, time_df, weight_df):  # 將df 有值得部分轉乘graph 格式
        global g_values
        g_values = []
        for col in self.time_df.columns:
            for idx in self.time_df.index:
                if self.time_df.loc[[idx], [col]].isnull().values[0][0] == False:  # 有值存在
                    if self.weight_df.loc[[idx], [col]].isnull().values[0][0] == False:
                        g_values.append((str(col), str(idx), self.time_df.loc[[idx], [col]].values[0][0] /
                                         self.weight_df.loc[[idx], [col]].values[0][0]))
        return g_values

    def draw_graph(self, g, g_values, threshold, orders, position):
        global elarge, esmall, pos1, g1
        for (s, e, w) in g_values:  # 藉由g_values :(起點,終點,權重) 格式,添加到 g nodes中
            g.add_edge(s, e, weight=w)
        elarge, esmall = [], []
        for (u, v, d) in g.edges(data=True):
            try:
                if d['weight'] > threshold:
                    elarge.append((u, v, d))
                elif d['weight'] <= threshold:  # 該時間/權重 <= 閥值: 虛線欄線
                    esmall.append((u, v, d))
            except:
                continue
        #         pos=nx.spring_layout(g) #隨機節點位置
        g1 = g
        for idx, pos_v in enumerate(position):  # 添加固定position 位置座標
            #             print(orders[idx],type(orders[idx]),pos_v)
            g.nodes[orders[idx]]['pos'] = pos_v
        pos = nx.get_node_attributes(g, 'pos')
        pos1 = pos
        # 依照POS位置資訊劃出節點位置
        nx.draw_networkx_nodes(g, pos, node_size=80) # 以g 畫節點大小
        # 劃出權重大小的線
        nx.draw_networkx_edges(g, pos, edgelist=elarge, width=2)
        nx.draw_networkx_edges(g, pos, edgelist=esmall, width=2, alpha=1, edge_color='blue', style='-')  # dashed
        #         nx.draw_networkx_edge_labels(g, pos,'k', edge_weight)
        # labels標籤定義
        nx.draw_networkx_labels(g, pos, font_size=10, font_family='sans-serif') #以g畫節點內的文字
        return g

    def best_route(self, g, start_node, end_node, method='dijkstra', plot_best_route_=True):
        if method == 'dijkstra':
            best_route_ = nx.dijkstra_path(g, start_node, end_node, weight='weight')
            print('dijkstra length:', nx.dijkstra_path_length(g, best_route_[0], best_route_[-1]))
        elif method == 'bellman_ford_path':
            best_route_ = nx.bellman_ford_path(g, start_node, end_node, weight='weight')
            print('bellman_ford length:', nx.bellman_ford_path_length(g, best_route_[0], best_route_[-1]))
        elif method == 'astar':
            best_route_ = nx.astar_path(g, start_node, end_node)
            print('astar length:', nx.astar_path_length(g, best_route_[0], best_route_[-1]))
        bestlist = []
        for idx, v in enumerate(best_route_):
            if idx == len(best_route_) - 1:
                break
            for vv in self.g_values:
                if best_route_[idx] == vv[0] and best_route_[idx + 1] == vv[1]:
                    bestlist.append((best_route_[idx], best_route_[idx + 1], {'weight': vv[-1]}))
        if plot_best_route_ == True:
            pos = nx.get_node_attributes(g, 'pos')  # 得到g中所有點的座標   {'A': (1, 1), 'D': (30, 25), ....}
            nx.draw_networkx_edges(g, pos, edgelist=bestlist, width=4, alpha=1, edge_color='green', style='-')
        return best_route_, bestlist

if __name__ == '__main__':
    image, nodes_pos = load_and_check_nodes_map(background_map_path='../imgs/map_1.png',
                                                nodes_txt_path='../points/map1_points.txt',
                                                drew_nodes=True)
    orders = [str(i) for i in range(len(nodes_pos))]
    position = nodes_pos
    time_df = pd.read_csv('../df_tables/map1_time.csv', index_col=[0])
    weight_df = pd.read_csv('../df_tables/map1_weight.csv', index_col=[0])

    g = nx.Graph()  # 建立空白圖功能

    plt.figure(figsize=(20, 10))
    plt.title('<Undirected graph (無向圖)>', fontproperties="SimSun", fontsize=15);
    plt.axis('on');
    plt.xticks([]);
    plt.yticks([])

    R = Route(g, time_df, weight_df, threshold=15, order_list=orders, pos_list=position)
    best_route_, bestlist = R.best_route(g, '0', '15', method='dijkstra',
                                         plot_best_route_=True)  # dijkstra  #bellman_ford_path
    print('BEST_ROUTE:', best_route_)
    print(bestlist)
    imgplot = plt.imshow(image)
    final_img = imgplot.get_array()  # return image
    print(final_img.shape)
    plt.show()