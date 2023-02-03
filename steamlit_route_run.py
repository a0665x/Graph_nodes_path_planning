import cv2
import os
import streamlit as st
import numpy as np
from PIL import Image
import math
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import base64
from scipy.ndimage.interpolation import rotate
from utils_graph.Route_funs import load_and_check_nodes_map , Route

class DrawLineWidget(object):
    def __init__(self, points, image_path):
        self.original_image = cv2.imread(image_path)
        for idx, p in enumerate(points):
            cv2.circle(self.original_image, p, 18, (255, 0, 0), 3)
            cv2.putText(self.original_image, f'{idx}', p, cv2.FONT_HERSHEY_SIMPLEX, 3, (140, 0, 255), 5, cv2.LINE_AA)


        cv2.namedWindow('image', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.setMouseCallback('image', self.extract_coordinates)
        self.clone = self.original_image.copy()
        # List to store start/end points
        self.image_coordinates = []

    def extract_coordinates(self, event, x, y, flags, parameters):
        # Record starting (x,y) coordinates on left mouse button click
        if event == cv2.EVENT_LBUTTONDOWN:
            self.image_coordinates = [(x, y)]

        # Record ending (x,y) coordintes on left mouse bottom release
        elif event == cv2.EVENT_LBUTTONUP:
            self.image_coordinates.append((x, y))
            print('Starting: {}, Ending: {}'.format(self.image_coordinates[0], self.image_coordinates[1]))
            # Draw line
            #             cv2.line(self.clone, self.image_coordinates[0], self.image_coordinates[1], (255,100,12), 2)
            cv2.arrowedLine(self.clone, self.image_coordinates[0], self.image_coordinates[1], (255, 100, 12), 2,
                            line_type=cv2.LINE_8, shift=0, tipLength=0.06)
            cv2.arrowedLine(self.clone, self.image_coordinates[1], self.image_coordinates[0], (255, 100, 12), 2,
                            line_type=cv2.LINE_8, shift=0, tipLength=0.06)

            cv2.imshow("image", self.clone)
        # ===========================================================
        #         df
        # ===========================================================
        # Clear drawing boxes on right mouse button click
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.clone = self.original_image.copy()

    def show_image(self):
        return self.clone




def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
        unsafe_allow_html=True)

def brighten_image(image, amount):
    img_bright = cv2.convertScaleAbs(image, beta=amount)
    return img_bright


def blur_image(image, amount):
    blur_img = cv2.GaussianBlur(image, (11, 11), amount)
    return blur_img

def canny_image(image,canny_amount):
    canny_img = cv2.Canny(image , canny_amount[0],canny_amount[1])
    return canny_img

def hongh_image(image , hongh_amount):
    lines = cv2.HoughLinesP(image, 1, np.pi / 180, 200, minLineLength = hongh_amount[0], maxLineGap = hongh_amount[1])
    ang_list = []
    for line in lines:
        x1 = line[0][0]
        y1 = line[0][1]
        x2 = line[0][2]
        y2 = line[0][3]
        cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), 10)
        v = (x2 - x1, y2 - y1)
        angle = math.atan2(v[1], v[0])
        angle = round(angle * 180 / math.pi, 2)
        ang_list.append(angle)
    try:
        # ang_list = [a if abs(a) < 45 else 90 - abs(a) for a in ang_list]
        ang_list = [a  for a in ang_list if abs(a) < 45]
        bins = np.arange(-45, 45, 5)
        hist, edges = np.histogram(ang_list, bins)
        must_range_bin = edges[np.where(hist == np.max(hist))[0][0]:np.where(hist == np.max(hist))[0][0]+2]
        range_ang = [a for a in ang_list if a > must_range_bin[0] and a <= must_range_bin[1]]
        do_ang = round(np.array(range_ang).mean(), 2)
    except:
        do_ang = ' No detected '
        pass
    return image , do_ang

def enhance_details(img):
    hdr = cv2.detailEnhance(img, sigma_s=12, sigma_r=0.15)
    return hdr


def main_loop():
    add_bg_from_local('background_img_0.jpg')
    st.title("Costmap Nodes Routes")
    st.subheader("This app will help you find the best route by using nodes on costmap.")
    st.text("select your img and points txt file")
    # 左邊條碼bar 使用參數
    #========================圖像處理選單欄======================================
    options = st.sidebar.multiselect( 'What are your favorite image procession',
        ['Canny', 'Blur', 'Brighten','HonghLine'],[])
    # =======================================
    # 上傳數據
    image_file = st.file_uploader("Upload Your Image", type=['jpg', 'png', 'jpeg'])
    if not image_file:
        return None

    original_image = Image.open(image_file).convert("L")
    processed_image = np.array(original_image)

    for p in options:
        if p == 'Blur':
            blur_rate = st.sidebar.slider("Blurring", min_value=0.5, max_value=3.5)
            processed_image = blur_image(processed_image, blur_rate)
            # print('1',options)
        elif p == 'Brighten':
            brightness_amount = st.sidebar.slider("Brightness", min_value=-50, max_value=50, value=0) # 使用一個bar 拖拉
            processed_image = brighten_image(processed_image, brightness_amount)
            # print('2',options)
        elif p=='Canny':
            canny_amount = st.sidebar.slider('Canny',50.0, 250.0, (75.0, 175.0)) # 用兩個bar 拖拉
            processed_image = canny_image(processed_image, canny_amount)
            # print('3',options)
        elif p=='HonghLine':
            hongh_amount = st.sidebar.slider('HonghLine',0.0,150.0,(10.0,90.0))
            processed_image, do_ang = hongh_image(processed_image, hongh_amount)
            st.sidebar.write(f'optimal rotation angle:{do_ang} degree')
            # print('4',options)

            adject_img = st.sidebar.checkbox('adject image')
            if adject_img:
                processed_image = rotate(processed_image, angle=float(do_ang))


    apply_enhancement_filter = st.sidebar.checkbox('Enhance Details')

    if apply_enhancement_filter:
        processed_image = enhance_details(processed_image)

    st.text("Original Image vs Processed Image")

    st.image([processed_image])
    txt_file = st.file_uploader("Upload Your points.txt", type=['txt', 'csv'])
    if not txt_file:
        return None
    #==========================================================
    # print('txt_file.name:',txt_file.name)
    image_path = os.getcwd()+'/imgs/'+image_file.name
    nodes_path = os.getcwd()+'/points/'+txt_file.name
    print('save_nodes_path:',nodes_path)
    with open(nodes_path, 'r') as f:
        lines = f.readlines()
    points = [tuple(map(int, i.strip('\n')[1:-1].split(','))) for i in lines]
    draw_line_widget = DrawLineWidget(points, image_path)
    # st.image([processed_image,draw_line_widget.show_image()])
    st.image([draw_line_widget.show_image()])
    #===========================================================
    time_table_path = st.file_uploader("Upload Your time_table", type=['csv'])
    weight_table_path = st.file_uploader("Upload Your weight_table", type=['csv'])
    #===========================================================
    # 顯示 選擇節點起點/終點
    st.subheader('generating the best route')
    start_node = st.radio(label='Start node:', options=[str(i) for i in range(len(points))])
    end_node = st.radio(label='End node:', options=[str(i) for i in range(len(points))])
    aigo = st.radio(label='algorithm select:', options=[ 'dijkstra' ,'bellman_ford_path','astar'])
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
    #==========================================================
    btnResult = st.button('Run Best Route')

    if btnResult:
    #==========================================================
        # 執行路徑規劃算法
        image, nodes_pos = load_and_check_nodes_map(background_map_path=image_path,
                                                    nodes_txt_path=nodes_path,
                                                    drew_nodes=True)
        orders = [str(i) for i in range(len(nodes_pos))]
        position = nodes_pos

        time_df = pd.read_csv(time_table_path, index_col=[0])
        weight_df = pd.read_csv(weight_table_path, index_col=[0])

        g = nx.Graph()  # 建立空白圖功能

        plt.figure()
        plt.title('<Undirected graph (無向圖)>', fontproperties="SimSun", fontsize=15);
        plt.axis('on');
        plt.xticks([]);
        plt.yticks([])

        R = Route(g, time_df, weight_df, threshold=15, order_list=orders, pos_list=position)
        best_route_, bestlist = R.best_route(g, start_node, end_node, method=aigo,
                                             plot_best_route_=True)  # dijkstra  #bellman_ford_path , #astar
        plt.imshow(image)
        #==========
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        #==========
        plt.savefig('./map1_best_route.jpg',dpi=300)
        graph_img = Image.open('./map1_best_route.jpg').convert("RGB")
        st.subheader(f'BEST Route: {best_route_}')
        st.image([graph_img],width = 100 , use_column_width='auto') #'auto' or 'always' or 'never' or bool
        # plt.show()

    hide_streamlit_style = """
                <style>
                footer {visibility: hidden;}
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)


if __name__ == '__main__':
    main_loop()