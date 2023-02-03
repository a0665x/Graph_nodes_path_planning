import pandas as pd
from PIL import Image
import streamlit as st
# pip install streamlit-drawable-canvas
from streamlit_drawable_canvas import st_canvas
import math
import cv2
import os
import numpy as np
# cd frontend
# npm run start


def node_id(x, y, Nodes):
    D = 1e10
    for idx, p in enumerate(Nodes):
        d = math.sqrt( ((p[0]-x)**2)+((p[1]-y)**2) )
        if d < D:
            D, NODE_ID = d, idx
    return NODE_ID

# Specify canvas parameters in application
drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("point", "freedraw", "line", "rect", "circle", "transform")
)

stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
if drawing_mode == 'point':
    point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
stroke_color = st.sidebar.color_picker("Stroke color hex: ")
bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])

realtime_update = st.sidebar.checkbox("Update in realtime", True)
if bg_image != None:
    h_ratio , w_ratio = Image.open(bg_image).size[1]/400 , Image.open(bg_image).size[0]/600
# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    background_image=Image.open(bg_image) if bg_image else None,
    update_streamlit=realtime_update,
    height=400,
    width=400,
    drawing_mode=drawing_mode,
    point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
    key="canvas",
)

# Do something interesting with the image data and paths
# if canvas_result.image_data is not None:
#     st.image(canvas_result.image_data)

if canvas_result.json_data is not None:
    objects = pd.json_normalize(canvas_result.json_data["objects"])  # need to convert obj to str because PyArrow
    for col in objects.select_dtypes(include=['object']).columns:
        objects[col] = objects[col].astype("str")
    # st.dataframe(objects)


    # btnResult = st.button('export nodes positions')
    form = st.form(key='my_form')
    NODE_txt_path = form.text_input(label='Enter NODES txt path and submit!')
    submit_button_nodes = form.form_submit_button(label='Submit')
    Nodes_prev = ''
    if submit_button_nodes == True and len(objects.select_dtypes(include=['object']).columns) != 0:
        # df = objects[['type', 'left', 'top' , 'angle' ,'radius']].copy()
        df_c = objects.loc[objects['type']=='circle',['type', 'left', 'top' , 'angle' ,'radius']]
        df_c['cos'] = df_c['angle'].apply(lambda x: math.cos(x * math.pi / 180))
        df_c['sin'] = df_c['angle'].apply(lambda x: math.sin(x * math.pi / 180))
        df_c['rcos'] = df_c['radius'].values * df_c['cos'].values
        df_c['rsin'] = df_c['radius'].values * df_c['sin'].values
        df_c['real_cx'] = (df_c['left'] + df_c['rcos'])*w_ratio
        df_c['real_cy'] = (df_c['top'] +  df_c['rsin'])*h_ratio
        df_c['real_cx'] = df_c['real_cx'].apply(lambda x : int(x) if type(x)==float else x)
        df_c['real_cy'] =  df_c['real_cy'].apply(lambda x : int(x) if type(x)==float else x)
        # print(df_c[['type', 'left', 'top' , 'angle' ,'radius' , 'cos' , 'sin' ,'rcos' , 'rsin' ,'real_cx' , 'real_cy']])
        center_xy_real = list(zip(df_c['real_cx'],df_c['real_cy']))
        # st.dataframe(df_c.style.highlight_max(['real_cx' , 'real_cy'],axis=0))
        Nodes =  center_xy_real

        with open(NODE_txt_path,'w') as f:
            for p in center_xy_real:
                f.write(str(p)+'\n')
    #==============================================================================



    # btnResult = st.button('show weight lines info')
    # if btnResult == True and len(objects.select_dtypes(include=['object']).columns) != 0:
        df_l = objects.loc[objects['type'] == 'line', ['type', 'left', 'top' , 'x1',  'y1' , 'x2', 'y2']]
        df_l['x1'] = df_l['left'] + df_l['x1']
        df_l['x2'] = df_l['left'] + df_l['x2']
        df_l['y1'] = df_l['top'] + df_l['y1']
        df_l['y2'] = df_l['top'] + df_l['y2']

        df_l['x1'] = df_l['x1'] * w_ratio
        df_l['x2'] = df_l['x2'] * w_ratio
        df_l['y1'] = df_l['y1'] * h_ratio
        df_l['y2'] = df_l['y2'] * h_ratio
        for c in ['x1',  'y1' , 'x2', 'y2']:
            df_l[c] = df_l[c].apply(lambda x:int(x))
        from_xy = list(zip(df_l['x1'], df_l['y1']))
        from_node = []
        for ls in from_xy:
            idx = node_id(ls[0], ls[1], Nodes)
            from_node.append(idx)
        df_l['from_node'] = from_node

        to_xy = list(zip(df_l['x2'], df_l['y2']))
        to_node = []
        for le in to_xy:
            idx = node_id(le[0], le[1], Nodes)
            to_node.append(idx)
        df_l['to_node'] = to_node

        # st.table(df_l)
        # st.table(df_l.style.highlight_max(['y2'],axis=0))

    #==============================================================================

    # simple_btnResult = st.button('export nodes and w/t lines')
    # if simple_btnResult:
    simple_np = canvas_result.image_data
    try:
        with open(NODE_txt_path, 'r') as f:
            lines = f.readlines()

        st.subheader('select your pair nodes and update time_spend and weight')
        from_node = st.radio(label='from node:', options=[str(i) for i in range(len(lines))])
        to_node = st.radio(label='to node:', options=[str(i) for i in range(len(lines))])
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

        form = st.form(key="my-form")
        value_t = st.text_input("keyin time")
        value_w = st.text_input("keyin weight")


    #===============================================================================================
        df_time_save_path = './df_tables/' + bg_image.name.split('.')[-2] + '_time.csv'
        df_weight_save_path = './df_tables/' + bg_image.name.split('.')[-2] + '_weight.csv'






        df_upload_btnResult = st.button('upload time/weight infomation')
        if df_upload_btnResult == True:
            if  os.path.exists(df_time_save_path) and os.path.exists(df_weight_save_path): # 如果兩個 tabel (time / weight)
                df_t = pd.read_csv(df_time_save_path, index_col=0)
                df_w = pd.read_csv(df_weight_save_path, index_col=0)
                try:
                    df_t.iloc[int(to_node), int(from_node)] = float(value_t)
                    df_w.iloc[int(to_node), int(from_node)] = float(value_w)
                    df_t.to_csv(df_time_save_path)
                    df_w.to_csv(df_weight_save_path)
                except: # keyin 非數值 or 空白
                    df_t.iloc[int(to_node), int(from_node)] = np.NaN
                    df_w.iloc[int(to_node), int(from_node)] = np.NaN
                    df_t.to_csv(df_time_save_path)
                    df_w.to_csv(df_weight_save_path)
                    pass
            else:
                df_t = pd.DataFrame(columns=[str(i) for i in range(len(lines))],index = [str(i) for i in range(len(lines))])
                df_t.to_csv( df_time_save_path , encoding = 'utf-8')
                df_w = pd.DataFrame(columns=[str(i) for i in range(len(lines))], index=[str(i) for i in range(len(lines))])
                df_w.to_csv(df_weight_save_path, encoding='utf-8')

            #===============================================================================================
            l_s = tuple([int(i) for i in lines[int(from_node)].replace('\n','').strip(')').strip('(').split(',')])
            l_e = tuple([int(i) for i in lines[int(to_node)].replace('\n', '').strip(')').strip('(').split(',')])
            cv2.line(simple_np, (int(l_s[0]/w_ratio),int(l_s[1]/h_ratio)) , (int(l_e[0]/w_ratio),int(l_e[1]/h_ratio)) , (100, 100, 100,100), 2)
            nodes_info = list(zip(np.where(df_t.notna())[1], np.where(df_t.notna())[0]))  # [(from,to),(from,to)....]
            for ns in nodes_info:
                ns_f = tuple([int(i) for i in lines[int(ns[0])].replace('\n','').strip(')').strip('(').split(',')])
                ns_e = tuple([int(i) for i in lines[int(ns[1])].replace('\n', '').strip(')').strip('(').split(',')])

                cv2.line(simple_np, (int(ns_f[0] / w_ratio), int(ns_f[1] / h_ratio)),
                         (int(ns_e[0] / w_ratio), int(ns_e[1] / h_ratio)), (100, 100, 100, 100), 2)



            for idx , str_p in enumerate(lines):
                point_s = str_p.replace('\n','')
                each_point = [int(i) for i in point_s.strip(')').strip('(').split(',')]
                cv2.putText(simple_np, f'{idx}', (int(each_point[0]/w_ratio),int(each_point[1]/h_ratio)), cv2.FONT_HERSHEY_DUPLEX, 0.5, (100, 100, 100,200), 1)

            st.image(simple_np)
        else:
            for idx , str_p in enumerate(lines):
                point_s = str_p.replace('\n','')
                each_point = [int(i) for i in point_s.strip(')').strip('(').split(',')]
                cv2.putText(simple_np, f'{idx}', (int(each_point[0]/w_ratio),int(each_point[1]/h_ratio)), cv2.FONT_HERSHEY_DUPLEX, 0.5, (100, 100, 100,200), 1)
            st.image(simple_np)
    except:
        # if bg_image is not None:
        #     for idx , str_p in enumerate(lines):
        #         point_s = str_p.replace('\n','')
        #         each_point = [int(i) for i in point_s.strip(')').strip('(').split(',')]
        #         cv2.putText(simple_np, f'{idx}', (int(each_point[0]/w_ratio),int(each_point[1]/h_ratio)), cv2.FONT_HERSHEY_DUPLEX, 0.5, (100, 100, 100,200), 1)
        #     st.image(simple_np)
        pass








