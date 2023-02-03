# Graph_nodes_path_planning
## step1:
#### pip install -r requirements.txt
## step2:
#### streamlit run steamlit_route_run.py

Initially, you can download a 2D map and adjust the parameters of the image processing using the slide bar, so that the more Hough lines there are, the better. Then, by clicking "adjust image", you can adjust the rotation angle of the image to make it look more upright.
![image](https://user-images.githubusercontent.com/44718189/216522998-6625a625-567f-41a6-978c-76aff63d17a9.png)


By clicking to load the map coordinates in the "points" folder, you can visualize the nodes (points) coordinates on the map, and then by loading the "time table" and "weight table", you can select the starting node and the ending node.
![image](https://user-images.githubusercontent.com/44718189/216522895-38825a08-c91f-4331-b641-fffafc51ca4c.png)

Press the "Run Best Route" button to find the best path between the nodes (shown in green).
![image](https://user-images.githubusercontent.com/44718189/216523531-a6fa516e-67aa-44ef-879f-2a65b2373046.png)
