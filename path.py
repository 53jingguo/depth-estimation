import os
dir = "D:\\Data\\3D60"
list_file = "D:\\project\\UniFuse-Unidirectional-Fusion-main\\UniFuse-Unidirectional-Fusion-main\\UniFuse\\datasets\\3D60_test.txt"
newfile= "D:\\project\\UniFuse-Unidirectional-Fusion-main\\UniFuse-Unidirectional-Fusion-main\\UniFuse\\train.txt"
rgb_depth_list = []
with open(list_file) as f:
    lines = f.readlines()
    for line in lines:
        file = line.strip().split(" ")[0]
        file_name = os.path.join(dir, file)
        splits = file_name.split('.')
        rot_ang = splits[0].split('_')[-1]
        file_name = splits[0][:-len(rot_ang)] + "0." + splits[-2] + "." + splits[-1]
        if os.path.isfile(file_name):
            with open("D:\\project\\UniFuse-Unidirectional-Fusion-main\\UniFuse-Unidirectional-Fusion-main\\UniFuse\\3D60_test.txt", "a", encoding='utf-8') as file:
                file.writelines(line)
                file.close()


