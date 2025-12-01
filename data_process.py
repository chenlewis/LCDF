import os
import shutil

# def copy_images_every_nth(input_folder, output_folder, n=3):
#     # 确保目标文件夹存在，如果不存在则创建
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#
#     # 获取输入文件夹中的所有图片文件
#     all_files = os.listdir(input_folder)
#     images = [f for f in all_files if f.lower().endswith(('.tif', '.png', '.jpg', '.jpeg', '.bmp', '.gif'))]  # 根据需要调整图片格式
#
#     # 遍历所有图片，按间隔复制
#     for i in range(0, len(images), n):
#         # 获取当前图片路径
#         img_path = os.path.join(input_folder, images[i])
#
#         # 设置目标路径
#         target_path = os.path.join(output_folder, images[i])
#
#         # 复制文件
#         shutil.copy(img_path, target_path)
#         print(f"复制文件: {images[i]} 到 {target_path}")
#
#
# # 示例调用
# input_folder = 'F:/Datasets/TDSC_Dataset/DM_TMTP_dot/images/2'  # 输入图片文件夹路径
# output_folder = 'E:/ASDPAD/SurrogateModel/dataroot1/trainB'  # 输出图片文件夹路径
# copy_images_every_nth(input_folder, output_folder, n=3)

import os
import shutil

# Function to move files based on the condition
def move_fake_B_images(source_dir, destination_dir):
    # Ensure destination directory exists
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Traverse the source directory
    for filename in os.listdir(source_dir):
        # Check if 'fake_B' is in the filename
        if 'fake_B' in filename:
            file_path = os.path.join(source_dir, filename)
            # Check if it's a file
            if os.path.isfile(file_path):
                # Move the file to the destination directory
                shutil.move(file_path, os.path.join(destination_dir, filename))
                print(f'Moved: {filename}')

# Example usage
source_directory = '/home/lyj/ASDPAD/SurrogateModel/results/stageone_train_20250219/test_30/images'  # Replace with the path of the source folder
destination_directory = '/home/data1/lyj/ROD/image_with_moire_Cyclegan/images/0/'  # Replace with the path of the destination folder

move_fake_B_images(source_directory, destination_directory)
