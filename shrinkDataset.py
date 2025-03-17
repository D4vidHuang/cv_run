import os
import shutil


def extract_top5_images(source_root, dest_root):
    """
    Extracts the first 5 images from each subfolder in the given directory structure
    and saves them to another directory while maintaining the folder structure.

    :param source_root: Path to the source directory (Raindrop)
    :param dest_root: Path to the destination directory (shrinkRaindrop)
    """
    for pic_folder in os.listdir(source_root):
        pic_path = os.path.join(source_root, pic_folder)
        if not os.path.isdir(pic_path):
            continue

        for class_folder in os.listdir(pic_path):
            class_path = os.path.join(pic_path, class_folder)
            if not os.path.isdir(class_path):
                continue

            dest_class_path = os.path.join(dest_root, pic_folder, class_folder)
            os.makedirs(dest_class_path, exist_ok=True)

            image_files = sorted(os.listdir(class_path))[:5]  # Get first 5 images

            for image_file in image_files:
                source_image_path = os.path.join(class_path, image_file)
                dest_image_path = os.path.join(dest_class_path, image_file)
                shutil.copy2(source_image_path, dest_image_path)  # Copy image

extract_top5_images("D:/DSAIT/DayRainDrop_Train", "shrinkRaindrop")