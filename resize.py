import os
import glob
from PIL import Image

# Resize all images in the folder
"""def resize_images(
    target_size=(256, 256), input_folder=r"input_folder", output_folder=r"output_folder"
):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in glob.glob(os.path.join(input_folder, "*.jpg")):
        try:
            img = Image.open(filename)
            img.thumbnail(target_size, Image.LANCZOS)

            output_path = os.path.join(output_folder, os.path.basename(filename))
            img.save(output_path)
            print(f"Изменен размер: {filename} -> {output_path}")

        except Exception as e:
            print(f"Ошибка при обработке {filename}: {e}")

    print("Обработка завершена!")"""


# Resize only one image
def resize_image(filename, input_folder, output_folder, target_size=(256, 256)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    try:
        input_path = os.path.join(input_folder, filename)

        img = Image.open(input_path)
        img.thumbnail(target_size, Image.LANCZOS)

        output_path = os.path.join(output_folder, os.path.basename(filename))
        img.save(output_path)

        print(f"Resized: {filename} -> {output_path}")

    except Exception as e:
        print(f"Error while resizing {filename}: {e}")

    print("Resizing completed")
