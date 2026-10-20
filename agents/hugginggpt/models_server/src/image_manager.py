import os
import requests
from PIL import Image
import uuid


class ImageManager:
    def __init__(self, save_dir):
        self.save_dir = save_dir

    def is_valid_path(self, path):
        if self.path_exists(path):
            return True
        elif self.url_exists(path):
            return True
        else:
            return False

    def url_exists(self, url):
        try:
            response = requests.head(url)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def path_exists(self, path):
        return os.path.exists(path)

    def load_image(self, url):
        if os.path.exists(url):
            img = Image.open(url)
            return img
        elif self.url_exists(url):
            img = Image.open(requests.get(url, stream=True).raw)
            return img
        else:
            raise ValueError("Image URL or path does not exist")

    def save_image(self, img):
        file_name = str(uuid.uuid4())[:4] + ".png"
        output_path = os.path.join(self.save_dir, file_name)
        img.convert("RGB").save(output_path)
        return output_path
