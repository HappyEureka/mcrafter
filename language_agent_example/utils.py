import base64
from mimetypes import guess_type
from PIL import Image
import io

def local_image_to_data_url(image_path):
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:{mime_type};base64,{base64_encoded_data}"

def numpy_to_data_url(numpy_array, mime_type="image/png"):
    image = Image.fromarray(numpy_array)
    buffer = io.BytesIO()
    image.save(buffer, format=mime_type.split("/")[-1].upper())
    base64_encoded_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:{mime_type};base64,{base64_encoded_data}"

def print_color(*args, **kwargs):
    color = kwargs['color']
    color_code = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'magenta': '\033[95m',
        'cyan': '\033[96m',
        'white': '\033[97m',
        'black': '\033[30m',
    }.get(color, '\033[30m')
    print(color_code, end='')
    print(*args, end='')
    print('\033[0m')