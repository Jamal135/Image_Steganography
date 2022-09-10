''' Creation Date: 27/08/2022 '''


from io import BufferedReader, BytesIO
from os import path
from tempfile import TemporaryDirectory
from typing import BinaryIO #, IO
from functools import reduce
from itertools import product
from secrets import token_hex
from random import seed, sample, randint
from PIL import Image


DEFAULT_ENVKEY = '122stegodefault2923283283238232' # Instance key
HEADER_LENGTH = 13 # Length of the header bits string


class Size:
    ''' Specifies dimensions of image file. '''
    width: int
    ''' Pixel width of the image file. '''
    height: int
    ''' Pixel height of the image file. '''
    pixels: int
    ''' Number of pixels in image file. '''


class Config:
    ''' Specifies steganography configuration. '''
    colours: list
    ''' Unique list of any integers 0-2 for RGB colours. '''
    indexs: list
    ''' Unique list of any integers 0-7 for bit indexs. '''
    encrypt: bool
    ''' Boolean if data is encrypted or plaintext. '''
    noise: bool
    ''' Boolean if empty data space is filled or untouched. '''
    volume: int
    ''' Integer number data bit positions per pixel. '''
    key: int
    ''' Integer password for how data is stored in image. '''
    method: str # Revisit later
    ''' If all, all colours used, else if random one picked per pixel. '''


class FileData:
    ''' Specifies file and file data. '''
    filename: str
    ''' Name of the file with file extension. '''
    data: bytes
    ''' The bytes of the given file. '''


def verify_string(items: list):
    ''' Purpose: Check all items in list are valid strings. '''
    for item in items:
        if not isinstance(item, str):
            raise ValueError(f'Variable is invalid string: {item}')


def load_image(file: BinaryIO):
    ''' Returns: Image object and class of width, height, and size. '''
    img = Image.open(file)
    size = img.size
    return img, size(width = size[0], height = size[1], pixels = size[0] * size[1])


# def env_extract():
#     ''' Returns: Environment key else default key. '''
#     envkey = getenv('ENVIRONMENTKEY')
#     if envkey is None:
#         return '122stegodefault2923283283238232'
#     else:
#         return envkey


def shuffle(key: int, data):
    ''' Returns: Data shuffled with key as seed. '''
    seed(key)  # Same result with same key and data
    return sample(data, len(data))


def decimal_encoding(text: str):
    ''' Returns: Text converted to base10 integer. '''
    try:
        return int(reduce(lambda a, b: a * 256 + b, map(ord, text), 0))
    except Exception as error:
        raise ValueError(f'Failed to encode: {text}') from error


def generate_context(key: int, envkey: str, image: Image, size: Size, key_pixels: int = 16):
    ''' Returns: List of tuple coordinates in image and image specific key. '''
    coords = [*product(range(size.width), range(size.height))]
    environment_key = decimal_encoding(envkey)
    coords = shuffle(environment_key, coords)
    key = decimal_encoding(key)
    key *= (size.pixels * 99)  # Adjust key by image size
    coords = shuffle(key, [*product(range(size.width), range(size.height))])
    pixels = [image.getpixel((coords[point][0], coords[point][1]))
              for point in range(key_pixels - 1)]
    key *= (sum(map(sum, pixels)))  # Adjust key by key pixels
    coords = shuffle(key, coords[key_pixels:])
    return coords, key


def generate_header(config: Config):
    ''' Returns: Built binary header data specifying settings. '''
    method_bin = '1' if config.method == 'random' else '0'
    encrypt_bin = '1' if config.encrypt else '0'
    colour_table = ['0', '0', '0']
    for colour in config.colours:
        colour_table[colour] = '1'
    colour_bin = ''.join(colour_table)
    index_table = ['0', '0', '0', '0', '0', '0', '0', '0']
    for index in config.indexs:
        index_table[index] = '1'
    index_bin = ''.join(index_table)
    return method_bin + encrypt_bin + colour_bin + index_bin


def random_sample(key: int, options: list, length: int, number_picked: int = 1):
    ''' Returns: Variable length list of lists of selected options. '''
    seed(key)
    return [sample(options, k=number_picked) for _ in range(length)]


def integer_conversion(data: int, method: str):
    ''' Returns: Number converted to or from binary. '''
    if method == 'binary':
        return bin(data).replace('0b', '').zfill(8)
    else:
        return int(data, 2)


def attach_header(image: Image.Image, key: int, header: str, coords: list):
    ''' Returns: Modified image with header data attached for extraction. '''
    header_coords = coords[:HEADER_LENGTH]
    colours = random_sample(key, [0, 1, 2], HEADER_LENGTH)
    colours = [item for sublist in colours for item in sublist]
    for i, position in enumerate(header_coords):
        pixel = list(image.getpixel((position[0], position[1])))
        value = integer_conversion(pixel[colours[i]], 'binary')
        modified_value = integer_conversion(value[:-1] + header[i], 'integer')
        pixel[colours[i]] = modified_value
        image.putpixel((coords[i][0], coords[i][1]), tuple(pixel))
    return coords[HEADER_LENGTH:], image


def list_verification(variable: str, items: list, allowed: list):
    ''' Purpose: Tests that list variable is valid. '''
    try:
        if any(item not in allowed for item in items):
            raise ValueError(f'Invalid {variable} list argument: {items}')
        return len(set(items)) == len(items)
    except Exception as error:
        raise ValueError(f'Invalid {variable} list argument: {items}') from error


def bool_verification(variable: str, value: bool):
    ''' Purpose: Tests that boolean variable is valid. '''
    if value not in [True, False]:
        raise ValueError(f'Invalid boolean {variable} argument: {value}')


def str_verification(variable: str, value: str, allowed: list):
    ''' Purpose: Tests that string argument variable is valid. '''
    if value not in allowed:
        raise ValueError(f'Invalid string {variable} argument: {value}')


def build_object(key: int, method: str, encrypt: bool, colours: list,
                 indexs: list, noise: bool = False):
    ''' Returns: Configuration object of steganographic storage settings. '''
    if colours is None:
        colours = [0, 1, 2]
    if indexs is None:
        indexs = [6, 7]
    str_verification('method', method, ['random', 'all'])
    list_verification('indexs', indexs, [0, 1, 2, 3, 4, 5, 6, 7])
    list_verification('colours', colours, [0, 1, 2])
    bool_verification('encrypt', encrypt)
    bool_verification('noise', noise)
    volume = len(colours) * len(indexs) if method == 'all' else len(indexs)
    return Config(volume = volume, colours = colours, encrypt = encrypt,
                  indexs = indexs, method = method, noise = noise, key = key)


def binary_encode(file: BinaryIO):
    ''' Returns: File at data of data convert to binary. '''
    data_bytes = file.name.encode('utf-8') + file.read()
    return ''.join(f'{byte:08b}' for byte in data_bytes)


def binary_decode(data: str):
    ''' Returns: Binary string converted to file of data. '''
    data_bytes = int(data, 2).to_bytes((len(data) + 7) // 8, byteorder='big')
    file_bytes, data_bytes = data_bytes.split(b'..', 1)
    file = BytesIO()
    file.name = file_bytes.decode('utf-8')
    file.write(data_bytes)
    file.seek(0)
    return file


def generate_numbers(min_value: int, max_value: int, number_values: int):
    ''' Returns: Variable length string of random numbers in range. '''
    seed(token_hex(64))
    return ''.join([str(randint(min_value, max_value)) for _ in range(number_values)])


def generate_message(config: Config, data: BinaryIO, coords: list):
    ''' Returns: Generated binary data to be attached to image. '''
    capacity = len(coords)
    encoded = binary_encode(data)
    end_key_size = len(integer_conversion(capacity, 'binary'))
    data_size = len(encoded)
    size = end_key_size + data_size
    if size > capacity:  # Test if message can fit inside the image
        raise ValueError(f'Message size exceeded by {size - capacity} bits')
    noise = generate_numbers(0, 1, capacity - size) if config.noise else ''
    end_key = integer_conversion(data_size, 'binary').zfill(end_key_size)
    return end_key + encoded + noise  # Binary, end key specifies index of data end


def generate_coords(config: Config, size: Size, pixel_coords: list):
    ''' Returns: Shuffled data location tuples (Width, Height, Colour, Index). '''
    if config.method == 'random':  # If random need to pick random colour option per pixel
        if len(config.colours) == 1:  # Don't random sample if only one colour option
            colours = [config.colours] * size.pixels
        else:
            colours = random_sample(config.key, config.colours, size.pixels)
    data_coords = []
    for i, coordinate in enumerate(pixel_coords):
        for colour in colours[i] if config.method == 'random' else config.colours:
            data_coords.extend((coordinate[0], coordinate[1], colour, index)
                               for index in config.indexs)
    return shuffle(config.key, data_coords)


def attach_data(image: Image.Image, config: Config, binary_message: str, coords: list):
    ''' Returns: Image with all required pixels steganographically modified. '''
    if not config.noise:  # Optimise if not modifying every pixel
        coords = coords[:len(binary_message)]
    for i, position in enumerate(coords):
        pixel = list(image.getpixel((position[0], position[1])))
        value = list(integer_conversion(pixel[position[2]], 'binary'))
        value[position[3]] = binary_message[i]
        modified_value = integer_conversion(''.join(value), 'integer')
        pixel[position[2]] = modified_value
        image.putpixel((coords[i][0], coords[i][1]), tuple(pixel))
    return image


# def uniquify(file: str):
#     ''' Returns: File path unique from existing files. '''
#     if not exists(file):
#         return file
#     filename, extension = splitext(file)
#     count = 1
#     while exists(file):
#         file = f'{filename}_{str(count)}{extension}'
#         count += 1
#     return file

# def save_image(filename: str, Image: Image, overwrite: bool, extension: str = '.png'):
#     ''' Returns: Saved image at location output. '''
#     filename = f'Files/{filename[:-4]}_stego122{extension}' if filename.endswith(extension) \
#         else f'Files/{filename}_stego122{extension}'
#     if not overwrite:
#         filename = uniquify(f'{filename}')
#     Image.save(filename)


def extract_header(img: Image, key: int, coords: list):
    ''' Returns: Header data extracted and unpacked. '''
    header_coords = coords[:HEADER_LENGTH]
    colours = random_sample(key, [0, 1, 2], HEADER_LENGTH)
    colours = [item for sublist in colours for item in sublist]
    header = []
    for i, position in enumerate(header_coords):
        pixel = list(img.getpixel((position[0], position[1])))
        value = integer_conversion(pixel[colours[i]], 'binary')
        header.append(value[-1])
    method = 'random' if header[0] == '1' else 'all'
    encrypt = header[2] == '1'
    colours = [i for i in range(3) if header[i + 3] == '1']
    indexs = [i for i in range(8) if header[i + 6] == '1']
    return [method, encrypt, colours, indexs], coords[HEADER_LENGTH:]


def extract_data(img: Image, coords: list):
    ''' Returns: All binary data extracted from given coordinates. '''
    data = []
    for position in coords:
        pixel = list(img.getpixel((position[0], position[1])))
        value = list(integer_conversion(pixel[position[2]], 'binary'))
        data.append(value[position[3]])
    return ''.join(data)


def extract_message(img: Image, coords: list):
    ''' Returns: Data that was steganographically inside the image. '''
    capacity = len(coords)
    end_key_size = len(integer_conversion(capacity, 'binary'))
    try:
        end_key = extract_data(img, coords[:end_key_size])
        data_size = integer_conversion(end_key, 'integer')
        coords = coords[end_key_size: data_size + end_key_size]
        return extract_data(img, coords)
    except Exception as error:
        raise ValueError('Invalid data extracted') from error


def data_insert(image_file: BufferedReader, input_file: BufferedReader, key: str = '999',
                envkey: str = DEFAULT_ENVKEY, method: str = 'random', colours: list = None,
                indexs: list = None, noise: bool = False, encrypt: bool = False):
    ''' Returns: Selected image with secret data steganographically attached. '''
    verify_string([key])
    old_image, size = load_image(image_file)
    coords, image_key = generate_context(key, envkey, old_image, size)
    config = build_object(image_key, method, encrypt, colours, indexs, noise)
    header = generate_header(config) # Specifies Configuration for extract
    cut_coords, image = attach_header(old_image, image_key, header, coords)
    data_coords = generate_coords(config, size, cut_coords)
    binary_message = generate_message(config, input_file, data_coords)
    image = attach_data(image, config, binary_message, data_coords)
    with TemporaryDirectory() as tempdir:
        image.save(path.join(tempdir, f'temp.{image.format}'))
        return open(path.join(tempdir, f'temp.{image.format}'), 'rb')


def data_extract(file: BinaryIO, key: str = '999', envkey: str = DEFAULT_ENVKEY):
    ''' Returns: Data steganographically extracted from selected image. '''
    verify_string([key])
    image, size = load_image(file)
    coords, image_key = generate_context(key, envkey, image, size)
    setup, cut_coords = extract_header(image, image_key, coords)
    config = build_object(
        image_key, setup[0], setup[1], setup[2], setup[3], setup[4])
    data_coords = generate_coords(config, size, cut_coords)
    binary = extract_message(image, data_coords)
    return binary_decode(binary)
