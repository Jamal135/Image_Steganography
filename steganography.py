''' Creation Date: 27/08/2022 '''


from PIL import Image
from functools import reduce
from itertools import product
from secrets import token_hex
from random import seed, sample, randint


def verify_string(items: list):
    ''' Purpose: Check all items in list are valid strings. '''
    for item in items:
        if not isinstance(item, str):
            raise ValueError(f'Variable is invalid string: {item}')


def decimal_encoding(text: str):
    ''' Returns: Text converted to base10 integer. '''
    try:
        return int(reduce(lambda a, b: a * 256 + b, map(ord, text), 0))
    except Exception as e:
        raise ValueError(f'Failed to encode: {text}') from e


def load_image(filename: str, type: str = '.png'):
    ''' Returns: Image object and Enum of WIDTH and HEIGHT. '''
    if not filename.endswith(type): # Only support PNG
        filename += type
    try:
        image = Image.open(f'Images/{filename}')
    except Exception as e:
        raise ValueError(f'No .PNG at Images/{filename}') from e
    size = image.size
    class Size():
        WIDTH = size[0]
        HEIGHT = size[1]
        PIXELS = size[0] * size[1]
    return image, Size


def shuffle(key: int, data):
    ''' Returns: Data shuffled with key as seed. '''
    seed(key) # Same result with same key and data
    return sample(data, len(data))


def generate_context(key: int, Image: Image, Size: object, key_pixels: int = 16):
    ''' Returns: List of tuple coordinates in image and image specific key. '''
    key = decimal_encoding(key)
    key *= (Size.PIXELS * 99) # Adjust key by image size
    coords = shuffle(key, [*product(range(Size.WIDTH), range(Size.HEIGHT))])
    pixels = [Image.getpixel((coords[point][0], coords[point][1]))
              for point in range(key_pixels - 1)]
    key *= (sum(map(sum, pixels))) # Adjust key by key pixels
    coords = shuffle(key, coords[key_pixels:])
    return coords, key


def generate_header(Config: object):
    ''' Returns: Built binary header data specifying settings. '''
    method_bin = '1' if Config.METHOD == 'random' else '0'
    stored_bin = '1' if Config.STORED == 'data' else '0'
    encrypt_bin = '1' if Config.ENCRYPT == True else '0'
    colour_table = ['0', '0', '0']
    for colour in Config.COLOURS:
        colour_table[colour] = '1'
    colour_bin = ''.join(colour_table)
    index_table = ['0', '0', '0', '0', '0', '0', '0', '0']
    for index in Config.INDEXS:
        index_table[index] = '1'
    index_bin = ''.join(index_table)
    return method_bin + stored_bin + encrypt_bin + colour_bin + index_bin


def random_sample(key: int, options: list, length: int, number_picked: int = 1):
    ''' Returns: Variable length list of lists of selected options. '''
    seed(key)
    return [sample(options, k = number_picked) for _ in range(length)]


def integer_conversion(data: int, method: str):
    ''' Returns: Number converted to or from binary. '''
    if method == 'binary':
        return bin(data).replace('0b', '').zfill(8)
    else:
        return int(data, 2)


def attach_header(Image: Image, key: int, header: str, coords: list):
    ''' Returns: Modified image with header data attached for extraction. '''
    LENGTH = 14 # Stored as random method, any colour, smallest index
    header_coords = coords[:LENGTH]
    colours = random_sample(key, [0,1,2], LENGTH)
    colours = [item for sublist in colours for item in sublist]
    for i, position in enumerate(header_coords):
        pixel = list(Image.getpixel((position[0], position[1])))
        value = integer_conversion(pixel[colours[i]], 'binary')
        modified_value = integer_conversion(value[:-1] + header[i], 'integer')
        pixel[colours[i]] = modified_value 
        Image.putpixel((coords[i][0], coords[i][1]), tuple(pixel))
    return coords[LENGTH:], Image    


def list_verification(variable: str, items: list, allowed: list):
    ''' Purpose: Tests that list variable is valid. '''
    try:
        if any(item not in allowed for item in items):
            raise ValueError(f'Invalid {variable} list argument: {items}')
        return len(set(items)) == len(items)
    except Exception as e:
        raise ValueError(f'Invalid {variable} list argument: {items}') from e


def bool_verification(variable: str, value: bool):
    ''' Purpose: Tests that boolean variable is valid. '''
    if value not in [True, False]:
        raise ValueError(f'Invalid boolean {variable} argument: {value}')


def str_verification(variable: str, value: str, allowed: list):
    ''' Purpose: Tests that string argument variable is valid. '''
    if value not in allowed:
        raise ValueError(f'Invalid string {variable} argument: {value}')


def build_object(key: int, method: str, stored: str, encrypt: bool, 
                 colours: list, indexs: list, noise: bool = False):
    ''' Returns: Configuration object of steganographic storage settings. '''
    if colours is None:
        colours = [0, 1, 2]
    if indexs is None:
        indexs = [6, 7]
    str_verification('method', method, ['random', 'all'])
    str_verification('stored', stored, ['data', 'file'])
    list_verification('indexs', indexs, [0,1,2,3,4,5,6,7])
    list_verification('colours', colours, [0,1,2])
    bool_verification('encrypt', encrypt)
    bool_verification('noise', noise)
    class Config:
        VOLUME = len(colours) * len(indexs) if method == 'all' else len(indexs)
        COLOURS = colours
        ENCRYPT = encrypt
        STORED = stored
        INDEXS = indexs
        METHOD = method
        NOISE = noise
        KEY = key
    return Config


def binary_conversion(data: str, method: str):
    ''' Returns: Data converted to or from binary. '''
    if method == 'binary':
        return ''.join([bin(byte)[2:].zfill(8) for byte in bytearray(data, 'utf-8')])
    byte_list = int(data, 2).to_bytes(len(data) // 8, byteorder='big')
    return byte_list.decode('utf-8')


def binary_file(data: str):
    ''' Returns: . '''
    if not data.endswith('.7z'):
        pass


def generate_numbers(min_value: int, max_value: int, number_values: int):
    ''' Returns: Variable length string of random numbers in range. '''
    seed(token_hex(64))
    return ''.join([str(randint(min_value, max_value)) for _ in range(number_values)])


def generate_message(Config: object, data: str, coords: list):
    ''' Returns: Generated binary data to be attached to image. '''
    capacity = len(coords)
    if Config.STORED == 'data':
        data = binary_conversion(data, 'binary')
    else:
        data = binary_file(data)
    end_key_size = len(integer_conversion(capacity, 'binary'))
    data_size = len(data)
    size = end_key_size + data_size
    if size > capacity: # Test if message can fit inside the image
        raise ValueError(f'Message size exceeded by {size - capacity} bits')
    noise = generate_numbers(0, 1, capacity - size) if Config.NOISE else ''
    end_key = integer_conversion(data_size, 'binary').zfill(end_key_size)
    return end_key + data + noise # Binary, end key specifies index of data end


def generate_coords(Config: object, Size: object, pixel_coords: list):
    ''' Returns: Shuffled data location tuples (Width, Height, Colour, Index). '''
    if Config.METHOD == 'random': # If random need to pick random colour option per pixel
        if len(Config.COLOURS) == 1: # Don't random sample if only one colour option
            colours = [Config.COLOURS] * Size.PIXELS
        else:
            colours = random_sample(Config.KEY, Config.COLOURS, Size.PIXELS)
    data_coords = []
    for i, coordinate in enumerate(pixel_coords):
        for colour in colours[i] if Config.METHOD == 'random' else Config.COLOURS:
            data_coords.extend((coordinate[0], coordinate[1], colour, index) 
                               for index in Config.INDEXS)
    return shuffle(Config.KEY, data_coords)


def attach_data(Image: Image, Config: object, binary_message: str, coords: list):
    ''' Returns: Image with all required pixels steganographically modified. '''
    if not Config.NOISE: # Optimise if not modifying every pixel
        coords = coords[:len(binary_message)]
    for i, position in enumerate(coords):
        pixel = list(Image.getpixel((position[0], position[1])))
        value = list(integer_conversion(pixel[position[2]], 'binary'))
        value[position[3]] = binary_message[i]
        modified_value = integer_conversion(''.join(value), 'integer')
        pixel[position[2]] = modified_value
        Image.putpixel((coords[i][0], coords[i][1]), tuple(pixel))
    return Image


def save_image(filename: str, Image: Image, type: str = '.png'):
    ''' Returns: Saved image at location output. '''
    filename = f'{filename[:-4]}_result{type}' if filename.endswith(type) \
                else f'{filename}_result{type}'
    Image.save(f'Images/{filename}')


def extract_header(Image: Image, key: int, coords: list):
    ''' Returns: Header data extracted and unpacked. '''
    LENGTH = 14 # Header coded to 1 for true, 0 for false
    header_coords = coords[:LENGTH]
    colours = random_sample(key, [0,1,2], LENGTH)
    colours = [item for sublist in colours for item in sublist]
    header = []
    for i, position in enumerate(header_coords):
        pixel = list(Image.getpixel((position[0], position[1])))
        value = integer_conversion(pixel[colours[i]], 'binary')
        header.append(value[-1])
    method = 'random' if header[0] == '1' else 'all'
    stored = 'data' if header[1] == '1' else 'file'
    encrypt = header[2] == '1'
    colours = [i for i in range(3) if header[i + 3] == '1']
    indexs = [i for i in range(8) if header[i + 6] == '1']
    return [method, stored, encrypt, colours, indexs], coords[LENGTH:]


def extract_data(Image: Image, coords: list):
    ''' Returns: All binary data extracted from given coordinates. '''
    data = []
    for position in coords:
        pixel = list(Image.getpixel((position[0], position[1])))
        value = list(integer_conversion(pixel[position[2]], 'binary'))
        data.append(value[position[3]])
    return ''.join(data)


def extract_message(Image: Image, coords: list):
    ''' Returns: Data stored steganographically within the image. '''
    capacity = len(coords)
    end_key_size = len(integer_conversion(capacity, 'binary'))
    try:
        end_key = extract_data(Image, coords[:end_key_size]) 
        data_size = integer_conversion(end_key, 'integer')
        coords = coords[end_key_size: data_size + end_key_size]
        binary_message = extract_data(Image, coords)
        return binary_conversion(binary_message, 'data')
    except Exception as e:
        raise ValueError('Invalid data extracted') from e


def data_insert(filename: str, data: str, key: str = '999', method: str = 'random',
                stored: str = 'data', colours: list = None, indexs: list = None, 
                noise: bool = False, encrypt: bool = False):
    ''' Returns: Selected image with secret data steganographically attached. '''
    verify_string([filename, data, key])
    Image, Size = load_image(filename)
    coords, image_key = generate_context(key, Image, Size)
    Config = build_object(image_key, method, stored, encrypt, colours, indexs, noise)
    header = generate_header(Config) # Specifies Configuration for extract
    cut_coords, Image = attach_header(Image, image_key, header, coords)
    data_coords = generate_coords(Config, Size, cut_coords)
    binary_message = generate_message(Config, data, data_coords)
    Image = attach_data(Image, Config, binary_message, data_coords)
    save_image(filename, Image)


def data_extract(filename: str, key: str = '999'):
    ''' Returns: Data steganographically extracted from selected image. '''
    verify_string([filename, key])
    Image, Size = load_image(filename)
    coords, image_key = generate_context(key, Image, Size)
    setup, cut_coords = extract_header(Image, image_key, coords)
    Config = build_object(image_key, setup[0], setup[1], setup[2], setup[3], setup[4])
    data_coords = generate_coords(Config, Size, cut_coords)
    return extract_message(Image, data_coords)
