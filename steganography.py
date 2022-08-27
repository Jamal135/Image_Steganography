''' Creation Date: 27/08/2022 '''


from PIL import Image
from functools import reduce
from itertools import product
from secrets import token_hex
from random import seed, sample, randint


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
    seed(key)
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


def generate_header(Configuration: object):
    ''' Returns: Built binary header data specifying settings. '''
    colours = list(Configuration.COLOURS.values())
    indexs = list(Configuration.INDEXS.values())
    method = Configuration.METHOD
    method_bool = "1" if method == "random" else "0"
    colour_table = ["0", "0", "0"]
    for colour in colours:
        colour_table[colour] = "1"
    index_table = ["0", "0", "0", "0", "0", "0", "0", "0"]
    for index in indexs:
        index_table[index] = "1"
    return method_bool + "".join(colour_table) + "".join(index_table)


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
    length = len(header) # Stored as random method any colour, smallest index
    colours = random_sample(key, [0,1,2], length)
    colours = [item for sublist in colours for item in sublist]
    pixels = [list(Image.getpixel((coords[point][0], coords[point][1])))
              for point in range(length - 1)]
    for i, pixel in enumerate(pixels):
        value = integer_conversion(pixel[colours[i]], 'binary')
        modified_value = integer_conversion(value[:-1] + header[i], 'integer')
        pixels[i][colours[i]] = modified_value 
    pixels = [tuple(list) for list in pixels] # Convert back to tuples
    [Image.putpixel((coords[point][0], coords[point][1]), pixels[point])
     for point in range(length - 1)]
    return coords[length:], Image    


def build_object(key: int, method: str, noise: bool, colours: list, indexs: list):
    ''' Returns: Configuration object of steganographic storage settings. '''
    if colours == None:
        colours = [0,1,2] # Default to all colours
    if indexs == None:
        indexs = [6,7] # Default to two least significant bits
    colour_list = ['red', 'green', 'blue']
    index_list = ['1st', '2nd', '3rd', '4th', '5th', '6th', '7th']
    class Configuration():
        COLOURS = {colour_list[i]: i for i, _ in enumerate(colour_list) if i in colours}
        INDEXS = {index_list[i]: i + 1 for i, _ in enumerate(index_list) if i + 1 in indexs}
        VOLUME = len(colours) * len(indexs) if method == 'all' else len(indexs) 
        METHOD = method
        NOISE = noise
        KEY = key
    return Configuration


def binary_conversion(data: str, method: str):
    ''' Returns: Data converted to or from binary. '''
    if method == 'binary':
        return ''.join([bin(byte)[2:].zfill(8) for byte in bytearray(data, 'utf-8')])
    else:
        byte_list = int(data, 2).to_bytes(len(data) // 8, byteorder = 'big')
        return byte_list.decode('utf-8')


def generate_numbers(min_value: int, max_value: int, number_values: int):
    ''' Returns: Variable length string of random numbers in range. '''
    seed(token_hex(64))
    return "".join([str(randint(min_value, max_value)) for _ in range(number_values)])


def generate_message(Configuration: object, data: str, coords: list):
    ''' Returns: '''
    capacity = len(coords)
    data = binary_conversion(data, 'binary')
    end_key_size = len(integer_conversion(capacity, 'binary'))
    data_size = len(data)
    size = end_key_size + data_size
    if size > capacity: # Test if message can fit inside the image
        raise ValueError(f"Message size exceeded by {size - capacity} bits")
    noise = generate_numbers(0, 1, capacity - size) if Configuration.NOISE else ''
    end_key = integer_conversion(data_size, 'binary').zfill(end_key_size)
    return end_key + data + noise # Binary, end key specifies index of data end


def generate_coords(Configuration: object, Size: object, pixel_coords: list):
    ''' Returns: Shuffled data location tuples (Width, Height, Colour, Index). '''
    colours = list(Configuration.COLOURS.values())
    indexs = list(Configuration.INDEXS.values())
    method = Configuration.METHOD
    key = Configuration.KEY
    length = Size.PIXELS
    if method == 'random': # If random need to pick random colour option per pixel
        colours = random_sample(key, colours, length)
    data_coords = []
    for i, coordinate in enumerate(pixel_coords):
        for colour in colours[i] if method == 'random' else colours:
            for index in indexs:
                data_coords.append((coordinate[0], coordinate[1], colour, index))
    return shuffle(key, data_coords)


def attach_data(Image: Image, Configuration: object, binary_message: str, coords: list):
    ''' Returns: Image with all required pixels steganographically modified. '''
    if not Configuration.NOISE: # Optimise if not modifying every pixel
        coords = coords[:len(binary_message)]
    pixels = [list(Image.getpixel((location[0], location[1]))) for location in coords]
    for i, point in enumerate(coords):
        value = list(integer_conversion(pixels[i][point[2]], 'binary'))
        value[point[3]] = binary_message[i]
        modified_value = integer_conversion(''.join(value), 'integer')
        pixels[i][point[2]] = modified_value
    pixels = [tuple(list) for list in pixels] # Convert back to tuples
    [Image.putpixel((coords[i][0], coords[i][1]), point)
     for i, point in enumerate(pixels)]
    return Image


def save_image(filename: str, Image: Image, type: str = '.png'):
    ''' Returns: Saved image at location output. '''
    if not filename.endswith(type):
        filename = f'{filename}_result{type}'
    else:
        filename = f'{filename[:-4]}_result{type}'
    Image.save(f'Images/{filename}')


def data_insert(filename: str, key: str, data: str, method: str = 'random', 
                colours: list = None, indexs: list = None, noise: bool = True):
    ''' Returns: Selected image with secret data steganographically attached. '''
    Image, Size = load_image(filename)
    coords, image_key = generate_context(key, Image, Size)
    Configuration = build_object(image_key, method, noise, colours, indexs)
    header = generate_header(Configuration) # Specifies Configuration for extract
    cut_coords, Image = attach_header(Image, image_key, header, coords)
    data_coords = generate_coords(Configuration, Size, cut_coords)
    binary_message = generate_message(Configuration, data, data_coords)
    Image = attach_data(Image, Configuration, binary_message, data_coords)
    save_image(filename, Image)
    
# Bug where 'all' method does not work. Perhaps coords area? Perhaps attach data issue... not sure
data_insert('gate', "I like pineapples with toast", "hello world", method = 'all', indexs = [0,1,2,3,4,5,6,7])
