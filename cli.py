''' Creation Date: 08/09/2022 '''


from contextlib import suppress
from sys import argv
from os import getenv, path
from posixpath import splitext
from genericpath import exists
from dotenv import load_dotenv
from steganography import data_extract, data_insert


def uniquify(file_path: str):
    ''' Returns: File path unique from existing files. '''
    if not exists(file_path):
        return file_path
    filename, extension = splitext(file_path)
    count = 1
    while exists(file_path):
        file_path = f'{filename}-{str(count)}{extension}'
        count += 1
    return file_path


def default_name(file_path: str):
    ''' Returns: File path with "-steg" added at end. '''
    filename, extension = splitext(file_path)
    if extension == '':
        return f'{filename}-steg'
    return f'{filename}-steg{extension}'


def extract():
    ''' Returns: Steganographically extracted file from image. '''
    try:
        argv[1]
    except IndexError:
        print(f'Usage: {argv[0]} <file> <outpath (default=.)> [key]')
        exit()
    load_dotenv()
    envkey = getenv('ENVIRONMENTKEY')
    args = {'file': open(argv[1], 'rb')}
    with suppress(IndexError):
        args['key'] = argv[3]
    if envkey is not None:
        args['envkey'] = envkey
    with data_extract(**args) as data:
        try:
            outpath = argv[2]
        except IndexError:
            outpath = uniquify(path.join('.', data.name))
        with open(outpath, 'wb') as output:
            output.write(data.read())


def insert():
    ''' Returns: Image with data steganographically attached. '''
    try:
        argv[1]
    except IndexError:
        print(f'Usage: {argv[0]} <file> <data> [key]')
        exit()
    load_dotenv()
    envkey = getenv('ENVIRONMENTKEY')
    args = {'input_file': open(argv[2], 'rb')}
    with suppress(IndexError):
        args['key'] = argv[4]
    if envkey is not None:
        args['envkey'] = envkey
    with open(argv[1], 'rb') as args['image_file']:
        with data_insert(**args) as image:
            try:
                output = argv[3]
            except IndexError:
                output = uniquify(default_name(argv[1]))
            with open(output, "wb") as output:
                output.write(image.read())
