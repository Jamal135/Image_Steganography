''' Creation Date: 08/09/2022 '''


from contextlib import suppress
from os import getenv, path
from sys import argv
from dotenv import load_dotenv
from steganography import data_extract, data_insert


def extract():
    ''' Returns: steganographically extracted file from image. '''
    try:
        argv[1]
    except IndexError:
        print(f"Usage: {argv[0]} <file> <outpath (default=.)> [key]")
        exit()
    load_dotenv()
    envkey = getenv("ENVIRONMENTKEY")
    args = {"file": open(argv[1], 'rb')}
    with suppress(IndexError):
        args["key"] = argv[3]
    if envkey is not None:
        args["envkey"] = envkey
    with data_extract(**args) as data:
        try:
            outpath = argv[2]
        except IndexError:
            outpath = path.join(".", data.name)
        with open(outpath, "wb") as output:
            output.write(data.read())


def insert():
    ''' Returns: Image with data steganographically attached. '''
    try:
        argv[1]
    except IndexError:
        print(f"Usage: {argv[0]} <file> <data> [key]")
        exit()
    load_dotenv()
    envkey = getenv("ENVIRONMENTKEY")
    args = {'input_file': open(argv[2], 'rb')}
    with suppress(IndexError):
        args["key"] = argv[4]
    if envkey is not None:
        args["envkey"] = envkey
    with open(argv[1], 'rb') as args["image_file"]:
        with data_insert(**args) as image:
            with open(argv[3], "wb") as output:
                output.write(image.read())
