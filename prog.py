from os import getenv, path
from steganography import data_extract, data_insert
from sys import argv
from dotenv import load_dotenv

def extract():
    try: argv[1]
    except IndexError: 
        print(f"Usage: {argv[0]} <file> <outpath (default=.)> [key]")
        exit()
    load_dotenv()
    envkey = getenv("ENVIRONMENTKEY")
    args = {}
    args["file"] = open(argv[1], 'rb')
    try:
        args["key"] = argv[3]
    except IndexError:
        pass
    if envkey is not None:
        args["envkey"] = envkey
    with data_extract(**args) as data:
        try: outpath = argv[2]
        except IndexError: outpath = path.join(".", data.name)
        with open(outpath, "wb") as o:
            o.write(data.read())

def insert():
    try: argv[1]
    except IndexError:
        print(f"Usage: {argv[0]} <file> <data> [key]")
        exit()
    load_dotenv()
    envkey = getenv("ENVIRONMENTKEY")
    args = {}
    args["input_file"] = open(argv[2], 'rb')
    try:
        args["key"] = argv[4]
    except IndexError:
        pass
    if envkey is not None:
        args["envkey"] = envkey
    with open(argv[1], 'rb') as args["image_file"]:
        with data_insert(**args) as image:
            with open(argv[3], "wb") as o:
                o.write(image.read())