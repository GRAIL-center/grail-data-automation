import os

def getBase(file):
    base = os.path.splitext(file)[0]
    if "attachment" in base:
        print(base)
        base = base.split("__")
        base = '__s'.join(base[:-1])
    return base

def organizeDuplicates(folder):
    bases = [getBase(i) for i in folder]
    # print(bases)
    find = lambda a, l: [i for i in l if getBase(i).startswith(a)]
    return [find(i, folder) for i in bases]

def extractFolder(folder_name, folder_path="./comments"):
    folder = f"{folder_path}/{folder_name}"
    files = os.listdir(folder)
    files = [i for i in files if i.lower().endswith(".pdf")]
    print(organizeDuplicates(files))
    
extractFolder("2021-10861")