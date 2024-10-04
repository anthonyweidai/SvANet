# System
TextColors = {
    "end_colour": "\033[0m",
    "bold": "\033[1m", # 033 is the escape code and 1 is the color code 
    "error": "\033[31m", # red
    "light_green": "\033[32m",
    "light_yellow": "\033[33m",
    "light_blue": "\033[34m",
    "light_cyan": "\033[36m",
    "warning": "\033[37m", # white
}


# Metric
LogMetric = ["accuracy", "miou", "map"]
WeightMetric = {
    "classificaiton": "maxacc", 
    "segmentation": "maxiou", 
}


# Dataset
RESDICT = {
    "default": {
        "classification" : 224, "segmentation": 512, 
        "detection": 320, "ins_segmentation": 512,
        "autoencoder": 224,
    }, # "regression"
    
    # segmentation
    **dict.fromkeys(
        [
            "spermhealth",
            "isic2018t1", "polypgen",
            "lizard", "atlas", "kits23", "fives", "dynamicnuclear", "tissuenet",
        ], 
        512
    ),
}

# True if using the default image size of dataset, otherwise False
PRE_RESIZE = {
    "default": False,
    **dict.fromkeys(
        [
            "spermhealth",
            "isic2018t1", "polypgen",
            "lizard", "atlas", "kits23", "fives",
            "dynamicnuclear", "tissuenet",
        ], 
    False
    ),
}

CLASS_NAMES = {
    # classification, autoencoder
    "isic2018t3": ["akiec", "bcc", "bkl", "df", "mel", "nv"],
    "isic2018t1": ["symptoms"],
    **dict.fromkeys(["hyperkvasir", "kvasir-seg", "polypgen"], ["symptoms"]),
    "spermhealth": ["normal", "abnormal"],
    "atlas": ["liver", "tumour"],
    "kits23": ["kidney", "tumor", "cyst"],
    "fives": ["amd", "dr", "glaucoma", "health"], # amd: age-related macular degeneration, dr: diabetic retinopathy
    "dynamicnuclear": ["nuclear"],
    "tissuenet": ["cell", "nucleus"],
}