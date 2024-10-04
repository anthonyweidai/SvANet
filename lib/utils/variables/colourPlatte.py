import imgviz


COLOUR_CODES = {
    "twoclasses": [
        [0, 0, 0], # background
        [255, 255, 255]
    ],
    **dict.fromkeys(
        [
            "default", 
            "spermhealth", "atlas", "fives", "kits23"
        ], 
        list(map(list, imgviz.label_colormap(256)))
    ),
    }

