import imgviz


COLOUR_CODES = {
    'twoclasses': [
        [0, 0, 0], # background
        [255, 255, 255]
        ],
    **dict.fromkeys(
        [
            'default', 
            'semspermna', 'semspermnav1', 'semspermnav2', 
            'atlas', 'fives',
        ], 
        list(map(list, imgviz.label_colormap(256)))
    ),
    }

