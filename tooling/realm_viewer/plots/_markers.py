_MARKER_MAP = {
    'o': 'circle',
    's': 'square',
    '^': 'triangle-up',
    'v': 'triangle-down',
    '<': 'triangle-left',
    '>': 'triangle-right',
    'D': 'diamond',
    'P': 'cross',
    'X': 'x',
    '*': 'star',
}


def mpl_marker_to_plotly(marker):
    return _MARKER_MAP.get(marker, 'circle')
