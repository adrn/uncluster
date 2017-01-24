from collections import OrderedDict
from cycler import cycler

palette = OrderedDict()
palette['k'] = palette['black'] = '#444444'
palette['b'] = palette['blue'] = '#2166AC'
palette['g'] = palette['green'] = '#006837'
palette['r'] = palette['red'] = '#B2182B'
palette['p'] = palette['purple'] = '#762A83'
palette['o'] = palette['orange'] = '#E08214'
palette['t'] = palette['teal'] = '#80CDC1'
palette['pink'] = '#C51B7D'
palette['y'] = palette['yellow'] = '#FEE08B'

# Stylistic things:
mpl_style = {

    # Lines
    'lines.linewidth': 1.5,
    'lines.antialiased': True,
    'lines.marker': '.',
    'lines.markersize': 5.,

    # Patches
    'patch.linewidth': 1.0,
    'patch.facecolor': '#348ABD',
    'patch.edgecolor': '#CCCCCC',
    'patch.antialiased': True,

    # images
    'image.origin': 'upper',

    # colormap
    'image.cmap': 'hesperia',

    # Font
    'font.size': 16.0,
    'text.latex.preamble': r'\usepackage{amsmath}',
    'axes.unicode_minus': False,

    # Axes
    'axes.facecolor': '#FFFFFF',
    'axes.edgecolor': '#555555',
    'axes.linewidth': 1.0,
    'axes.grid': False,
    'axes.titlesize': 'x-large',
    'axes.labelsize': 'large',
    'axes.labelcolor': 'k',
    'axes.axisbelow': True,
    'axes.prop_cycle': cycler('color', [palette['black'], palette['blue'], palette['green'],
                                        palette['red'], palette['purple'], palette['orange'],
                                        palette['teal'], palette['pink'], palette['yellow']]),

    # Ticks
    'xtick.top': True,
    'xtick.major.size': 8,
    'xtick.minor.size': 4,
    'xtick.major.pad': 6,
    'xtick.minor.pad': 6,
    'xtick.color': '#555555',
    'xtick.direction': 'in',
    'ytick.right': True,
    'ytick.major.size': 8,
    'ytick.minor.size': 4,
    'ytick.major.pad': 6,
    'ytick.minor.pad': 6,
    'ytick.color': '#555555',
    'ytick.direction': 'in',
    'xtick.labelsize': 'medium',
    'ytick.labelsize': 'medium',

    # Legend
    'legend.fancybox': True,
    'legend.loc': 'best',

    # Figure
    'figure.figsize': [8, 6],
    'figure.facecolor': '1.0',
    'figure.subplot.hspace': 0.5,

    # Other
    'savefig.dpi': 300,
}
