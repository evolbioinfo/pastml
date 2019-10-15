import colorsys

NUM2COLOURS = {
    1: ['#1b9e77'],
    2: ['#d95f02', '#1b9e77'],
    3: ['#a6cee3', '#1f78b4', '#b2df8a'],
    4: ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3'],
    5: ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00'],
    6: ['#7fc97f', '#beaed4', '#fdc086', '#ffff99', '#386cb0', '#f0027f'],
    7: ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f'],
    8: ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00'],
    9: ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6'],
    10: ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a'],
    11: ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a',
         '#ffff99'],
    12: ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a',
         '#ffff99', '#b15928'],
    13: ['#cc9566', '#ccc466', '#a4cc66', '#75cc66', '#66cc85', '#66ccb4', '#66b4cc',
         '#6685cc', '#7566cc', '#a466cc', '#cc66c4', '#cc6695', '#cc6666']
}

WHITE = '#ffffff'


def get_enough_colours(num_unique_values):
    """
    Generates and returns an array of `num_unique_values` HEX colours.
    :param num_unique_values: int, number of colours to be generated.
    :return: array of str, containing colours in HEX format.
    """
    if num_unique_values in NUM2COLOURS:
        return NUM2COLOURS[num_unique_values]
    vs = ['#%02x%02x%02x' % tuple(rgb) for rgb in
          (map(lambda x: int(x * 255), colorsys.hsv_to_rgb(*hsv)) for hsv in
           ((_ / num_unique_values, 0.25 * (1 + (_ % 3)), .8) for _ in range(1, num_unique_values + 1)))]
    if num_unique_values < 20:
        return vs[::5] + vs[1::5] + vs[2::5] + vs[3::5] + vs[4::5]
    return vs[::10] + vs[1::10] + vs[2::10] + vs[3::10] + vs[4::10] \
           + vs[5::10] + vs[6::10] + vs[7::10] + vs[8::10] + vs[9::10]


def hex_to_rgb(value):
    value = value.lstrip('#')
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))
