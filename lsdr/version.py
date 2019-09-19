version_info = (0, 1, 2)
# format:
# ('lsdr_major', 'lsdrp_minor', 'lsdr_patch')

def get_version():
    "Returns the version as a human-format string."
    return '%d.%d.%d' % version_info

__version__ = get_version()