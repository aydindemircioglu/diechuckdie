"""BLA."""
import re

__version__ = '1.1.0.dev0'
__version_info__ = tuple(
    re.match(r'(\d+\.\d+\.\d+).*', __version__).group(1).split('.'))
