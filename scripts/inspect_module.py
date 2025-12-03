import importlib
import inspect
import sys
from pathlib import Path

# Ensure project root is on sys.path so 'src' package can be imported when running this script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

m = importlib.import_module('src.password_features')
print('module file:', getattr(m, '__file__', None))
print('non-dunder attrs:', sorted([a for a in dir(m) if not a.startswith('__')]))

try:
    src = inspect.getsource(m)
    print('\n--- SOURCE ---\n')
    print(src)
except Exception as e:
    print('Could not get source:', e)

# Print a few specific attributes
for name in ('FEATURE_LIST', 'extract_features', 'build_feature_matrix'):
    print(f"{name}:", getattr(m, name, None))
