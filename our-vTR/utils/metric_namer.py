from typing import Any, Dict

def change_keys(d: Dict[str, Any], add, sub):
    for key in list(d.keys()):
        d[add + key[len(sub):]] = d.pop(key)

if __name__ == '__main__':
    d = {"ami": 2, "atumi": 1}
    change_keys(d, 'hi', 'a')
    print(d)