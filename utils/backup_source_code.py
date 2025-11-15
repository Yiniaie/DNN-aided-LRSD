import shutil
from pathlib import Path
import os
# def backup_source_code(src_dir, dst_dir):
#     def ignore_patterns(_, names):
#         return {'.idea', '__pycache__', '.ipynb_checkpoints'}
#         # 拷贝源码
#     backup_path = os.path.join(dst_dir, 'code')
#     os.makedirs(backup_path, exist_ok=True)
#     shutil.copytree(src_dir, backup_path, ignore=ignore_patterns, dirs_exist_ok=True)


def backup_source_code(src_dir, dst_dir):
    def ignore_patterns(dirpath, names):
        # 要排除的目录名和扩展名
        exclude_dirs = {'.idea', '__pycache__', '.ipynb_checkpoints', 'results', 'result', 'checkpoints', '.git'}
        exclude_exts = {'.pth', '.log', '.zip'}

        ignored = set()

        for name in names:
            full_path = os.path.join(dirpath, name)
            if os.path.isdir(full_path) and name in exclude_dirs:
                ignored.add(name)
            elif os.path.isfile(full_path) and os.path.splitext(name)[1] in exclude_exts:
                ignored.add(name)
        return ignored

    backup_path = os.path.join(dst_dir, 'code')
    # os.makedirs(backup_path, exist_ok=True)
    # shutil.copytree(src_dir, backup_path, ignore=ignore_patterns, dirs_exist_ok=True)
    shutil.copytree(src_dir, backup_path, ignore=ignore_patterns)