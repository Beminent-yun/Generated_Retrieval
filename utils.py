import os
import re
import sys
import requests
from pathlib import Path
from urllib.parse import unquote
from typing import Callable, Iterable, Iterator, Optional, TypeVar
import zipfile
import tarfile
from rich.progress import track



def download_file(url, save_dir='./datasets', filename=None, timeout=30):
    """
    单一职责：从 URL 下载文件到本地，支持断点跳过与进度显示。

    :param url      : 文件下载链接
    :param save_dir : 本地保存目录，不存在则自动创建
    :param filename : 指定保存文件名；为 None 时依次从 Content-Disposition、URL 路径中提取
    :param timeout  : 连接与读取超时（秒），默认 30s；设为 None 则不限时（有挂死风险）
    :return         : 下载后的本地文件绝对路径，失败则返回 None
    """
    os.makedirs(save_dir, exist_ok=True)

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    save_path = None  # 提前声明，供 except 块中的清理逻辑使用
    try:
        # timeout=(connect_timeout, read_timeout)，防止服务器无响应时永久挂起
        with requests.get(url, stream=True, headers=headers, timeout=timeout) as response:
            response.raise_for_status()

            # --- 文件名提取（优先级：参数指定 > Content-Disposition > URL 路径）---
            if not filename:
                content_disposition = response.headers.get('content-disposition', '')
                if content_disposition:
                    fnames = re.findall('filename="?([^";]+)"?', content_disposition)
                    if fnames:
                        # strip() 去除 Content-Disposition 中可能附带的空格或 \r
                        filename = fnames[0].strip()
                if not filename:
                    filename = os.path.basename(url).split('?')[0]
                filename = unquote(filename).strip()

            if not filename:
                filename = 'unknown.dat'

            save_path = os.path.join(save_dir, filename)
            total_size = int(response.headers.get('content-length', 0))

            # 跳过逻辑：文件已存在且大小与 Content-Length 一致则无需重下
            # 注意：服务器不返回 Content-Length（total_size==0）时跳过此检查，强制重下
            if os.path.exists(save_path) and total_size > 0 and os.path.getsize(save_path) == total_size:
                print(f"⏩ 文件已存在，跳过: {filename}")
                return save_path

            print(f"⬇ 下载 {filename} ({total_size / 1024 / 1024:.2f} MB)")
            with open(save_path, 'wb') as f:
                for chunk in track(
                    response.iter_content(chunk_size=8192),
                    description=filename,
                    total=total_size,
                    bytes_mode=True,
                    on_advance=lambda p, t, chunk: p.advance(t, len(chunk)),
                ):
                    if chunk:
                        f.write(chunk)

            return save_path

    except Exception as e:
        print(f"✘ 下载失败: {e}")
        # 清理因异常中断产生的残留部分文件，避免下次因大小不符而误判
        if save_path and os.path.exists(save_path):
            os.remove(save_path)
            print(f"✘ 已清理残留文件: {os.path.basename(save_path)}")
        return None
    
def extract_archive(file_path, remove_archive=False, recursive=False):
    """
    单一职责：解压文件 (支持递归)
    :param file_path: 压缩包路径
    :param remove_archive: 解压后是否删除原压缩包 (节省空间)
    :param recursive: 是否递归解压内部的压缩包
    :return: 解压后的目录路径
    """
    if not os.path.exists(file_path):
        return None

    # 1. 定义解压目标目录
    # 规则：解压到与文件名同名的文件夹中
    # ./downloads/data.zip -> ./downloads/data/
    file_dir = os.path.dirname(file_path)
    file_name = os.path.basename(file_path)

    # 智能去掉后缀
    folder_name = file_name
    for ext in ['.tar.gz', '.tar.bz2', '.tar.xz', '.tgz', '.zip', '.tar']:
        if folder_name.endswith(ext):
            folder_name = folder_name.replace(ext, '')
            break

    extract_path = os.path.join(file_dir, folder_name)
    os.makedirs(extract_path, exist_ok=True)

    try:
        is_extracted = False

        # 2. 识别并解压
        if zipfile.is_zipfile(file_path):
            print(f"📦 解压 ZIP: {file_name} → {folder_name}/")
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            is_extracted = True

        elif tarfile.is_tarfile(file_path):
            print(f"📦 解压 TAR: {file_name} → {folder_name}/")
            with tarfile.open(file_path, 'r:*') as tar_ref:
                # filter='data' 防止路径穿越攻击（即 zip slip：压缩包内含 ../../../etc/passwd 等恶意路径）
                # Python 3.12+ 若不设置此参数会发出 DeprecationWarning
                # Python < 3.12 不支持该参数，通过 hasattr 兼容两个版本
                if hasattr(tarfile, 'data_filter'):
                    tar_ref.extractall(extract_path, filter='data')
                else:
                    tar_ref.extractall(extract_path)
            is_extracted = True

        # 3. 后处理：删除原包 & 递归逻辑
        if is_extracted:
            if remove_archive:
                os.remove(file_path) # 删除原始压缩包，释放空间

            if recursive:
                # 遍历解压出来的所有文件
                for root, dirs, files in os.walk(extract_path):
                    for f in files:
                        full_path = os.path.join(root, f)
                        # 递归调用自己！
                        # 注意：这里我们默认递归解压后的内部包也删除，防止空间爆炸
                        extract_archive(full_path, remove_archive=True, recursive=True)

            return extract_path
        else:
            # 不是压缩包，直接返回原路径或None
            return None

    except Exception as e:
        print(f"✘ 解压失败 {file_path}: {e}")
        return None
    
def get_dataset_with_url(url, save_dir='./datasets', recursive_unzip=False):
    """
    高层 API：一键获取数据集
    """
    # Step 1: 下载
    archive_path = download_file(url, save_dir)

    if archive_path:
        # Step 2: 解压 (解耦优势：我可以选择不解压，或者换个地方解压)
        # 这里演示解压
        data_dir = extract_archive(archive_path, remove_archive=False, recursive=recursive_unzip)

        if data_dir:
            print(f"✔ 数据集准备就绪: {data_dir}")
            return data_dir
        else:
            print(f"⚠ 下载完成，但无需解压或解压失败: {archive_path}")
            return archive_path

    return None

