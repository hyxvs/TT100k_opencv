import os
import requests
from tqdm import tqdm

DATA_ZIP_URL = "http://cg.cs.tsinghua.edu.cn/traffic-sign/data_model_code/data.zip"
zip_path = r"E:\TT100K_data.zip"
# 确保目录存在
os.makedirs(os.path.dirname(zip_path), exist_ok=True)

def download_file(url, dst):
    print("Downloading:", url)
    # 发送GET请求，stream=True表示分块下载（不一次性加载到内存）
    with requests.get(url, stream=True, allow_redirects=True) as r:
        # 检查HTTP请求是否成功（状态码200），失败则抛出异常
        r.raise_for_status()
        # 获取文件总大小（从响应头的Content-Length字段），默认值0
        total = int(r.headers.get("content-length", 0))
        # 打开本地文件（wb=二进制写入模式），同时初始化进度条
        with open(dst, "wb") as f, tqdm(
            total=total,       # 进度条总长度（文件总字节数）
            unit="B",          # 进度单位：字节
            unit_scale=True,   # 自动缩放单位（如KB、MB、GB）
            desc="Downloading" # 进度条前缀文字
        ) as pbar:
            # 分块读取响应内容，chunk_size=8192表示每次读取8KB
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:  # 过滤空块（避免写入无效数据）
                    f.write(chunk)  # 将分块写入文件
                    pbar.update(len(chunk))  # 更新进度条（增加已下载的字节数）

    print("Download finished:", dst)

if not os.path.exists(zip_path):
    download_file(DATA_ZIP_URL, zip_path)
else:
    print("File already exists, skip download:")
    print(zip_path)
