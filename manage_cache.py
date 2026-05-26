#!/usr/bin/env python3
"""
STT 服务缓存管理工具

用法:
    python manage_cache.py status      # 查看缓存状态
    python manage_cache.py list        # 列出所有模型
    python manage_cache.py download base  # 下载 base 模型
    python manage_cache.py delete tiny  # 删除 tiny 模型
    python manage_cache.py clear       # 清理所有缓存
"""

import sys
import requests
import json

STT_SERVICE_URL = "http://localhost:8001"


def print_json(data):
    """美化打印 JSON"""
    print(json.dumps(data, indent=2, ensure_ascii=False))


def cmd_status():
    """查看缓存状态"""
    print("=== 缓存状态 ===")
    response = requests.get(f"{STT_SERVICE_URL}/cache/status")
    print_json(response.json())


def cmd_list():
    """列出所有模型"""
    print("=== 可用模型 ===")
    response = requests.get(f"{STT_SERVICE_URL}/cache/models")
    models = response.json()["models"]
    
    for model in models:
        status = "✅ 已下载" if model["downloaded"] else "❌ 未下载"
        print(f"{model['name']:8} | {model['size_mb']:6}MB | {status}")


def cmd_download(model_name):
    """下载模型"""
    print(f"=== 下载模型：{model_name} ===")
    response = requests.post(
        f"{STT_SERVICE_URL}/cache/download",
        json={"model": model_name}
    )
    result = response.json()
    
    if result["success"]:
        print(f"✅ 下载成功！")
        print(f"   模型：{result['model']}")
        print(f"   仓库：{result['repo_id']}")
        print(f"   大小：{result['size_mb']}MB")
        print(f"   路径：{result['cache_path']}")
    else:
        print(f"❌ 下载失败：{result['error']}")
        sys.exit(1)


def cmd_delete(model_name):
    """删除模型"""
    print(f"=== 删除模型：{model_name} ===")
    response = requests.delete(f"{STT_SERVICE_URL}/cache/models/{model_name}")
    result = response.json()
    
    if result["success"]:
        print(f"✅ 删除成功！释放 {result['freed_mb']}MB 空间")
    else:
        print(f"❌ 删除失败：{result['error']}")
        sys.exit(1)


def cmd_clear():
    """清理所有缓存"""
    print("=== 清理所有缓存 ===")
    print("⚠️  警告：这将删除所有已下载的模型！")
    
    # 确认
    confirm = input("确认清理？(y/N): ")
    if confirm.lower() != 'y':
        print("已取消")
        return
    
    response = requests.delete(f"{STT_SERVICE_URL}/cache/clear")
    result = response.json()
    
    if result["success"]:
        print(f"✅ 清理成功！释放 {result['freed_mb']}MB 空间")
    else:
        print(f"❌ 清理失败：{result['error']}")
        sys.exit(1)


def cmd_help():
    """显示帮助"""
    print(__doc__)
    print("可用命令:")
    print("  status              - 查看缓存状态")
    print("  list                - 列出所有模型")
    print("  download <model>    - 下载模型 (tiny/base/small/medium/large)")
    print("  delete <model>      - 删除模型")
    print("  clear               - 清理所有缓存")
    print("  help                - 显示帮助")


def main():
    if len(sys.argv) < 2:
        cmd_help()
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "status":
        cmd_status()
    elif command == "list":
        cmd_list()
    elif command == "download":
        if len(sys.argv) < 3:
            print("错误：需要指定模型名称")
            print("用法：python manage_cache.py download <model>")
            sys.exit(1)
        cmd_download(sys.argv[2])
    elif command == "delete":
        if len(sys.argv) < 3:
            print("错误：需要指定模型名称")
            print("用法：python manage_cache.py delete <model>")
            sys.exit(1)
        cmd_delete(sys.argv[2])
    elif command == "clear":
        cmd_clear()
    elif command == "help":
        cmd_help()
    else:
        print(f"未知命令：{command}")
        cmd_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
