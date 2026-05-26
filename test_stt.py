#!/usr/bin/env python3
"""
测试公共 STT 服务
"""

import requests
import tempfile
import os

# 服务地址
SERVER_URL = "http://localhost:8001"

def test_health():
    """测试健康检查"""
    print("测试健康检查...")
    response = requests.get(f"{SERVER_URL}/health")
    print(f"✅ 健康状态：{response.json()}")
    return response.json()

def test_models():
    """测试模型列表"""
    print("\n测试模型列表...")
    response = requests.get(f"{SERVER_URL}/models")
    print(f"✅ 可用模型：{response.json()}")
    return response.json()

def test_transcribe_file(audio_file: str):
    """测试文件转录"""
    print(f"\n测试转录：{audio_file}")
    
    with open(audio_file, "rb") as f:
        response = requests.post(
            f"{SERVER_URL}/transcribe",
            files={"file": f},
            data={"language": "zh"}
        )
    
    result = response.json()
    print(f"✅ 转录结果：{result}")
    return result

def test_with_sample():
    """使用示例音频测试（需要实际音频文件）"""
    # 这里可以放一个测试音频文件的路径
    test_file = "/tmp/test_voice.ogg"
    if os.path.exists(test_file):
        return test_transcribe_file(test_file)
    else:
        print(f"⚠️  测试文件不存在：{test_file}")
        print("提示：可以用 feishu_voice 工具生成一个测试音频")
        return None

if __name__ == "__main__":
    print("=" * 60)
    print("公共 STT 服务测试")
    print("=" * 60)
    
    try:
        test_health()
        test_models()
        test_with_sample()
        
        print("\n" + "=" * 60)
        print("✅ 所有测试完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 测试失败：{e}")
        import traceback
        traceback.print_exc()
