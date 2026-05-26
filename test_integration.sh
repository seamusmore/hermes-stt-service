#!/bin/bash
# 公共 STT 服务集成测试脚本
# 用法：/mnt/stt-service/test_integration.sh

set -e

echo "============================================================"
echo "        公共 STT 服务切换测试报告"
echo "============================================================"
echo "测试时间：$(date '+%Y-%m-%d %H:%M:%S')"
echo "测试环境：Hermes Agent"
echo "STT 服务：/mnt/stt-service (port 8001)"
echo ""

# 测试 1: 健康检查
echo "------------------------------------------------------------"
echo "【测试 1】服务健康检查"
echo "------------------------------------------------------------"
curl -s http://localhost:8001/health | python3 -m json.tool
echo ""

# 测试 2: 模型列表
echo "------------------------------------------------------------"
echo "【测试 2】可用模型列表"
echo "------------------------------------------------------------"
curl -s http://localhost:8001/models | python3 -m json.tool
echo ""

# 测试 3: stt_service_tool 工具测试
echo "------------------------------------------------------------"
echo "【测试 3】stt_service_tool 工具测试"
echo "------------------------------------------------------------"
cd /home/admin/.hermes/hermes-agent && source venv/bin/activate && python -c "
from tools.stt_service_tool import check_stt_service, stt_transcribe
import os

status = check_stt_service()
print(f'服务状态：{status}')

test_file = '/tmp/test_voice.ogg'
if os.path.exists(test_file):
    result = stt_transcribe(audio_path=test_file, language='zh')
    print(f'转录成功：{result.get(\"success\")}')
    print(f'识别文本：{result.get(\"text\")}')
    print(f'处理耗时：{result.get(\"processing_time_ms\", 0)}ms')
else:
    print('⚠️  测试文件不存在，跳过转录测试')
"
echo ""

# 测试 4: transcription_tools 网关集成测试
echo "------------------------------------------------------------"
echo "【测试 4】transcription_tools 网关集成测试"
echo "------------------------------------------------------------"
cd /home/admin/.hermes/hermes-agent && source venv/bin/activate && python -c "
from tools.transcription_tools import transcribe_audio
import os

test_file = '/tmp/test_voice.ogg'
if os.path.exists(test_file):
    result = transcribe_audio(test_file)
    print(f'转录成功：{result.get(\"success\")}')
    print(f'识别文本：{result.get(\"transcript\")}')
    print(f'提供者：{result.get(\"provider\")}')
else:
    print('⚠️  测试文件不存在，跳过转录测试')
"
echo ""

echo "============================================================"
echo "                    测试结果汇总"
echo "============================================================"
echo ""
echo "✅ 所有测试完成！"
echo ""
