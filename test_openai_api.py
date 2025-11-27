#!/usr/bin/env python3
"""
测试 qwen_vl_vllm 模块的 OpenAI API 支持

使用方法：
1. 先启动 vLLM OpenAI 兼容服务器：
   python -m vllm.entrypoints.openai.api_server \
       --model Qwen/Qwen3-VL-2B-Thinking \
       --port 8000

2. 运行此测试脚本：
   python test_openai_api.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """测试导入"""
    print("测试导入模块...")
    try:
        from subjects import qwen_vl_vllm
        print("✓ 成功导入 qwen_vl_vllm 模块")
        return True
    except Exception as e:
        print(f"✗ 导入失败: {e}")
        return False

def test_openai_available():
    """测试 OpenAI 客户端是否可用"""
    print("\n测试 OpenAI 客户端...")
    try:
        from openai import OpenAI
        print("✓ OpenAI 客户端可用")
        return True
    except ImportError:
        print("✗ OpenAI 客户端不可用，请安装: pip install openai")
        return False

def test_vllm_available():
    """测试 vLLM 是否可用"""
    print("\n测试 vLLM...")
    try:
        from vllm import LLM, SamplingParams
        print("✓ vLLM 可用")
        return True
    except ImportError:
        print("✗ vLLM 不可用（仅在使用本地推理时需要）")
        return False

def test_config_loading():
    """测试配置加载"""
    print("\n测试配置加载...")
    from subjects.qwen_vl_vllm import _load_template, _build_sampling_params
    
    # 测试基本模板加载
    model_args = {
        "prompt_template": "Test prompt: {video_text}",
        "template_variables": {"default_var": "test"}
    }
    
    try:
        template = _load_template(model_args)
        result = template.render({"video_text": "hello"})
        assert "hello" in result
        print("✓ 模板加载和渲染成功")
    except Exception as e:
        print(f"✗ 模板加载失败: {e}")
        return False
    
    # 测试采样参数
    try:
        sampling_opts = {"max_tokens": 256, "temperature": 0.5}
        params = _build_sampling_params(sampling_opts)
        assert params["max_tokens"] == 256
        assert params["temperature"] == 0.5
        print("✓ 采样参数构建成功")
    except Exception as e:
        print(f"✗ 采样参数构建失败: {e}")
        return False
    
    return True

def test_openai_connection(api_base="http://localhost:8000/v1"):
    """测试 OpenAI API 连接"""
    print(f"\n测试 OpenAI API 连接 ({api_base})...")
    try:
        from openai import OpenAI
        client = OpenAI(base_url=api_base, api_key="EMPTY")
        
        # 尝试列出模型
        models = client.models.list()
        print(f"✓ 成功连接到 API 服务器")
        print(f"  可用模型: {[m.id for m in models.data]}")
        return True
    except Exception as e:
        print(f"✗ 连接失败: {e}")
        print("  请确保 vLLM 服务器正在运行")
        return False

def main():
    print("=" * 60)
    print("Qwen3-VL vLLM 模块测试")
    print("=" * 60)
    
    results = []
    
    # 运行测试
    results.append(("模块导入", test_imports()))
    results.append(("OpenAI 客户端", test_openai_available()))
    results.append(("vLLM", test_vllm_available()))
    results.append(("配置加载", test_config_loading()))
    results.append(("API 连接", test_openai_connection()))
    
    # 打印总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{name:20s}: {status}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！")
        return 0
    else:
        print("\n⚠️  部分测试失败，请检查上述错误信息")
        return 1

if __name__ == "__main__":
    sys.exit(main())
