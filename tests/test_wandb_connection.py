#!/usr/bin/env python3
"""
测试wandb连接和上传能力的诊断脚本
"""

import time
import os
import sys

def test_wandb_basic():
    """测试基本的wandb导入和初始化"""
    try:
        import wandb
        print("✓ wandb 导入成功")
        return True
    except ImportError as e:
        print(f"✗ wandb 导入失败: {e}")
        return False

def test_wandb_online_init():
    """测试wandb在线初始化"""
    try:
        import wandb
        print("\n=== 测试wandb在线初始化 ===")
        
        # 尝试在线初始化
        run = wandb.init(
            project="wandb-test", 
            name="connection_test",
            config={"test": True},
            settings=wandb.Settings(
                _disable_stats=True,
                _disable_meta=True,
            )
        )
        
        print("✓ wandb 在线初始化成功")
        print(f"  - Run ID: {run.id}")
        print(f"  - Project: {run.project}")
        print(f"  - URL: {run.get_url()}")
        
        # 测试简单的数据上传
        for i in range(5):
            wandb.log({"test_metric": i * 0.1}, step=i)
            print(f"  - 已记录步骤 {i}")
            time.sleep(0.5)  # 小延迟
        
        wandb.finish()
        print("✓ wandb 数据上传成功")
        return True
        
    except Exception as e:
        print(f"✗ wandb 在线模式失败: {e}")
        print(f"  错误类型: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

def test_wandb_offline_init():
    """测试wandb离线初始化"""
    try:
        import wandb
        print("\n=== 测试wandb离线模式 ===")
        
        # 设置离线模式
        os.environ["WANDB_MODE"] = "offline"
        
        # 尝试离线初始化
        run = wandb.init(
            project="wandb-test-offline", 
            name="offline_test",
            config={"test": True, "offline": True},
            settings=wandb.Settings(
                _disable_stats=True,
                _disable_meta=True,
            )
        )
        
        print("✓ wandb 离线初始化成功")
        print(f"  - Run ID: {run.id}")
        print(f"  - Project: {run.project}")
        
        # 测试离线数据记录
        for i in range(5):
            wandb.log({"offline_metric": i * 0.2}, step=i)
            print(f"  - 已记录离线步骤 {i}")
        
        wandb.finish()
        print("✓ wandb 离线数据记录成功")
        print("  - 数据保存到本地，使用 'wandb sync' 上传")
        return True
        
    except Exception as e:
        print(f"✗ wandb 离线模式失败: {e}")
        print(f"  错误类型: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

def test_network_conditions():
    """测试网络条件"""
    print("\n=== 测试网络环境 ===")
    
    # 测试基本网络连接
    try:
        import requests
        response = requests.get("https://api.wandb.ai/", timeout=10)
        print(f"✓ wandb API连接成功 (状态码: {response.status_code})")
    except Exception as e:
        print(f"✗ wandb API连接失败: {e}")
    
    # 测试DNS解析
    try:
        import socket
        ip = socket.gethostbyname("api.wandb.ai")
        print(f"✓ DNS解析成功: api.wandb.ai -> {ip}")
    except Exception as e:
        print(f"✗ DNS解析失败: {e}")
    
    # 检查代理设置
    proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']
    proxy_found = False
    for var in proxy_vars:
        if var in os.environ:
            print(f"⚠ 检测到代理设置: {var}={os.environ[var]}")
            proxy_found = True
    
    if not proxy_found:
        print("ℹ 未检测到代理设置")

def test_system_resources():
    """检查系统资源"""
    print("\n=== 检查系统资源 ===")
    
    try:
        import psutil
        
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=1)
        print(f"CPU使用率: {cpu_percent}%")
        
        # 内存使用率
        memory = psutil.virtual_memory()
        print(f"内存使用率: {memory.percent}%")
        
        # 网络状态
        net_io = psutil.net_io_counters()
        print(f"网络发送: {net_io.bytes_sent / (1024*1024):.2f} MB")
        print(f"网络接收: {net_io.bytes_recv / (1024*1024):.2f} MB")
        
    except ImportError:
        print("psutil未安装，无法检查系统资源")
    except Exception as e:
        print(f"系统资源检查失败: {e}")

def main():
    print("🔍 wandb连接诊断工具")
    print("=" * 50)
    
    # 基本测试
    if not test_wandb_basic():
        print("\n❌ wandb未正确安装，请运行: pip install wandb")
        return
    
    # 网络环境测试
    test_network_conditions()
    
    # 系统资源测试
    test_system_resources()
    
    # wandb在线测试
    online_success = test_wandb_online_init()
    
    # wandb离线测试
    offline_success = test_wandb_offline_init()
    
    # 总结
    print("\n" + "=" * 50)
    print("📋 诊断结果总结:")
    print("=" * 50)
    
    if online_success:
        print("✅ wandb在线模式工作正常")
        print("   建议: 可以使用默认的在线模式")
    else:
        print("❌ wandb在线模式存在问题")
        if offline_success:
            print("✅ wandb离线模式工作正常")
            print("   建议: 使用 --wandb_offline=True 参数")
            print("   训练后使用 'wandb sync' 上传数据")
        else:
            print("❌ wandb离线模式也存在问题")
            print("   建议: 使用 --use_wandb=False 禁用wandb")
    
    print("\n🔧 解决方案:")
    if not online_success:
        print("1. 检查网络连接和防火墙设置")
        print("2. 如果在公司网络，联系网络管理员")
        print("3. 尝试使用离线模式: --wandb_offline=True")
        print("4. 或完全禁用: --use_wandb=False")

if __name__ == "__main__":
    main() 