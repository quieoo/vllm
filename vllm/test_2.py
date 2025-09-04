#!/usr/bin/python3


import socket, os
import vllm.rpc_pb2 as rpc
import time
import importlib
import ast
import types
from typing import List
import sys

def parse_modules_from_file(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    try:
        data = ast.literal_eval(text)
    except Exception as e:
        raise ValueError(f"cannot parse file to list: {e}")
    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        raise ValueError("file content must be list of str")
    return data

def to_top_level(mod: str) -> str:
    return mod.split(".", 1)[0]

def import_one(mod: str):
    t0 = time.perf_counter()
    try:
        importlib.import_module(mod)
        dt = (time.perf_counter() - t0) * 1000
        # print(f"[OK]   {mod:<40} {dt:8.2f} ms")
        return True, dt, None
    except Exception as e:
        dt = (time.perf_counter() - t0) * 1000
        # print(f"[FAIL] {mod:<40} {dt:8.2f} ms  ({e})")
        return False, dt, e

def preload_modules(path):
    modules = parse_modules_from_file(path)
    modules = sorted({to_top_level(m) for m in modules})
    start_time=time.perf_counter()
    for mod in modules:
        import_one(mod)
    end_time=time.perf_counter()
    print(f"preload modules time: {end_time-start_time}")

# 封装转储操作
def dump_process(socket_path, images_dir):
    """
    通过 RPC 触发 CRIU 转储操作
    :param socket_path: CRIU 服务端套接字路径
    :param images_dir: 转储镜像存储目录
    :return: 转储是否成功（True/False）
    """
    try:
        # 连接服务端
        s = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        s.connect(socket_path)

        # 构造 DUMP 请求
        req = rpc.criu_req()
        req.type = rpc.DUMP
        # req.opts.leave_running = True  # 转储后原进程继续运行
        req.opts.shell_job=True
        # req.opts.log_level = 4
        req.opts.images_dir_fd = os.open(images_dir, os.O_DIRECTORY)
        # req.opts.network_lock = rpc.SKIP

        # 发送请求
        s.send(req.SerializeToString())

        # 接收响应
        resp = rpc.criu_resp()
        resp.ParseFromString(s.recv(1024))
        # 验证响应
        if resp.type != rpc.DUMP:
            print("转储失败：意外的响应类型")
            return False
        if not resp.success:
            print("转储失败：CRIU 执行错误")
            return False
        print("转储成功！镜像存储于：{}".format(images_dir))
        if resp.dump.restored:
            print("CRIU恢复进程")
        return True
    except Exception as e:
        print("转储异常：{}".format(str(e)))
        return False
    finally:
        s.close()
        if 'req' in locals():  # 增加存在性检查
            os.close(req.opts.images_dir_fd)


def save_dump(callback=None):
    dump_socket=os.environ.get("CRIUDUMP_SOCKET",None)
    dump_model=os.environ.get("CRIUDUMP_MODEL",None)
    if dump_socket is not None and dump_model is not None:
        restored = {}
        if "torch.cuda" in sys.modules:
            cuda_mod = sys.modules["torch.cuda"]
            # 备份常用查询函数
            for name in ("is_available","device_count","get_device_capability",
                        "current_device","synchronize"):
                if hasattr(cuda_mod, name):
                    restored[name] = getattr(cuda_mod, name)
            # 软覆盖为“无 GPU”的安全实现
            cuda_mod.is_available = lambda: False
            cuda_mod.device_count = lambda: 0
            cuda_mod.get_device_capability = lambda *a, **k: (0, 0)
            cuda_mod.current_device = lambda: 0
            cuda_mod.synchronize = lambda *a, **k: None
        else:
            # 如果还没加载过 torch.cuda，则注入一个最小“假模块”
            fake = types.ModuleType("torch.cuda")
            fake.is_available = lambda: False
            fake.device_count = lambda: 0
            fake.get_device_capability = lambda *a, **k: (0, 0)
            fake.current_device = lambda: 0
            fake.synchronize = lambda *a, **k: None
            sys.modules["torch.cuda"] = fake
            cuda_mod = fake

        try:
            if callback is not None:
                ret = callback()
        finally:
            # 4) 恢复 torch.cuda
            if restored:
                for k, v in restored.items():
                    setattr(sys.modules["torch.cuda"], k, v)
            else:
                # 如果我们注入了假模块且之前不存在真实模块，就把它移除
                if cuda_mod is sys.modules.get("torch.cuda"):
                    del sys.modules["torch.cuda"]
        # preload necessary modules    
        # preload_modules("/mnt/n0/sslm/ServerlessLLM/tools/CRIU/criu_rpc/vllm_need_libs.txt")
        print(f"Dump socket: {dump_socket}, dump model: {dump_model}")
        dump_process(dump_socket, dump_model+"/imgs")

        return ret
    else:
        if callback is not None:
            return callback()