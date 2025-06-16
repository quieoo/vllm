#!/usr/bin/python3


import socket, os
import vllm.rpc_pb2 as rpc

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