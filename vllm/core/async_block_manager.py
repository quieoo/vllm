from sllm_store.client import SllmStoreClient
from vllm.backgroud_logger import logger as bg_logger
import time
import os

class AsyncBlockManager:
    def __init__(self, block_size, model_name, device_id):
        # the block size has already multiplied by the number of layers
        self.block_size = block_size    # 2 *  block_size(num of slots in one block) * num_kv_heads * head_size * sizeof(cache_t)
        self.model_path=os.path.join(model_name, "rank_0")
        self.device_id = device_id
        self.store = SllmStoreClient("127.0.0.1:8073")
        self.total_blocks = 0
        self.global_block_table= {}
        self.free_blocks_list=[]
        print(f"[AsyncBlockManager] Init with block size: {self.block_size}, model path: {self.model_path}, device id: {self.device_id}")
        # 检查block_size是否能够对齐到16字节
        if self.block_size % 16 != 0:
            print(f"[AsyncBlockManager] Block size {self.block_size} is not aligned to 16 bytes")
        bg_logger.info(f"[AsyncBlockManager] Init with block size: {self.block_size}, model path: {self.model_path}, device id: {self.device_id}")
        
    def load_available_block(self):
        self.total_blocks=self.store.get_available_blocks_on_gpu(self.model_path, self.block_size, self.device_id)
        bg_logger.info(f"[AsyncBlockManager] Total blocks: {self.total_blocks}")
        return self.total_blocks
    def get_available_blocks(self):
        return self.total_blocks
    
    def allocate_blocks(self, num_blocks):
        t1=time.time()
        allocated_blocks = self.store.allocate_blocks_on_gpu(self.device_id, self.block_size, self.model_path, num_blocks)
        t2=time.time()
        bg_logger.info(f"[AsyncBlockManager] Need {num_blocks} blocks and Allocated {len(allocated_blocks)}blocks in {(t2-t1)*1000:.4f} ms")

        return allocated_blocks
    
    def check_allocate_blocks(self, blocks):
        new_block_cnt=0
        block_mapping={}
        for block in blocks:
            if block not in self.global_block_table:
                new_block_cnt+=1
            else:
                block_mapping[block]=self.global_block_table[block]
        allocated_blocks=[]
        if new_block_cnt>0:
            allocated_blocks=self.allocate_blocks(new_block_cnt)
            used_block_idx=0
            for block in blocks:
                if block not in self.global_block_table:
                    self.global_block_table[block]=allocated_blocks[used_block_idx]
                    block_mapping[block]=self.global_block_table[block]
                    used_block_idx+=1
        return block_mapping

    def check_allocate_blocks_v1(self, blocks):
        block_mapping = {}
        need_allocate_blocks = []
        
        # 首先检查已有的映射，收集需要分配的block_id
        for block in blocks:
            if block in self.global_block_table:
                block_mapping[block] = self.global_block_table[block]
            else:
                need_allocate_blocks.append(block)
        
        if not need_allocate_blocks:
            return block_mapping
        
        # 尝试从free_blocks_list中分配
        num_need = len(need_allocate_blocks)
        num_free = len(self.free_blocks_list)
        num_from_free = min(num_need, num_free)
        
        # 从free_blocks_list中取前num_from_free个
        free_allocated = self.free_blocks_list[:num_from_free]
        # 剩余的free_blocks_list
        self.free_blocks_list = self.free_blocks_list[num_from_free:]
        
        # 分配free_allocated的块
        for i in range(num_from_free):
            block_id = need_allocate_blocks[i]
            self.global_block_table[block_id] = free_allocated[i]
            block_mapping[block_id] = free_allocated[i]
        
        # 计算还需要分配的数量
        remaining = num_need - num_from_free
        if remaining > 0:
            # 调用allocate_blocks分配剩余的
            new_allocated = self.allocate_blocks(remaining)
            for i in range(remaining):
                block_id = need_allocate_blocks[num_from_free + i]
                self.global_block_table[block_id] = new_allocated[i]
                block_mapping[block_id] = new_allocated[i]
            
            bg_logger.info(f"[AsyncBlockManager] Allocated {remaining} new blocks from ReuseStore")
        
        if num_from_free > 0:
            bg_logger.info(f"[AsyncBlockManager] Allocated {num_from_free} blocks from free_blocks_list")
        
        return block_mapping

    
    def free_blocks(self, logical_block_ids):
        # 将物理地址加入free_blocks_list
        for block in logical_block_ids:
            if block in self.global_block_table:
                self.free_blocks_list.append(self.global_block_table[block])
                # 从映射表中删除
                del self.global_block_table[block]

        
        