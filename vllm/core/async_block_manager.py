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
        
        
        