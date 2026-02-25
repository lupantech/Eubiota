from typing import List, Tuple, Dict
import re
import os
import shutil
import torch
import argparse
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForTokenClassification, AutoModelForVision2Seq
# Use ThreadPoolExecutor for I/O within a process, and ProcessPoolExecutor for parallelizing across directories
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from torch.distributed._tensor import DTensor, Shard, Placement


def merge_by_placement(tensors: List[torch.Tensor], placement: Placement):
    """
    Merges a list of tensors based on their placement specification.
    """
    if placement.is_replicate():
        return tensors[0]
    elif placement.is_partial():
        raise NotImplementedError("Partial placement is not supported yet")
    elif placement.is_shard():
        return torch.cat(tensors, dim=placement.dim).contiguous()
    else:
        raise ValueError(f"Unsupported placement: {placement}")

def merge_checkpoint_directory(local_dir: str, hf_upload_path: str = None):
    try:
        print(f"[{os.getpid()}] Starting processing for: {local_dir}")

        # Detect world_size
        world_size = None
        for f in os.listdir(local_dir):
            match = re.match(r"model_world_size_(\d+)_rank_0\.pt", f)
            if match:
                world_size = int(match.group(1))
                break
        if not world_size:
            print(f"[{os.getpid()}] No sharded files found.")
            return

        # Parallel load all shards
        def load_rank(rank):
            path = os.path.join(local_dir, f'model_world_size_{world_size}_rank_{rank}.pt')
            return torch.load(path, map_location='cpu', weights_only=True)

        with ThreadPoolExecutor(max_workers=min(64, os.cpu_count())) as executor:
            model_state_dict_lst = list(executor.map(load_rank, range(world_size)))

        # Extract device_mesh from first DTensor
        pivot_key = next(k for k, v in model_state_dict_lst[0].items() if isinstance(v, DTensor))
        device_mesh = model_state_dict_lst[0][pivot_key].device_mesh
        assert device_mesh.mesh_dim_names == ('fsdp',), "Only FSDP supported"

        total_shards = device_mesh.mesh.size(0)
        assert total_shards == world_size

        merged_state_dict = {}
        param_placements = {}

        for key in model_state_dict_lst[0]:
            tensors = []
            placements = None
            is_dtensor = False
            for rank_sd in model_state_dict_lst:
                tensor = rank_sd[key]
                if isinstance(tensor, DTensor):
                    is_dtensor = True
                    tensors.append(tensor._local_tensor)
                    if placements is None:
                        placements = tensor.placements
                else:
                    tensors.append(tensor)
            if is_dtensor:
                merged = merge_by_placement(tensors, placements[0])  # merge
                merged_state_dict[key] = merged.to(torch.bfloat16)    # unified conversion to bfloat16
            else:
                merged_state_dict[key] = tensors[0].bfloat16()

        hf_path = os.path.join(local_dir, 'huggingface')
        os.makedirs(hf_path, exist_ok=True)

        from transformers.utils import WEIGHTS_NAME
        save_path = os.path.join(hf_path, WEIGHTS_NAME)
        torch.save(merged_state_dict, save_path)

        config_src = os.path.join(local_dir, "config.json")
        if os.path.exists(config_src):
            shutil.copy(config_src, os.path.join(hf_path, "config.json"))

        print(f"[{os.getpid()}] Merged model saved to {hf_path}")

        if hf_upload_path:
            from huggingface_hub import HfApi
            api = HfApi()
            api.create_repo(repo_id=hf_upload_path, private=False, exist_ok=True)
            api.upload_folder(folder_path=hf_path, repo_id=hf_upload_path, repo_type="model")

    except Exception as e:
        import traceback
        print(f"[{os.getpid()}] Error: {e}")
        traceback.print_exc()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Merge sharded FSDP model checkpoints in parallel.")
    parser.add_argument(
        '--local_dirs', 
        required=True, 
        type=str, 
        nargs='+', 
        help="A list of paths for your saved model directories."
    )
    parser.add_argument(
        "--hf_upload_paths", 
        default=None, 
        type=str, 
        nargs='+', 
        help="A list of Hugging Face repo paths to upload to. Must match the number of local_dirs."
    )
    args = parser.parse_args()

    upload_paths = args.hf_upload_paths
    if upload_paths:
        if len(upload_paths) != len(args.local_dirs):
            raise ValueError("The number of --hf_upload_paths must match the number of --local_dirs.")
    else:
        upload_paths = [None] * len(args.local_dirs)
        
    max_procs = min(16, os.cpu_count() or 4)
    
    # Create a list of argument tuples for each task
    process_args = list(zip(args.local_dirs, upload_paths))

    print(f"Found {len(process_args)} directories to process. Starting parallel execution with up to {max_procs} processes.")
    
    with ProcessPoolExecutor(max_workers=max_procs) as executor:
        # Loop through the arguments and submit each task to the executor
        # The *args syntax unpacks the tuple into separate arguments for the function
        for args_tuple in process_args:
            executor.submit(merge_checkpoint_directory, *args_tuple)

    print("All processing jobs are complete.")

"""
checkpoint file structure:
```file
checkpoints/
└── Eubiota/
    └── EXP_NAME/
        └── global_step_2/
            └── actor/
                ├── huggingface/                 ← HF config/tokenizer
                │   ├── config.json
                │   ├── tokenizer.json
                │   └── ...
                ├── model_world_size_8_rank_0.pt ← SHARD WEIGHT (FSDP)
                ├── model_world_size_8_rank_1.pt
                └── ...
                └── extra_state_world_size_8_rank_*.pt ← OTHER STATES, E.G. OPTIM & EXTRA STATES         
```

execution code example:
python util/model_merger.py --local_dirs $(find checkpoints/... -type d -name "actor")
python util/model_merger.py --local_dirs checkpoints/.../actor
"""
