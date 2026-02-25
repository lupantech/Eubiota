import random
from contextlib import contextmanager
from copy import deepcopy
from typing import Dict, Tuple

import torch
from omegaconf import OmegaConf
from pprint import pprint
from tqdm import tqdm

from codetiming import Timer
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    AdvantageEstimator,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.utils.metric import reduce_metrics
from verl.utils.tracking import Tracking

from .daemon import AgentModeDaemon

import os
import json
import uuid
from collections import defaultdict

import time

import numpy as np

@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    if name not in timing_raw:
        timing_raw[name] = 0
    timing_raw[name] += timer.last


class AgentFlowTrainer(RayPPOTrainer):
    """
    Specialized PPO trainer for agent-based reinforcement learning.

    This trainer is designed specifically for scenarios where the model interacts with
    external environments, tools, or APIs through an AgentFlowServer. It simplifies
    the training loop by removing the complex conditional logic present in the original
    RayPPOTrainer and focusing on the agent mode workflow.

    Key differences from RayPPOTrainer:
    1. Uses AgentModeDaemon for server communication
    2. Simplified data flow without pop/union operations
    3. Direct batch processing through agent daemon
    4. Streamlined validation using agent_mode validation
    """

    def _validate(self):
        assert len(self.val_dataloader) == 1, "Please set val_batch_size to None for better throughput."

        # no empty check dataloader
        try:
            test_data = next(iter(self.val_dataloader))
        except StopIteration:
            raise ValueError("Validation dataloader is empty. Check your validation dataset.")

        # no empty check key
        print(f"Validation data keys: {test_data.keys()}")
        for key, value in test_data.items():
            if isinstance(value, list):
                print(f"Validation data {key} length: {len(value)}")
                if len(value) == 0:
                    print(f"Warning: Empty data in {key}")
            elif isinstance(value, torch.Tensor):
                print(f"Validation data {key} shape: {value.shape}")
                if value.numel() == 0:
                    print(f"Warning: Empty tensor in {key}")
            else:
                print(f"Validation data {key} type: {type(value)}")

        # no empty check
        if not test_data or all((isinstance(v, list) and len(v) == 0) or (isinstance(v, torch.Tensor) and v.numel() == 0) for v in test_data.values()):
            raise ValueError("Validation data is empty. Check your validation dataset.")

        test_batch = DataProto.from_single_dict(test_data)
        # test_batch.non_tensor_batch["step"] = np.ones_like(test_batch.non_tensor_batch["question"]) * self.global_steps
        self.async_rollout_manager.wake_up()
        self.agent_mode_daemon.set_up_data_and_server(
            test_batch.non_tensor_batch,
            self.async_rollout_manager.server_addresses,
            is_train=False,
        )

        # whether persisting queueing 
        if self.agent_mode_daemon._total_tasks_queued == 0:
            raise ValueError("No validation tasks were queued. Check data preparation.")

        self.agent_mode_daemon.run_until_all_finished()

        # Check if we have any completed rollouts, with more detailed error reporting
        completed_count = len(self.agent_mode_daemon._completed_rollouts)
        valid_count = len([r for r in self.agent_mode_daemon._completed_rollouts.values()
                          if r.triplets and len(r.triplets) > 0])
        original_count = self.agent_mode_daemon._total_tasks_queued

        completion_rate = completed_count / original_count if original_count > 0 else 0
        print(f"Validation summary: {completed_count}/{original_count} total rollouts ({completion_rate:.1%}), {valid_count} valid rollouts")

        # More lenient validation acceptance
        if completed_count == 0:
            raise ValueError("No validation tasks completed. Check server and agent execution.")

        # Accept partial results if we have some reasonable completion
        min_acceptable_rate = 0.1  # Accept if at least 10% completed
        if completion_rate < min_acceptable_rate:
            raise ValueError(f"Insufficient validation completion: {completion_rate:.1%} < {min_acceptable_rate:.1%}. "
                           f"Only {completed_count}/{original_count} tasks completed.")

        if valid_count == 0:
            print("Warning: No valid validation rollouts (all have empty triplets), using fallback metrics")
        else:
            print(f"Validation proceeding with {valid_count} valid rollouts ({valid_count/completed_count:.1%} of completed)")

        test_metrics = self.agent_mode_daemon.get_test_metrics()

        self.agent_mode_daemon.clear_data_and_server()
        self.async_rollout_manager.sleep()
        return test_metrics

    def _train_step(self, batch_dict: dict) -> dict:
        # Isolate in a separate method to automatically recycle the variables before validation.
        batch: DataProto = DataProto.from_single_dict(batch_dict)
        metrics = {}
        timing_raw = {}

        # data key check & no empty check
        print(f"Training data keys: {batch_dict.keys()}")
        for key, value in batch_dict.items():
            if isinstance(value, list):
                print(f"Training data {key} length: {len(value)}")
                if len(value) == 0:
                    print(f"Warning: Empty data in {key}")
            elif isinstance(value, torch.Tensor):
                print(f"Training data {key} shape: {value.shape}")
                if value.numel() == 0:
                    print(f"Warning: Empty tensor in {key}")
            else:
                print(f"Training data {key} type: {type(value)}")

        # ensure no empty
        if not batch_dict or all((isinstance(v, list) and len(v) == 0) or (isinstance(v, torch.Tensor) and v.numel() == 0) for v in batch_dict.values()):
            raise ValueError("Training data is empty. Check your training dataset.")

        with _timer("step", timing_raw):
            # When agent mode is enabled, we read the batch as it is.
            gen_batch = batch

            # generate a batch
            with _timer("gen", timing_raw):
                # gen_batch.non_tensor_batch["step"] = np.ones_like(gen_batch.non_tensor_batch["question"]) * self.global_steps
                self.async_rollout_manager.wake_up()
                self.agent_mode_daemon.set_up_data_and_server(
                    gen_batch.non_tensor_batch, self.async_rollout_manager.server_addresses
                )

                if self.agent_mode_daemon._total_tasks_queued == 0:
                    raise ValueError("No training tasks were queued. Check data preparation.")

                self.agent_mode_daemon.run_until_all_finished()

                if len(self.agent_mode_daemon._completed_rollouts) == 0:
                    raise ValueError("No training tasks completed. Check server and agent execution.")

                batch, agent_metrics = self.agent_mode_daemon.get_train_data_batch(
                    max_prompt_length=self.config.data.max_prompt_length,
                    max_response_length=self.config.data.max_response_length,
                    device=gen_batch.batch["fake_ids"].device,
                )
                metrics.update(agent_metrics)
                self.agent_mode_daemon.clear_data_and_server()
                self.async_rollout_manager.sleep()

            if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                with _timer("gen_max", timing_raw):
                    gen_baseline_batch = deepcopy(gen_batch)
                    gen_baseline_batch.meta_info["do_sample"] = False
                    gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                    batch = batch.union(gen_baseline_output)
                    reward_baseline_tensor = self.reward_fn(batch)
                    reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                    batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                    batch.batch["reward_baselines"] = reward_baseline_tensor

                    del gen_baseline_batch, gen_baseline_output

            # uid is used for algorithm like GRPO, should be aligned to data id
            batch.non_tensor_batch["uid"] = batch.non_tensor_batch["data_id_list"]

            batch.batch["response_mask"] = compute_response_mask(batch)

            # compute global_valid tokens
            batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

            with _timer("reward", timing_raw):
                # compute reward model score
                if self.use_rm:
                    reward_tensor = self.rm_wg.compute_rm_score(batch)
                    batch = batch.union(reward_tensor)

                reward_extra_infos_dict = {}

            # for agent mode, pad the lengths to calculate old log prob, ref, and values
            batch, pad_size = pad_dataproto_to_divisor(batch, self.actor_rollout_wg.world_size)

            # recompute old_log_probs
            with _timer("old_log_prob", timing_raw):
                old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                entropys = old_log_prob.batch["entropys"]
                response_masks = batch.batch["response_mask"]
                loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                metrics.update(old_log_prob_metrics)
                old_log_prob.batch.pop("entropys")
                batch = batch.union(old_log_prob)

            if self.use_reference_policy:
                # compute reference log_prob
                with _timer("ref", timing_raw):
                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                    batch = batch.union(ref_log_prob)

            # compute values
            if self.use_critic:
                with _timer("values", timing_raw):
                    values = self.critic_wg.compute_values(batch)
                    batch = batch.union(values)

            # for agent mode, unpad to calculate adv
            # it is important, as adv should be based on the raw traces
            batch = unpad_dataproto(batch, pad_size=pad_size)

            with _timer("adv", timing_raw):
                # if agent_mode is enabled, there is already token_level_scores
                # token_level_scores is not needed to compute here

                # compute rewards. apply_kl_penalty if available
                if self.config.algorithm.use_kl_in_reward:
                    batch, kl_metrics = apply_kl_penalty(
                        batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                    )
                    metrics.update(kl_metrics)
                else:
                    batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                # compute advantages, executed on the driver process

                norm_adv_by_std_in_grpo = self.config.algorithm.get(
                    "norm_adv_by_std_in_grpo", True
                )  # GRPO adv normalization factor

                batch = compute_advantage(
                    batch,
                    adv_estimator=self.config.algorithm.adv_estimator,
                    gamma=self.config.algorithm.gamma,
                    lam=self.config.algorithm.lam,
                    num_repeat=self.config.actor_rollout_ref.rollout.n,
                    norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                    config=self.config.algorithm,
                )

            # after advantages are assinged, we begin to drop (1) long prompt (2) floor to ppo minisize
            keep_indices = (~batch.batch["is_drop_mask"]).nonzero(as_tuple=True)[0]
            metrics["agent_mode/n_dropped_sample_because_of_prompt"] = (
                batch.batch["is_drop_mask"].shape[0] - keep_indices.shape[0]
            )
            batch = batch[keep_indices]
            # next, round to minibatch size
            mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
            n_transition = len(batch)

            random_indices = list(range(n_transition))
            random.shuffle(random_indices)
            batch.reorder(torch.tensor(random_indices).type(torch.int32))
            n_remained_transition = n_transition // mini_batch_size * mini_batch_size
            batch = batch[list(range(n_remained_transition))]
            metrics["agent_mode/n_dropped_sample_because_of_mini_batch"] = n_transition - n_remained_transition

            n_transition = len(batch)
            # make sure divisible by k_partitions for seqlen_balancing
            k_partitions = self.config.trainer.n_gpus_per_node  # usually equal to num_workers or 8
            n_remained_transition = n_transition // k_partitions * k_partitions
            if n_remained_transition != n_transition:
                batch = batch[list(range(n_remained_transition))]
            metrics["agent_mode/n_dropped_sample_because_of_gpu_partitions"] = n_transition - n_remained_transition

            # Agent mode note: Change the order of balance batch;
            #     1. first calculate advantage
            #     2. then drop the samples (too long prompt & floor to ppo minisize)
            #     3. balance
            # balance the number of valid tokens on each dp rank.
            # Note that this breaks the order of data inside the batch.
            # Please take care when you implement group based adv computation such as GRPO and rloo
            if self.config.trainer.balance_batch:
                self._balance_batch(batch, metrics=metrics)

            # update critic
            if self.use_critic:
                with _timer("update_critic", timing_raw):
                    critic_output = self.critic_wg.update_critic(batch)
                critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                metrics.update(critic_output_metrics)

            # implement critic warmup
            if self.config.trainer.critic_warmup <= self.global_steps:
                # update actor
                with _timer("update_actor", timing_raw):
                    batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                    actor_output = self.actor_rollout_wg.update_actor(batch)
                actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                metrics.update(actor_output_metrics)

        # compute training metrics
        metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
        metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

        n_gpus = self.resource_pool_manager.get_n_gpus()
        metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

        return metrics

    def _dump_rollout_data(self, inputs, outputs, scores, reward_extra_infos_dict, metrics, dump_path, is_train, batch, data_ids=None, ground_truths=None):
        data_type = 'train' if is_train else 'val'
        current_time = time.strftime("%Y%m%d_%H%M%S")
        step_dir = os.path.join(dump_path, data_type, f"step_{self.global_steps}_{current_time}")
        os.makedirs(step_dir, exist_ok=True)

        if data_ids is None:
            data_ids = batch.non_tensor_batch.get("data_id_list", [str(uuid.uuid4()) for _ in inputs])
        else:
            data_ids = data_ids[:len(inputs)] + [str(uuid.uuid4()) for _ in range(len(inputs) - len(data_ids))]

        question_groups = defaultdict(list)
        all_metrics = metrics.copy()

        for i, (input_text, output_text, score) in enumerate(zip(inputs, outputs, scores)):
            data_id = data_ids[i] if i < len(data_ids) else str(uuid.uuid4())

            record = {
                "query_index": i,
                "data_id": data_id,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "input": input_text,
                "output": output_text,
                "score": score,
                "metrics": {},
                "extra_info": {}
            }

            if ground_truths and i < len(ground_truths):
                record["ground_truth"] = ground_truths[i]

            for metric_name, metric_value in all_metrics.items():
                if isinstance(metric_value, (list, tuple)) and i < len(metric_value):
                    record["metrics"][metric_name] = metric_value[i]
                else:
                    record["metrics"][f"global_{metric_name}"] = metric_value

            if reward_extra_infos_dict:
                for key, values in reward_extra_infos_dict.items():
                    if i < len(values):
                        record["extra_info"][key] = values[i]

            question_groups[data_id].append(record)

        for data_id, records in question_groups.items():
            json_path = os.path.join(step_dir, f"query_{data_id}.json")

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)

        print(f"Successfully saved rollout data to {step_dir}")

    def fit(self):
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        assert self.async_rollout_mode, "If agent mode is enabled, async server must be enabled"
        self.agent_mode_daemon = AgentModeDaemon(
            self.config.agentflow.port,
            self.config.actor_rollout_ref.rollout.n,
            train_information={
                "model": self.config.actor_rollout_ref.model.path,
                "temperature": self.config.actor_rollout_ref.rollout.temperature,
            },
            tokenizer=self.tokenizer,
            mini_batch_size=self.config.actor_rollout_ref.actor.ppo_mini_batch_size,
            pad_token_id=self.tokenizer.pad_token_id,
            enable_rollout_validation=self.config.agentflow.get("enable_rollout_validation", True),
            max_empty_retries=self.config.agentflow.get("max_empty_retries", 2),
        )
        self.agent_mode_daemon.start()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            print(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                is_last_step = self.global_steps >= self.total_training_steps

                # train step
                metrics = self._train_step(batch_dict)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with _timer("validate", timing_raw):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                ):
                    with _timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()

                # step metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )

                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()

                    # This exit logic is to ensure a robust CI.
                    pprint(f"Flush the logger...")
                    del logger  # Make sure the loggers are flushed and closed properly
                    pprint(f"Training finished at step {self.global_steps}.")
                    return

                progress_bar.update(1)
                self.global_steps += 1