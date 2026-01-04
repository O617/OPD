# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from concurrent.futures import Future
from contextlib import nullcontext
from datetime import datetime
from types import SimpleNamespace
from codetiming import Timer

import torch
import zmq
import io
import os
import time
import queue
import threading

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager

teacher_topk_logps_padded, teacher_topk_indices_padded = None, None
DEBUG = False

def chunk_list(lst, n_chunks):
    """Split a list into chunks of equal length"""
    size = len(lst) // n_chunks
    for i, start in enumerate(range(0, len(lst), size)):
        if i == n_chunks - 1:
            yield lst[start:]
            return
        else:
            yield lst[start : start + size]


def serialize(data):
    buffer = io.BytesIO()
    torch.save(data, buffer)
    return buffer.getbuffer()


def deserialize(message):
    buffer = io.BytesIO(message)
    return torch.load(buffer)


def check_if_invalid(topk_logps, inputs):
    is_valid = True
    reason = ""
    for x in topk_logps:
        if x.isnan().any():
            is_valid = False
            reason = "nan"
            break
        elif x.isinf().any():
            is_valid = False
            reason = "inf"
            break
        elif (x == 0).any():
            is_valid = False
            reason = "zero"
            break
    if not is_valid:
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.tolist()
        with open("teacher_debug.log", "a") as f:
            f.write("{}\n".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            f.write(f"{reason}\n")
            f.write(f"{str(inputs)}\n")


class TeacherClient:
    def __init__(
        self,
        server_ip,
        server_port,
        num_microbatches=1,
        max_tokens=1,
        n_server_workers=1,
        temperature=1,
        only_response=False,
        max_seq_len=None,
    ) -> None:
        self.server_ip = server_ip
        self.server_port = server_port
        self.num_microbatches = num_microbatches
        self.n_server_workers = n_server_workers
        self.max_tokens = max_tokens
        self.task_queue = queue.Queue()
        self.mutex = threading.Lock() if n_server_workers > 1 else nullcontext()
        self.context = zmq.Context()
        self.temperature = temperature
        self.only_response = only_response
        self.max_seq_len = max_seq_len
        self._run()

    def bg_task(self):
        socket = self.context.socket(zmq.REQ)
        socket.connect(f"tcp://{self.server_ip}:{self.server_port}")
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.RCVTIMEO, 600000)  # 接收超时 30 分钟

        while True:
            futures = []
            inputs = []
            batch = []
            try:
                with self.mutex:
                    for _ in range(self.num_microbatches):
                        future, data = self.task_queue.get()
                        if DEBUG:
                            inputs.append(data)
                        futures.append(future)
                        batch.extend(data.tolist() if isinstance(data, torch.Tensor) else data)

                if self.max_seq_len:
                    max_tokens = [min(self.max_tokens, self.max_seq_len - len(prompt)) for prompt in batch]
                    request = {"prompt_token_ids": batch, "max_tokens": max_tokens}
                else:
                    request = {"prompt_token_ids": batch, "max_tokens": self.max_tokens}
                if self.temperature:
                    request["temperature"] = self.temperature
                if self.only_response:
                    request["only_response"] = True

                socket.send(serialize(request))
                raw = socket.recv()
                response = deserialize(raw)

                if isinstance(response, dict) and response.get("status") == "error":
                    reason = response.get("reason", "unknown")
                    err = RuntimeError(f"Teacher error: {reason}")
                    for f in futures:
                        f.set_exception(err)
                    continue

                required = ("responses", "teacher_topk_logprobs", "teacher_topk_indices")
                for k in required:
                    if k not in response:
                        raise RuntimeError(f"Invalid response: missing key '{k}'")

                total = len(response["teacher_topk_logprobs"])
                if self.num_microbatches <= 0 or total % self.num_microbatches != 0:
                    raise RuntimeError(f"Size mismatch: total={total}, num_microbatches={self.num_microbatches}")

                mbs = total // self.num_microbatches
                for i, future in enumerate(futures):
                    s, e = i * mbs, (i + 1) * mbs
                    responses = response["responses"][s:e]
                    teacher_topk_logps = response["teacher_topk_logprobs"][s:e]
                    if DEBUG:
                        check_if_invalid(teacher_topk_logps, inputs[i])
                    teacher_topk_indices = response["teacher_topk_indices"][s:e]
                    future.set_result((responses, teacher_topk_logps, teacher_topk_indices))

            except zmq.Again:
                err = TimeoutError(f"Timeout waiting for server {self.server_ip}:{self.server_port}")
                for f in futures:
                    f.set_exception(err)
                continue
            except Exception as e:
                for f in futures:
                    try:
                        f.set_exception(e)
                    except Exception:
                        pass
                continue

    def _run(self):
        for _ in range(self.n_server_workers):
            threading.Thread(target=self.bg_task, daemon=True).start()

    def submit(self, data):
        future = Future()
        self.task_queue.put((future, data))
        return future

    def __del__(self):
        self.context.destroy()

    def get_teacher_knowledge(self, batch: DataProto, is_async=False):
        """
        Retrieve teacher model's top-k predictions and log probabilities for knowledge distillation.

        Args:
            batch (DataProto): Input batch containing input_ids and attention_mask
            is_async (bool): Whether to use asynchronous processing

        Returns:
            If is_async=True: SimpleNamespace with get() method to process futures
            If is_async=False: Processed DataProto containing teacher knowledge

        Raises:
            RuntimeError: If teacher model request fails
        """

        input_ids = []
        attention_mask = batch.batch["attention_mask"].to(torch.bool)
        # response_length = batch.meta_info["response_length"]

        for ids, mask in zip(batch.batch["input_ids"], attention_mask, strict=False):
            input_ids.append(ids[mask].tolist())

        all_teacher_topk_logps = []
        all_teacher_topk_indices = []
        responses = []

        batch_size = len(input_ids)
        assert batch_size % self.n_server_workers == 0
        micro_batch_size = batch_size // self.n_server_workers
        futures = []
        tik1 = time.time()
        tok1 = tik1

        def cb(future):
            nonlocal tok1
            tok1 = max(tok1, time.time())

        for i in range(0, batch_size, micro_batch_size):
            fut = self.submit(input_ids[i : i + micro_batch_size])
            fut.add_done_callback(cb)
            futures.append(fut)

        def handle_futures():
            for future in futures:
                try:
                    response, teacher_topk_logps, teacher_topk_indices = future.result()
                except Exception as e:
                    raise RuntimeError(f"Teacher request failed: {e}") from e

                all_teacher_topk_logps.extend(teacher_topk_logps)
                all_teacher_topk_indices.extend(teacher_topk_indices)
                responses.extend(response)

            tik2 = time.time()
            # teacher_topk_logps = [x.to(params_dtype) for x in all_teacher_topk_logps]
            # teacher_topk_indices = [x.to(params_dtype) for x in all_teacher_topk_indices]
            teacher_topk_logps, teacher_topk_indices = all_teacher_topk_logps, all_teacher_topk_indices

            real_seq_lens = torch.tensor([x.size(0) for x in teacher_topk_logps], dtype=torch.int32)

            topk = teacher_topk_logps[0].size(-1)

            logp_dtype = teacher_topk_logps[0].dtype
            idx_dtype = teacher_topk_indices[0].dtype
            # teacher_knowledge_shape = list(batch.batch["input_ids"].shape) + [topk]
            teacher_knowledge_shape = list(batch.batch["input_ids"].shape)

            global teacher_topk_logps_padded, teacher_topk_indices_padded
            if (
                teacher_topk_logps_padded is None
                or teacher_topk_logps_padded.dtype != logp_dtype
                or teacher_topk_logps_padded.shape != torch.Size(teacher_knowledge_shape)
            ):
                teacher_topk_logps_padded = torch.zeros(*teacher_knowledge_shape, dtype=logp_dtype)
            else:
                teacher_topk_logps_padded.zero_()

            if (
                teacher_topk_indices_padded is None
                or teacher_topk_indices_padded.dtype != idx_dtype
                or teacher_topk_indices_padded.shape != torch.Size(teacher_knowledge_shape)
            ):
                teacher_topk_indices_padded = torch.zeros(*teacher_knowledge_shape, dtype=idx_dtype)
            else:
                teacher_topk_indices_padded.zero_()

            batch_size = attention_mask.size(0)
            for i in range(batch_size):
                # teacher_topk_logp = torch.zeros_like(teacher_topk_logps[i][:, 0])
                # teacher_topk_logp[1:] = teacher_topk_logps[i][:-1, 0]
                teacher_topk_logps_padded[i, attention_mask[i]] = teacher_topk_logps[i][:, 0]
                # breakpoint()
            #   teacher_topk_indices_padded[i, attention_mask[i]] = teacher_topk_indices[i][:, 0]

            return teacher_topk_logps_padded

            # output_batch = DataProto.from_single_dict(
            #     data={"real_seq_lens": real_seq_lens},
            # )

            # output_batch.non_tensor_batch.update(
            #     {
            #         "teacher_topk_logps": teacher_topk_logps_padded.numpy(),
            #         "teacher_topk_indices": teacher_topk_indices_padded.numpy(),
            #     }
            # )

            # tok2 = time.time()
            # output_batch = {
            #     "teacher_topk_logps": teacher_topk_logps_padded,
            #     "teacher_topk_indices": teacher_topk_indices_padded
            # }
            # output_batch.meta_info["timing"] = {"get_teacher_knowledge": (tok1 - tik1) + (tok2 - tik2)}

            # return output_batch

        if is_async:
            return SimpleNamespace(get=handle_futures)
        else:
            return handle_futures()


@register("opd")
class OPDRewardManager(AbstractRewardManager):
    """The reward manager."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        max_resp_len=None,
        overlong_buffer_cfg=None,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.reward_fn_key = reward_fn_key
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len
        self.teacher_client = TeacherClient(
            os.environ['TEACHER_SERVER_IP'], int(os.environ['TEACHER_SERVER_PORT']), n_server_workers=int(os.environ['TEACHER_N_WORKERS'])
        )

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, (
                f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"
            )
            assert self.max_resp_len >= self.overlong_buffer_cfg.len, (
                "max_resp_len must be larger than overlong_buffer.len"
            )

    def __call__(self, data: DataProto, return_dict: bool = False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        # For Evaluation
        # already_print_data_sources = {}

        # for i in range(len(data)):
        #     data_item = data[i]  # DataProtoItem

        #     prompt_ids = data_item.batch["prompts"]

        #     prompt_length = prompt_ids.shape[-1]

        #     valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
        #     valid_prompt_ids = prompt_ids[-valid_prompt_length:]

        #     response_ids = data_item.batch["responses"]
        #     valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
        #     valid_response_ids = response_ids[:valid_response_length]
        #     if self.overlong_buffer_cfg.enable:
        #         overlong_buffer_len = self.overlong_buffer_cfg.len
        #         expected_len = self.max_resp_len - overlong_buffer_len
        #         exceed_len = valid_response_length - expected_len
        #         overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
        #         overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
        #         reward += overlong_reward
        #         if self.overlong_buffer_cfg.log:
        #             reward_extra_info["overlong_reward"].append(overlong_reward)
        #             reward_extra_info["overlong"].append(overlong_reward < 0)

        #     reward_tensor[i, valid_response_length - 1] = reward

        reward = self.teacher_client.get_teacher_knowledge(data, False)

        if return_dict:
            return {
                "reward_tensor": reward,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward
