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
from functools import partial

import torch
import zmq
import io
import os
import time
import queue
import threading
import concurrent.futures
import warnings

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager

from transformers import AutoTokenizer

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
        teacher_ckpt_path,
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
        self.tokenizer = AutoTokenizer.from_pretrained(teacher_ckpt_path)
        self.student_tokenizer = None
        self._run()

    def retokenize_batch(self, batch):
        if self.tokenizer == self.student_tokenizer:
            return batch
        else:
            input_id_list = []
            bsz = len(batch)
            max_seq_len = 0

            for seq_idx in range(bsz):
                seq_str = ""
                input_ids_list = batch[seq_idx]
                token_idx = 0
                counter = 0
                while token_idx < len(input_ids_list):
                    counter = token_idx
                    if input_ids_list[token_idx] == self.student_tokenizer.bos_token_id:
                        while(input_ids_list[token_idx] != self.student_tokenizer.eos_token_id and token_idx < len(input_ids_list) - 1): token_idx += 1
                        seq_str += self.tokenizer.decode(self.tokenizer.bos_token_id)
                        seq_str += self.student_tokenizer.decode(input_ids_list[counter + 1:token_idx])
                        if input_ids_list[token_idx] == self.student_tokenizer.eos_token_id:
                            seq_str += self.tokenizer.decode(self.tokenizer.eos_token_id)
                        else:
                            seq_str += self.student_tokenizer.decode(input_ids_list[token_idx])
                        token_idx += 1
                    else:
                        while(input_ids_list[token_idx] != self.student_tokenizer.bos_token_id and token_idx < len(input_ids_list) - 1): token_idx += 1
                        seq_str += self.student_tokenizer.decode(input_ids_list[counter:token_idx])
                        if input_ids_list[token_idx] == self.student_tokenizer.bos_token_id:
                            pass
                        else:
                            seq_str += self.student_tokenizer.decode(input_ids_list[token_idx])
                            token_idx += 1
                new_input_id = self.tokenizer(seq_str)['input_ids']
                max_seq_len = max(max_seq_len, len(new_input_id))
                # print(self.student_tokenizer.decode(input_ids_list) == seq_str)
                input_id_list.append(new_input_id)

            # new_input_ids = torch.full([bsz, max_seq_len], self.tokenizer.eos_token_id, dtype=torch.int32)

            # for seq_idx in range(bsz):
            #     new_input_id = input_id_list[seq_idx]
            #     new_input_ids[seq_idx, : new_input_id.size()] = new_input_id
            return input_id_list
                

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

                batch = self.retokenize_batch(batch)
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

    def get_teacher_knowledge(self, batch: DataProto, is_async=False, student_tokenizer=None):
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

        assert student_tokenizer is not None, "To get knowledge of teacher, tokenizer of student must be passed"
        self.student_tokenizer = student_tokenizer
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

            def response_alignment(teacher_log_probs, student_token_ids, teacher_token_ids):
                assert len(self.tokenizer.decode(teacher_token_ids)) >= len(self.student_tokenizer.decode(student_token_ids[:-1])), "Output string length of teacher model must be not less than student's"
                if self.tokenizer.decode(teacher_token_ids).startswith(self.student_tokenizer.decode(student_token_ids)):
                    return teacher_log_probs, student_token_ids, teacher_token_ids
                else:
                    for i in range(len(teacher_token_ids)):
                        if (self.tokenizer.decode(teacher_token_ids[i:]).startswith(self.student_tokenizer.decode(student_token_ids))):
                            return teacher_log_probs[i:], student_token_ids, teacher_token_ids[i:]
                    return None, None, None
                
            def post_process_teacher_log_probs(teacher_log_probs, student_token_ids, teacher_token_ids):
                assert teacher_log_probs is not None, "teacher_log_probs should not be None"
                if self.tokenizer == student_tokenizer:
                    return teacher_log_probs
                new_teacher_log_probs = torch.zeros(len(student_token_ids) + 1, dtype=teacher_log_probs.dtype, device=teacher_log_probs.device)
                new_teacher_log_probs[-1] = teacher_log_probs[-1]
                
                counter_tec_seq, token_idx = 0, 0
                std_chunk, tec_chunk = [], []
                cat_str = ""
                
                while token_idx < len(student_token_ids) or counter_tec_seq < len(teacher_token_ids):
                    tec_chunk_str = self.tokenizer.decode(tec_chunk)
                    std_chunk_str = self.student_tokenizer.decode(std_chunk)
                    if (len(std_chunk) >= 6 and tec_chunk_str[-5:] == std_chunk_str[-5:]):
                        new_teacher_log_probs[token_idx - len(std_chunk):token_idx] = teacher_log_probs[counter_tec_seq - len(tec_chunk):counter_tec_seq].sum() / len(std_chunk)
                        std_chunk, tec_chunk = [], []
                    if (token_idx == len(student_token_ids) or (student_token_ids[token_idx] == self.tokenizer.eos_token_id) or (student_token_ids[token_idx] == self.tokenizer.bos_token_id)):
                        while((counter_tec_seq < len(teacher_token_ids)) and (teacher_token_ids[counter_tec_seq] != self.tokenizer.eos_token_id or teacher_token_ids[counter_tec_seq] != self.tokenizer.bos_token_id)): 
                            tec_chunk.append(teacher_token_ids[counter_tec_seq])
                            counter_tec_seq += 1
                        cat_str += std_chunk_str
                        new_teacher_log_probs[token_idx - len(std_chunk):token_idx] = teacher_log_probs[counter_tec_seq - len(tec_chunk):counter_tec_seq].sum() / len(std_chunk)
                        new_teacher_log_probs[token_idx] = teacher_log_probs[counter_tec_seq] if std_chunk[-1] == self.tokenizer.eos_token_id else new_teacher_log_probs[token_idx]
                        std_chunk, tec_chunk = [], []
                    elif (len(tec_chunk_str) > len(tec_chunk_str)):
                        std_chunk.append(student_token_ids[token_idx]) 
                        token_idx += 1
                    elif (len(tec_chunk_str) < len(std_chunk_str)):
                        tec_chunk.append(teacher_token_ids[counter_tec_seq])
                        counter_tec_seq += 1
                    elif (len(std_chunk) == 0):
                        std_chunk.append(student_token_ids[token_idx])
                        token_idx += 1
                    elif tec_chunk_str != std_chunk_str:
                        std_chunk.append(student_token_ids[token_idx])
                        token_idx += 1
                    else:
                        # assert tec_chunk_str == std_chunk_str, "Student's token chunk doesn't match teacher's"
                        cat_str += std_chunk_str
                        new_teacher_log_probs[token_idx - len(std_chunk):token_idx] = teacher_log_probs[counter_tec_seq - len(tec_chunk):counter_tec_seq].sum() / len(std_chunk)
                        std_chunk, tec_chunk = [], []
                
                # if cat_str != self.student_tokenizer.decode(student_token_ids):breakpoint()
                # assert cat_str == self.student_tokenizer.decode(student_token_ids), "Joint of Token chunk must match source token sequence"
                return new_teacher_log_probs
                            

            def process_single_item(i, teacher_topk_logps, input_ids, responses, attention_mask, teacher_topk_logps_padded):
                print(f"Data ID {i}")
                try:
                    teacher_log_probs, student_token_ids, teacher_token_ids = response_alignment(
                        teacher_topk_logps[i][:, 0], 
                        input_ids[i][1:], 
                        responses[i][1:]
                    )
                    if teacher_log_probs.is_cuda:
                        teacher_log_probs = teacher_log_probs.cpu()
                    
                    result = post_process_teacher_log_probs(
                        teacher_log_probs, 
                        student_token_ids, 
                        teacher_token_ids
                    )
                    return i, result, attention_mask[i], None
                except Exception as e:
                    warnings.warn(f"Error in post_process_teacher_log_probs: {e}. Setting teacher_topk_logps_padded to zeros.")
                    return i, None, attention_mask[i], e

            # 并行处理
            with concurrent.futures.ThreadPoolExecutor(128) as executor:
                process_func = partial(
                    process_single_item,
                    teacher_topk_logps=teacher_topk_logps,
                    input_ids=input_ids,
                    responses=responses,
                    attention_mask=attention_mask,
                    teacher_topk_logps_padded=teacher_topk_logps_padded
                )
                
                futures_ = [executor.submit(process_func, i) for i in range(batch_size)]
                
                for future in concurrent.futures.as_completed(futures_):
                    i, result, mask, error = future.result()
                    
                    if error is None:
                        teacher_topk_logps_padded[i, mask] = result
                    else:
                        teacher_topk_logps_padded[i, mask] = torch.zeros_like(
                            teacher_topk_logps_padded[i, mask]
                        )
            return teacher_topk_logps_padded

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
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.reward_fn_key = reward_fn_key
        self.max_resp_len = max_resp_len
        self.teacher_client = TeacherClient(
            os.environ['TEACHER_SERVER_IP'], int(os.environ['TEACHER_SERVER_PORT']), n_server_workers=int(os.environ['TEACHER_N_WORKERS']), teacher_ckpt_path=os.environ['TEACHER_CKPT_PATH'],
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
        is_validate = data.meta_info.get("validate", False)

        # For Evaluation
        if is_validate:
            reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

            already_print_data_sources = {}

            for i in range(len(data)):
                data_item = data[i]  # DataProtoItem

                prompt_ids = data_item.batch["prompts"]

                prompt_length = prompt_ids.shape[-1]

                valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
                valid_prompt_ids = prompt_ids[-valid_prompt_length:]

                response_ids = data_item.batch["responses"]
                valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                valid_response_ids = response_ids[:valid_response_length]

                # decode
                prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
                response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

                ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
                data_source = data_item.non_tensor_batch[self.reward_fn_key]
                extra_info = data_item.non_tensor_batch.get("extra_info", {})
                num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
                rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
                extra_info["num_turns"] = num_turns
                extra_info["rollout_reward_scores"] = rollout_reward_scores

                score = default_compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                )

                if isinstance(score, dict):
                    reward = score["score"]
                    # Store the information including original reward
                    for key, value in score.items():
                        reward_extra_info[key].append(value)
                else:
                    reward = score

                reward_tensor[i, valid_response_length - 1] = reward

                if data_source not in already_print_data_sources:
                    already_print_data_sources[data_source] = 0

                if already_print_data_sources[data_source] < self.num_examine:
                    already_print_data_sources[data_source] += 1
                    print("[prompt]", prompt_str)
                    print("[response]", response_str)
                    print("[ground_truth]", ground_truth)
                    if isinstance(score, dict):
                        for key, value in score.items():
                            print(f"[{key}]", value)
                    else:
                        print("[score]", score)
            reward = reward_tensor
        else:
            reward = self.teacher_client.get_teacher_knowledge(data, False, self.tokenizer)
        # breakpoint()

        if return_dict:
            return {
                "reward_tensor": reward,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward
