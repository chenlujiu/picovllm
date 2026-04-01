from collections import deque
from dataclasses import dataclass

from picovllm.config import Config
from picovllm.engine.sequence import Sequence, SequenceStatus
from picovllm.engine.block_manager import BlockManager

@dataclass
class SchedulerOutput:
    seqs: list[Sequence]
    num_scheduled_tokens: list[int]
    token_ids_list: list[list[int]]


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> SchedulerOutput:
        budget = self.max_num_batched_tokens
        scheduled_seqs = []
        num_scheduled_tokens = []
        token_ids_list = []

        # running
        temp_running = []
        while self.running:
            if budget == 0:
                temp_running.extend(self.running)
                self.running.clear()
                break

            seq = self.running.popleft()
            num_new = seq.num_tokens - seq.num_computed_tokens
            num_new = min(num_new, budget)

            is_decode = (seq.num_computed_tokens >= seq.num_prompt_tokens)
            if is_decode:
                # decode phase might need new block
                if num_new == 0 or not self.block_manager.can_append(seq):
                    self.preempt(seq)
                    continue
                self.block_manager.update_blocks(seq)
                token_ids_list.append([seq.last_token])
            else:
                # prefill phase blocks have all set
                start = seq.num_computed_tokens
                token_ids_list.append(seq.token_ids[start:start + num_new])

            budget -= num_new
            scheduled_seqs.append(seq)
            num_scheduled_tokens.append(num_new)
            temp_running.append(seq)
        self.running = deque(temp_running)

        # waiting
        while self.waiting and budget > 0:
            if len(scheduled_seqs) >= self.max_num_seqs:
                break
            seq = self.waiting[0]
            if not self.block_manager.can_allocate(seq):
                break
            self.block_manager.allocate(seq)

            remaining = len(seq) - seq.num_computed_tokens
            chunk_size = min(remaining, budget)
            budget -= chunk_size

            start = seq.num_computed_tokens
            token_ids_list.append(seq.token_ids[start:start+chunk_size])

            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
            num_scheduled_tokens.append(chunk_size)

        return SchedulerOutput(scheduled_seqs, num_scheduled_tokens, token_ids_list)

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, output: SchedulerOutput, token_ids: list[int]):
        token_idx = 0
        for seq, nst in zip(output.seqs, output.num_scheduled_tokens):
            seq.num_computed_tokens += nst

            prefill_complete = (seq.num_computed_tokens >= seq.num_prompt_tokens)
            if prefill_complete:
                token_id = token_ids[token_idx]
                seq.append_token(token_id)
                token_idx += 1
                if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                    seq.status= SequenceStatus.FINISHED
                    self.block_manager.deallocate(seq)
                    self.running.remove(seq)
            else:
                token_idx += 1



