import asyncio
import logging
from dataclasses import dataclass
from time import perf_counter

from .models import DetectorOutput
from .onnx_detector import (
    LocalOnnxDetector,
    ModelUnavailableError,
    sample_text_chunks,
)


logger = logging.getLogger("ai_article_check.performance")
logger.setLevel(logging.INFO)


@dataclass
class _QueuedAnalysis:
    chunks: list[str]
    future: asyncio.Future[DetectorOutput]


class InferenceBatcher:
    """Coalesce nearby article requests into bounded ONNX micro-batches."""

    def __init__(
        self,
        detector: LocalOnnxDetector,
        *,
        max_batch_chunks: int = 14,
        wait_ms: int = 40,
    ) -> None:
        if max_batch_chunks < 1:
            raise ValueError("max_batch_chunks must be positive")
        if wait_ms < 0:
            raise ValueError("wait_ms must be non-negative")
        self.detector = detector
        self.max_batch_chunks = max_batch_chunks
        self.wait_seconds = wait_ms / 1_000
        self._queue: list[_QueuedAnalysis] = []
        self._lock = asyncio.Lock()
        self._drain_scheduled = False

    async def analyze(self, text: str) -> DetectorOutput:
        chunks = sample_text_chunks(text)
        loop = asyncio.get_running_loop()
        future: asyncio.Future[DetectorOutput] = loop.create_future()
        job = _QueuedAnalysis(chunks=chunks, future=future)
        async with self._lock:
            self._queue.append(job)
            if not self._drain_scheduled:
                self._drain_scheduled = True
                loop.create_task(self._drain())
        return await future

    async def _drain(self) -> None:
        if self.wait_seconds:
            await asyncio.sleep(self.wait_seconds)
        while True:
            async with self._lock:
                jobs = self._queue
                self._queue = []
                if not jobs:
                    self._drain_scheduled = False
                    return
            await self._process(jobs)
            async with self._lock:
                if not self._queue:
                    self._drain_scheduled = False
                    return

    async def _process(self, jobs: list[_QueuedAnalysis]) -> None:
        flat_items = [
            (job_index, chunk)
            for job_index, job in enumerate(jobs)
            for chunk in job.chunks
        ]
        scores_by_job: list[list[float]] = [[] for _ in jobs]
        started = perf_counter()
        try:
            for start in range(0, len(flat_items), self.max_batch_chunks):
                batch_items = flat_items[start : start + self.max_batch_chunks]
                batch_chunks = [chunk for _, chunk in batch_items]
                scores = await asyncio.to_thread(
                    self.detector.score_chunks,
                    batch_chunks,
                )
                if len(scores) != len(batch_chunks):
                    raise ModelUnavailableError(
                        "ONNX detector returned an incomplete batch"
                    )
                for (job_index, _), score in zip(batch_items, scores, strict=True):
                    scores_by_job[job_index].append(score)

                for job_index, job in enumerate(jobs):
                    if job.future.done():
                        continue
                    job_scores = scores_by_job[job_index]
                    if len(job_scores) == len(job.chunks):
                        job.future.set_result(
                            self.detector.build_output(job.chunks, job_scores)
                        )
        except Exception as exc:
            for job in jobs:
                if not job.future.done():
                    job.future.set_exception(exc)
        finally:
            elapsed_ms = round((perf_counter() - started) * 1_000)
            logger.info(
                "inference_batch requests=%d chunks=%d runs=%d duration_ms=%d",
                len(jobs),
                len(flat_items),
                (
                    (len(flat_items) + self.max_batch_chunks - 1)
                    // self.max_batch_chunks
                ),
                elapsed_ms,
            )
