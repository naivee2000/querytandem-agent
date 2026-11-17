import json
import os
from pathlib import Path

from app.dataset import BaseDataset, load_dataset, save_dataset, DataItem
from app.llm import LLM
from app.llm.usage_ledger import run_for_item, submit_with_usage_context
from concurrent.futures import ThreadPoolExecutor, as_completed
from .generators import DCGenerator, SkeletonGenerator, ICLGenerator
from .fast_path import accepts_sql
from app.pipeline.validation import validate_pipeline_step
import time
from app.logger import logger
from app.progress import log_progress, should_checkpoint
from tqdm import tqdm
import traceback
from app.services import ArtifactStore, STAGE_ARTIFACT_FIELDS, configure_schema_service, load_stage_dataset, reset_schema_service


class SQLGenerationRunner:
    
    _llm: LLM = None
    _dataset: BaseDataset = None
    _thread_pool_executor: ThreadPoolExecutor = None
    _inner_thread_pool_executor: ThreadPoolExecutor = None
    
    _dc_generator: DCGenerator = None
    _skeleton_generator: SkeletonGenerator = None
    _icl_generator: ICLGenerator = None
    _artifact_store: ArtifactStore = None
    _extractor_max_retry: int = 3
    _stage_config = None
    _input_save_path: str = ""
    _dataset_config = None
    _parallelism: int = 16
    _progress_log_interval: int = 50
    _checkpoint_interval: int = 20
    
    def __init__(
        self,
        stage_config,
        dataset_config,
        input_save_path: str,
        extractor_max_retry: int,
        parallelism: int,
        progress_log_interval: int,
        checkpoint_interval: int,
    ):
        self._stage_config = stage_config
        if stage_config.icl_fast_path != "off" and (
            stage_config.icl_sampling_budget != 1 or stage_config.icl_example_count != 0
            or stage_config.icl_prompt_mode != "compact"
        ):
            raise ValueError("ICL routing requires one candidate and a zero-example compact prompt.")
        self._dataset_config = dataset_config
        self._input_save_path = input_save_path
        self._extractor_max_retry = extractor_max_retry
        self._parallelism = max(1, parallelism)
        self._progress_log_interval = max(1, progress_log_interval)
        self._checkpoint_interval = max(1, checkpoint_interval)
        self._artifact_store = ArtifactStore(
            self._stage_config.save_path,
            "sql_generation",
            STAGE_ARTIFACT_FIELDS["sql_generation"],
        )
        self._dataset, checkpoint_source = load_stage_dataset(
            load_dataset_fn=load_dataset,
            current_save_path=self._stage_config.save_path,
            fallback_load_path=self._input_save_path,
            artifact_store=self._artifact_store,
            stage_name="sql_generation",
        )
        logger.info(f"Initialized SQL generation dataset from {checkpoint_source}")
        configure_schema_service(max_value_example_length=self._dataset_config.max_value_example_length)
        self._llm = LLM(self._stage_config.llm)
        self._icl_llm = LLM(self._stage_config.icl_llm) if self._stage_config.icl_llm else self._llm
        logger.info(f"SQL generation parallelism: {self._parallelism}")
        self._thread_pool_executor = ThreadPoolExecutor(max_workers=self._parallelism)
        self._inner_thread_pool_executor = ThreadPoolExecutor(max_workers=self._parallelism)
        self._dc_generator = DCGenerator(extractor_max_retry=self._extractor_max_retry)
        self._skeleton_generator = SkeletonGenerator(extractor_max_retry=self._extractor_max_retry)
        self._icl_generator = ICLGenerator(
            few_shot_examples_path=self._stage_config.icl_few_shot_examples_path,
            prompt_mode=self._stage_config.icl_prompt_mode,
            example_count=self._stage_config.icl_example_count,
            extractor_max_retry=self._extractor_max_retry,
        )

    @classmethod
    def from_config(cls, app_config=None) -> "SQLGenerationRunner":
        if app_config is None:
            from app.config import get_config

            app_config = get_config()
        return cls(
            stage_config=app_config.sql_generation_config,
            dataset_config=app_config.dataset_config,
            input_save_path=app_config.schema_linking_config.save_path,
            extractor_max_retry=app_config.llm_extractor_config.max_retry,
            parallelism=app_config.run_config.parallelism,
            progress_log_interval=app_config.run_config.progress_log_interval,
            checkpoint_interval=app_config.run_config.checkpoint_interval,
        )
        
    def _generate_sql(self, data_item: DataItem) -> None:
        start_time = time.time()
        data_item.fast_path_sql = None
        
        # Track token usage for this specific data item
        total_token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        shared_root = os.environ.get("DEEPEYE_SHARED_CANDIDATES")
        cache_mode = os.environ.get("DEEPEYE_SHARED_CANDIDATES_MODE", "record")
        shared_path = None
        if shared_root:
            item_key = str(data_item.question_id)
            if not item_key.isdecimal():
                raise ValueError("Candidate replay requires numeric question IDs.")
            shared_path = Path(shared_root) / f"{item_key}.json"

        results = {}
        generation_tasks = {}
        policy = self._stage_config.icl_fast_path
        if policy != "off":
            results["icl"] = self._run_generator("icl", data_item, self._icl_generator,
                                                 self._icl_llm, self._stage_config.icl_sampling_budget)
            candidates, _ = results["icl"]
            if policy == "first":
                data_item.fast_path_sql = candidates[0] if candidates else "Error"
            elif candidates and accepts_sql(data_item.question, candidates[0], data_item.database_path, policy):
                data_item.fast_path_sql = candidates[0]

        if not data_item.fast_path_sql and shared_path is not None and (shared_path.exists() or cache_mode == "replay"):
            results.update(json.loads(shared_path.read_text()))

        if "dc" not in results and not data_item.fast_path_sql:
            generation_tasks["dc"] = submit_with_usage_context(self._inner_thread_pool_executor,
                self._run_generator, "dc", data_item, self._dc_generator, self._llm, self._stage_config.dc_sampling_budget
            )
            generation_tasks["skeleton"] = submit_with_usage_context(self._inner_thread_pool_executor,
                self._run_generator, "skeleton", data_item, self._skeleton_generator, self._llm, self._stage_config.skeleton_sampling_budget
            )
        if "icl" not in results:
            generation_tasks["icl"] = submit_with_usage_context(self._inner_thread_pool_executor,
                self._run_generator, "icl", data_item, self._icl_generator, self._icl_llm, self._stage_config.icl_sampling_budget
            )

        for name, future in generation_tasks.items():
            results[name] = future.result()

        if shared_path is not None and cache_mode == "record" and not shared_path.exists() and "dc" in results:
            shared_path.parent.mkdir(parents=True, exist_ok=True)
            pending = shared_path.with_suffix(".pending")
            pending.write_text(json.dumps({name: results[name] for name in ("dc", "skeleton")}))
            pending.replace(shared_path)

        metrics_root = os.environ.get("DEEPEYE_ICL_METRICS")
        if metrics_root:
            metrics_path = Path(metrics_root)
            metrics_path.mkdir(parents=True, exist_ok=True)
            (metrics_path / f"{int(data_item.question_id)}.json").write_text(json.dumps(results["icl"][1]))

        icl_sql_candidates, _ = results["icl"]
        data_item.first_icl_sql = icl_sql_candidates[0] if icl_sql_candidates else None
        if data_item.fast_path_sql:
            results = {"icl": results["icl"]}
        
        # Accumulate token usage
        for _, tokens in results.values():
            total_token_usage["prompt_tokens"] += tokens["prompt_tokens"]
            total_token_usage["completion_tokens"] += tokens["completion_tokens"]
            total_token_usage["total_tokens"] += tokens["total_tokens"]
        
        # Check if any generator failed (returned None)
        failed_generators = [name for name, (candidates, _) in results.items() if candidates is None]
        if data_item.fast_path_sql:
            data_item.sql_candidates = list(icl_sql_candidates or [])
        elif failed_generators:
            logger.error(f"Generator(s) {failed_generators} failed for item {data_item.question_id}, setting sql_candidates to None")
            data_item.sql_candidates = [] if os.environ.get("DEEPEYE_COUNT_NO_CANDIDATE") == "1" else None
        else:
            data_item.sql_candidates = [
                sql for name in ("dc", "skeleton", "icl") if name in results
                for sql in results[name][0]
            ]
        
        end_time = time.time()
        data_item.sql_generation_time = end_time - start_time
        data_item.sql_generation_llm_cost = total_token_usage
        data_item.total_time += data_item.sql_generation_time
        data_item.total_llm_cost = {
            "prompt_tokens": data_item.total_llm_cost["prompt_tokens"] + data_item.sql_generation_llm_cost["prompt_tokens"],
            "completion_tokens": data_item.total_llm_cost["completion_tokens"] + data_item.sql_generation_llm_cost["completion_tokens"],
            "total_tokens": data_item.total_llm_cost["total_tokens"] + data_item.sql_generation_llm_cost["total_tokens"],
        }
        
    @staticmethod
    def _run_generator(name, data_item, generator, llm, budget):
        try:
            return generator.generate(data_item, llm, budget)
        except Exception as error:
            logger.error(f"Error in {name} generation for item {data_item.question_id}: {error}")
            traceback.print_exc()
            return None, dict.fromkeys(("prompt_tokens", "completion_tokens", "total_tokens"), 0)

    def run(self):
        future_to_item = {}
        for data_item in self._dataset:
            if data_item.is_stage_complete("sql_generation"):
                logger.info(f"Skipping data item {data_item.question_id} because it has already been generated")
                continue
            future = self._thread_pool_executor.submit(run_for_item, self._generate_sql, data_item)
            future_to_item[future] = data_item
        for idx, future in tqdm(enumerate(as_completed(future_to_item), start=1), total=len(future_to_item), desc="Generating SQL"):
            future.result()
            self._artifact_store.record_item(future_to_item[future])
            log_progress("Generating SQL", idx, len(future_to_item), self._progress_log_interval, previous_completed=idx - 1)
            if should_checkpoint(idx, self._checkpoint_interval):
                self.save_result()
        logger.info("Generating SQL completed")
        
        # Validate that all required fields are filled
        self._artifact_store.flush()
        validate_pipeline_step(self._dataset, "sql_generation")
        self.save_result(materialize_snapshot=True)
        
        self._clean_up()
        
    def save_result(self, materialize_snapshot: bool = False):
        self._artifact_store.flush()
        if materialize_snapshot:
            save_dataset(self._dataset, self._stage_config.save_path)
            self._artifact_store.cleanup()
        
    def _clean_up(self):
        if self._thread_pool_executor is not None:
            self._thread_pool_executor.shutdown(wait=True)
            self._thread_pool_executor = None
        if self._inner_thread_pool_executor is not None:
            self._inner_thread_pool_executor.shutdown(wait=True)
            self._inner_thread_pool_executor = None
        self._llm = None
        self._icl_llm = None
        self._dataset = None
        self._dc_generator = None
        self._skeleton_generator = None
        self._icl_generator = None
        if self._artifact_store is not None:
            self._artifact_store.close()
        reset_schema_service()
