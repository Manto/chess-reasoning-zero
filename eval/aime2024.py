import torch
import lighteval
import argparse
from accelerate import Accelerator
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from lighteval.utils.utils import EnvConfig
from lighteval.utils.imports import is_accelerate_available
from datetime import timedelta

if is_accelerate_available():
    from accelerate import Accelerator, InitProcessGroupKwargs

    accelerator = Accelerator(
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=3000))]
    )
else:
    accelerator = None


def main(engine, model, is_instruct):
    print(
        f"Running evaluation with {engine} engine, model: {model}, is_instruct: {is_instruct}"
    )

    evaluation_tracker = EvaluationTracker(
        output_dir="./results",
        save_details=True,
        push_to_hub=True,
        hub_results_org="mantle0",
    )

    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.ACCELERATE,
        env_config=EnvConfig(cache_dir="tmp/"),
        # Remove the 2 parameters below once your configuration is tested
        override_batch_size=1,
        # max_samples=10,
    )

    if engine == "vllm":
        from lighteval.models.vllm.vllm_model import VLLMModelConfig

        model_config = VLLMModelConfig(
            pretrained=model,
            dtype="float16",
            use_chat_template=not is_instruct,
        )
    elif engine == "transformers":
        from lighteval.models.transformers.transformers_model import (
            TransformersModelConfig,
        )

        accelerator = Accelerator(device_placement=True)
        model_config = TransformersModelConfig(
            accelerator=accelerator,
            pretrained=model,
            dtype="float16",
            use_chat_template=not is_instruct,
        )

    task = "lighteval|gpqa|0|0"

    pipeline = Pipeline(
        tasks=task,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model_config=model_config,
    )

    pipeline.evaluate()
    pipeline.save_and_push_results()
    pipeline.show_results()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--engine", default="transformers", choices=["vllm", "transformers"]
    )
    parser.add_argument("--is_instruct", action="store_true")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B")

    args = parser.parse_args()
    main(args.engine, args.model, args.is_instruct)
