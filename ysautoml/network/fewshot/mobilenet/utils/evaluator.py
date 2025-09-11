# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import numpy as np
import torch

from ..utils.comm import all_gather, is_main_process, synchronize

class DatasetEvaluator:
    """
    Base class for a dataset evaluator.

    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.

    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        pass

    def process(self, inputs, outputs):
        """
        Process the pair of inputs and outputs.
        If they contain batches, the pairs can be consumed one-by-one using `zip`:

        .. code-block:: python

            for input_, output in zip(inputs, outputs):
                # do evaluation on single input/output pair
                ...

        Args:
            inputs (list): the inputs that's used to call the model.
            outputs (list): the return value of `model(inputs)`
        """
        pass

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.

        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:

                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """
        pass



class Evaluator(DatasetEvaluator):
    def __init__(self, num_classes=1000, distributed=True):
        """
        Args:
            distributed (bool): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
        """
        self._num_classes = num_classes
        self._distributed = distributed
        self._cpu_device = torch.device("cpu")

    def reset(self):
        self._top1 = 0.
        self._top5 = 0.
        self._total_samples = 0.

    def process(self, output, target, topk=(1, 5)):
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k)

        self._top1 += res[0].item() 
        self._top5 += res[1].item()
        self._total_samples += batch_size

    def evaluate(self):
        """
        Computes the precision@k for the specified values of k
        """
        if self._distributed:
            synchronize()
            top1_list = all_gather(self._top1)
            top5_list = all_gather(self._top5)
            total_samples_list = all_gather(self._total_samples)
            if not is_main_process():
                return

            self._top1 = 0.
            self._top5 = 0. 
            self._total_samples = 0.
            for t1,t5,ts in zip(top1_list, top5_list, total_samples_list):
                self._top1 += t1
                self._top5 += t5
                self._total_samples += ts

        t1_acc = 100 * self._top1 / self._total_samples
        t5_acc = 100 * self._top5 / self._total_samples
        res = {}
        res["num_samples"] = self._total_samples
        res["accs"] = (t1_acc, t5_acc)
        return res
