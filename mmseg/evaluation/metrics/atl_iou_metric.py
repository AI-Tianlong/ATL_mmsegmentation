# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence
from venv import logger

import numpy as np
import torch
from mmengine.dist import is_main_process
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from mmengine.utils import mkdir_or_exist
from PIL import Image
from prettytable import PrettyTable
from mmseg.utils import add_prefix

from mmseg.models.losses.atl_loss import (S2_5B_Dataset_21Classes_Map_nobackground,
                                          convert_low_level_label_to_High_level
                                          )
from mmseg.registry import METRICS


@METRICS.register_module()
class ATL_IoUMetric(BaseMetric):
    """ATL_IoUMetric evaluation metric. 同时计算 父类和子类的IoU.

    Args:
        ignore_index (int): Index that will be ignored in evaluation.
            Default: 255.
        iou_metrics (list[str] | str): Metrics to be calculated, the options
            includes 'mIoU', 'mDice' and 'mFscore'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        beta (int): Determines the weight of recall in the combined score.
            Default: 1.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        output_dir (str): The directory for output prediction. Defaults to
            None.
        format_only (bool): Only format result for results commit without
            perform evaluation. It is useful when you want to save the result
            to a specific format and submit it to the test server.
            Defaults to False.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
        print_level (int): The print level of the metric. Default: False.
    """
    # default_prefix = 'ATL_IoUMetric-L3的metric'  # 设置 default_prefix


    def __init__(self,
                 ignore_index: int = 255,
                 iou_metrics: List[str] = ['mIoU'],
                 nan_to_num: Optional[int] = None,
                 beta: int = 1,
                 collect_device: str = 'cpu',
                 output_dir: Optional[str] = None,
                 format_only: bool = False,
                 prefix: Optional[str] = None,
                 print_level = False,
                 **kwargs) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)


        self.L1results = []
        self.L2results = []
        self.L3results = []
        self.ignore_index = ignore_index
        self.metrics = iou_metrics
        self.nan_to_num = nan_to_num
        self.beta = beta
        self.output_dir = output_dir
        if self.output_dir and is_main_process():
            mkdir_or_exist(self.output_dir)
        self.format_only = format_only

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data and data_samples.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.  #原始图像数据和标签 [10,512,512][512,512]
            data_samples (Sequence[dict]): A batch of outputs from the model. #模型输出的pred_seg 和 seg_logits
        """
        import pdb

        # 22
        num_classes = len(self.dataset_meta['classes'])  # 这里就是22吧。因为是从数据集中获取的

        # pdb.set_trace()
        for data_sample in data_samples:

            # 构造pred_label
            i_seg_logits = data_sample['seg_logits']['data']  # [40,512,512] # 都在cuda:0上device='cuda:0'
            L1_seg_logits = i_seg_logits[0:5, :, :]  # [6,512,512]
            L2_seg_logits = i_seg_logits[5:16, :, :]  # [12,512,512]
            L3_seg_logits = i_seg_logits[16:37, :, :]  # [22,512,512]

            L1_pred_seg = torch.argmax(L1_seg_logits, dim=0)  # [512,512] [0-5]
            L2_pred_seg = torch.argmax(L2_seg_logits, dim=0)  # [512,512] [0-11]
            L3_pred_seg = torch.argmax(L3_seg_logits, dim=0)  # [512,512] [0-21]

            set_L1_pred_seg = set(L1_pred_seg.flatten().cpu().numpy().tolist())  #{0, 1, 2, 3}
            set_L2_pred_seg = set(L2_pred_seg.flatten().cpu().numpy().tolist())  #{0, 1, 3, 5, 6, 7}
            set_L3_pred_seg = set(L3_pred_seg.flatten().cpu().numpy().tolist())  #{0, 2, 5, 11, 13}

            # data_sample['pred_sem_seg']['data'] [1,512,512]-->[512,512]
            pred_label = data_sample['pred_sem_seg']['data'].squeeze(
            )  # 叠加处理过后的pred_sem_seg[] [512,512]

            # pdb.set_trace()

            # format_only always for test dataset without ground truth
            # 等价于，如果要验证：
            if not self.format_only:
                # label [512,512]
                label = data_sample['gt_sem_seg']['data'].squeeze().to(
                    i_seg_logits)  # 把label放到了gpu上，和pred_label一样的device

                label_list = convert_low_level_label_to_High_level(label, S2_5B_Dataset_21Classes_Map_nobackground)

                L1_label = label_list[0]  # [512,512] [0-5]
                L2_label = label_list[1]  # [512,512] [0-11]
                L3_label = label_list[2]  # [512,512] [0-21]

                set_label = set(label.flatten().cpu().numpy().tolist())  #{0.0, 2.0, 4.0, 10.0, 11.0, 12.0, 13.0, 16.0, 255.0}
                set_L1_label = set(L1_label.flatten().cpu().numpy().tolist())  #{0.0, 1.0, 2.0, 3.0, 255.0}
                set_L2_label = set(L2_label.flatten().cpu().numpy().tolist())  #{0.0, 1.0, 2.0, 5.0, 6.0, 7.0, 9.0, 255.0}
                set_L3_label = set(L3_label.flatten().cpu().numpy().tolist())  #{0.0, 2.0, 4.0, 10.0, 11.0, 12.0, 13.0, 16.0, 255.0}
                # pdb.set_trace()

                self.L1results.append(self.intersect_and_union(L1_pred_seg, L1_label, 5, self.ignore_index))
                self.L2results.append(self.intersect_and_union(L2_pred_seg, L2_label, 11, self.ignore_index))
                self.L3results.append(self.intersect_and_union(L3_pred_seg, L3_label, 21, self.ignore_index))


                self.results = [self.L1results, self.L2results, self.L3results]  # 会传入到mmengine-metric-evaluate

                # self.results.append(    #  len(self.results[0])
                #     self.intersect_and_union(pred_label, label, num_classes,
                #                              self.ignore_index))
                # pdb.set_trace()
            # format_result
            if self.output_dir is not None:
                basename = osp.splitext(osp.basename(
                    data_sample['img_path']))[0]
                png_filename = osp.abspath(
                    osp.join(self.output_dir, f'{basename}.png'))
                output_mask = pred_label.cpu().numpy()
                # The index range of official ADE20k dataset is from 0 to 150.
                # But the index range of output is from 0 to 149.
                # That is because we set reduce_zero_label=True.
                if data_sample.get('reduce_zero_label', False):
                    output_mask = output_mask + 1
                output = Image.fromarray(output_mask.astype(np.uint8))
                output.save(png_filename)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.  # 多进程的时候，有点问题啊。

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results. The key
                mainly includes aAcc, mIoU, mAcc, mDice, mFscore, mPrecision,
                mRecall.
        """
        
        
        
        
        logger: MMLogger = MMLogger.get_current_instance()
        if self.format_only:
            logger.info(f'results are saved to {osp.dirname(self.output_dir)}')
            return OrderedDict()
        # convert list of tuples to tuple of lists, e.g.
        # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
        # ([A_1, ..., A_n], ..., [D_1, ..., D_n])

        # import pdb;pdb.set_trace()

        results_L1 = tuple(zip(*results[0]))
        results_L2 = tuple(zip(*results[1]))
        results_L3 = tuple(zip(*results[2]))

        # import pdb;pdb.set_trace()

        assert len(results_L1) == 4 and len(results_L2) == 4 and len(
            results_L3) == 4

        total_area_intersect_L1 = sum(results_L1[0])  # I
        total_area_union_L1 = sum(results_L1[1])  # U
        total_area_pred_label_L1 = sum(results_L1[2])
        total_area_label_L1 = sum(results_L1[3])
        ret_metrics_L1 = self.total_area_to_metrics(
            total_area_intersect_L1, total_area_union_L1, total_area_pred_label_L1,
            total_area_label_L1, self.metrics, self.nan_to_num, self.beta)
        
        total_area_intersect_L2 = sum(results_L2[0])  # I
        total_area_union_L2 = sum(results_L2[1])  # U
        total_area_pred_label_L2 = sum(results_L2[2])
        total_area_label_L2 = sum(results_L2[3])
        ret_metrics_L2 = self.total_area_to_metrics(
            total_area_intersect_L2, total_area_union_L2, total_area_pred_label_L2,
            total_area_label_L2, self.metrics, self.nan_to_num, self.beta)
        
        total_area_intersect_L3 = sum(results_L3[0])  # I
        total_area_union_L3 = sum(results_L3[1])  # U
        total_area_pred_label_L3 = sum(results_L3[2])
        total_area_label_L3 = sum(results_L3[3])
        ret_metrics_L3 = self.total_area_to_metrics(
            total_area_intersect_L3, total_area_union_L3, total_area_pred_label_L3,
            total_area_label_L3, self.metrics, self.nan_to_num, self.beta)


        # 这里给他换掉！！！
        # class_names = self.dataset_meta['classes']
        class_names_L1 = list(
            S2_5B_Dataset_21Classes_Map_nobackground['Classes_Map_L1'].keys())
        class_names_L2 = list(
            S2_5B_Dataset_21Classes_Map_nobackground['Classes_Map_L2'].keys())
        class_names_L3 = list(
            S2_5B_Dataset_21Classes_Map_nobackground['Classes_Map_L3'].keys())

        # processed_class_names_L1 = [name.split('_')[-1] for name in class_names_L1]
        # processed_class_names_L2 = [name.split('_')[-1] for name in class_names_L2]
        # processed_class_names_L3 = [name.split('_')[-1] for name in class_names_L3]


        # pdb.set_trace()
        # summary table
        # OrderedDict([('aAcc', 75.64), ('IoU', 50.74), ('Acc', 63.67), ('Fscore', 64.4), ('Precision', 66.95), ('Recall', 63.67)])
        ret_metrics_summary_L1 = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics_L1.items()
        })  
        ret_metrics_summary_L2 = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics_L2.items()
        })
        ret_metrics_summary_L3 = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)  # 变成百分比的形式
            for ret_metric, ret_metric_value in ret_metrics_L3.items()  #不对 怎么全是6
        })

        metrics_L1 = dict()
        for key, val in ret_metrics_summary_L1.items():
            if key == 'aAcc':
                metrics_L1[key] = val
            else:
                metrics_L1['m' + key] = val

        metrics_L2 = dict()
        for key, val in ret_metrics_summary_L2.items():
            if key == 'aAcc':
                metrics_L2[key] = val
            else:
                metrics_L2['m' + key] = val

        metrics_L3 = dict()
        for key, val in ret_metrics_summary_L3.items():
            if key == 'aAcc':
                metrics_L3[key] = val
            else:
                metrics_L3['m' + key] = val

        # metrics_L1=add_prefix(metrics_L1, 'L1_level')  # decode.atl_loss
        # metrics_L2=add_prefix(metrics_L2, 'L2_level')  # decode.atl_loss
        # metrics_L3=add_prefix(metrics_L3, 'L3_level')  # decode.atl_loss

        # each class table
        ret_metrics_L1.pop('aAcc', None)
        ret_metrics_L2.pop('aAcc', None)
        ret_metrics_L3.pop('aAcc', None)

        ret_metrics_class_L1 = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics_L1.items()
        })

        ret_metrics_class_L2 = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics_L2.items()
        })

        ret_metrics_class_L3 = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics_L3.items()
        })

        ret_metrics_class_L1.update({'Class': class_names_L1})
        ret_metrics_class_L2.update({'Class': class_names_L2})
        ret_metrics_class_L3.update({'Class': class_names_L3})


        ret_metrics_class_L1.move_to_end('Class', last=False)  # 把class放在最前面
        ret_metrics_class_L2.move_to_end('Class', last=False)
        ret_metrics_class_L3.move_to_end('Class', last=False)

        class_table_data_L1 = PrettyTable()
        for key, val in ret_metrics_class_L1.items():
            class_table_data_L1.add_column(key, val)

        # import pdb
        # # pdb.set_trace()
        class_table_data_L2 = PrettyTable()
        for key, val in ret_metrics_class_L2.items():
            class_table_data_L2.add_column(key, val)

        class_table_data_L3 = PrettyTable()
        for key, val in ret_metrics_class_L3.items():
            class_table_data_L3.add_column(key, val)


        # 这里就已经可以打印层级的输出了！！！！
        print_log('per class results:', logger)
        print_log('\n' + class_table_data_L1.get_string(), logger=logger)
        print_log(metrics_L1, logger=logger)

        print_log('\n' + class_table_data_L2.get_string(), logger=logger)
        print_log(metrics_L2, logger=logger)

        print_log('\n' + class_table_data_L3.get_string(), logger=logger)
        print_log(metrics_L3, logger=logger)

        metrics = metrics_L3  # 最后这个metric，是去计算平均值的。
        # 然后进到 mmengine-evluator的evaluate()方法, line 77
        return metrics  #  一个process 一个 compute_metrics，两个需要覆盖的函数

    @staticmethod
    def intersect_and_union(pred_label: torch.tensor, label: torch.tensor,
                            num_classes: int, ignore_index: int):
        """Calculate Intersection and Union.

        Args:
            pred_label (torch.tensor): Prediction segmentation map
                or predict result filename. The shape is (H, W).
            label (torch.tensor): Ground truth segmentation map
                or label filename. The shape is (H, W).
            num_classes (int): Number of categories.
            ignore_index (int): Index that will be ignored in evaluation.

        Returns:
            torch.Tensor: The intersection of prediction and ground truth
                histogram on all classes.
            torch.Tensor: The union of prediction and ground truth histogram on
                all classes.
            torch.Tensor: The prediction histogram on all classes.
            torch.Tensor: The ground truth histogram on all classes.
        """

        mask = (label != ignore_index)
        pred_label = pred_label[mask]
        label = label[mask]

        intersect = pred_label[pred_label == label]
        area_intersect = torch.histc(
            intersect.float(), bins=(num_classes), min=0,
            max=num_classes - 1).cpu()
        area_pred_label = torch.histc(
            pred_label.float(), bins=(num_classes), min=0,
            max=num_classes - 1).cpu()
        area_label = torch.histc(
            label.float(), bins=(num_classes), min=0,
            max=num_classes - 1).cpu()
        area_union = area_pred_label + area_label - area_intersect
        return area_intersect, area_union, area_pred_label, area_label

    @staticmethod
    def total_area_to_metrics(total_area_intersect: np.ndarray,
                              total_area_union: np.ndarray,
                              total_area_pred_label: np.ndarray,
                              total_area_label: np.ndarray,
                              metrics: List[str] = ['mIoU'],
                              nan_to_num: Optional[int] = None,
                              beta: int = 1):
        """Calculate evaluation metrics
        Args:
            total_area_intersect (np.ndarray): The intersection of prediction
                and ground truth histogram on all classes.
            total_area_union (np.ndarray): The union of prediction and ground
                truth histogram on all classes.
            total_area_pred_label (np.ndarray): The prediction histogram on
                all classes.
            total_area_label (np.ndarray): The ground truth histogram on
                all classes.
            metrics (List[str] | str): Metrics to be evaluated, 'mIoU' and
                'mDice'.
            nan_to_num (int, optional): If specified, NaN values will be
                replaced by the numbers defined by the user. Default: None.
            beta (int): Determines the weight of recall in the combined score.
                Default: 1.
        Returns:
            Dict[str, np.ndarray]: per category evaluation metrics,
                shape (num_classes, ).
        """

        def f_score(precision, recall, beta=1):
            """calculate the f-score value.

            Args:
                precision (float | torch.Tensor): The precision value.
                recall (float | torch.Tensor): The recall value.
                beta (int): Determines the weight of recall in the combined
                    score. Default: 1.

            Returns:
                [torch.tensor]: The f-score value.
            """
            score = (1 + beta**2) * (precision * recall) / (
                (beta**2 * precision) + recall)
            return score

        if isinstance(metrics, str):
            metrics = [metrics]
        allowed_metrics = ['mIoU', 'mDice', 'mFscore']
        if not set(metrics).issubset(set(allowed_metrics)):
            raise KeyError(f'metrics {metrics} is not supported')

        all_acc = total_area_intersect.sum() / total_area_label.sum()
        ret_metrics = OrderedDict({'aAcc': all_acc})
        for metric in metrics:
            if metric == 'mIoU':
                iou = total_area_intersect / total_area_union
                acc = total_area_intersect / total_area_label
                ret_metrics['IoU'] = iou
                ret_metrics['Acc'] = acc
            elif metric == 'mDice':
                dice = 2 * total_area_intersect / (
                    total_area_pred_label + total_area_label)
                acc = total_area_intersect / total_area_label
                ret_metrics['Dice'] = dice
                ret_metrics['Acc'] = acc
            elif metric == 'mFscore':
                precision = total_area_intersect / total_area_pred_label
                recall = total_area_intersect / total_area_label
                f_value = torch.tensor([
                    f_score(x[0], x[1], beta) for x in zip(precision, recall)
                ])
                ret_metrics['Fscore'] = f_value
                ret_metrics['Precision'] = precision
                ret_metrics['Recall'] = recall

        ret_metrics = {
            metric: value.numpy()
            for metric, value in ret_metrics.items()
        }
        if nan_to_num is not None:
            ret_metrics = OrderedDict({
                metric: np.nan_to_num(metric_value, nan=nan_to_num)
                for metric, metric_value in ret_metrics.items()
            })
        return ret_metrics
