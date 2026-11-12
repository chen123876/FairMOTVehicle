from __future__ import absolute_import, division, print_function

import _init_paths
import os
import os.path as osp
import cv2
import torch
import logging
import motmetrics as mm
import numpy as np

from lib.tracker.multitracker import JDETracker
from lib.tracking_utils.evaluation import Evaluator
import lib.datasets.dataset.jde as datasets
from lib.tracking_utils.utils import mkdir_if_missing
from lib.opts import opts
from lib.tracking_utils import visualization as vis
from lib.tracking_utils.timer import Timer
from lib.tracking_utils.log import logger

# ----------------------------
# 写跟踪结果 (带置信度)
# ----------------------------
def write_results_score(filename, results, data_type='mot'):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},{s},1,-1,-1,-1\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, tid, s in zip(tlwhs, track_ids, scores):
                if tid < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=tid, x1=x1, y1=y1, w=w, h=h, s=s)
                f.write(line)
    logger.info('Results saved to {}'.format(filename))

# ----------------------------
# 单序列跟踪
# ----------------------------
def eval_seq(opt, dataloader, data_type, result_filename, score_thres=0.3,
             save_dir=None, show_image=False, frame_rate=30, use_cuda=True):
    if save_dir:
        mkdir_if_missing(save_dir)
    tracker = JDETracker(opt, frame_rate=frame_rate)
    timer = Timer()
    results = []
    frame_id = 0

    for i, (path, img, img0) in enumerate(dataloader):
        timer.tic()
        if use_cuda:
            blob = torch.from_numpy(img).cuda().unsqueeze(0)
        else:
            blob = torch.from_numpy(img).unsqueeze(0)

        online_targets = tracker.update(blob, img0)

        online_tlwhs = []
        online_ids = []
        online_scores = []

        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            score = t.score  # 获取置信度
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical and score >= score_thres:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                online_scores.append(score)

        timer.toc()
        results.append((frame_id + 1, online_tlwhs, online_ids, online_scores))

        # 可选可视化
        if show_image or save_dir is not None:
            online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=frame_id,
                                          fps=1. / max(1e-5, timer.average_time))
        if show_image:
            cv2.imshow('online_im', online_im)
        if save_dir is not None:
            cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)

        frame_id += 1

    # 保存结果
    write_results_score(result_filename, results, data_type)
    return frame_id, timer.average_time, timer.calls

# ----------------------------
# 主函数
# ----------------------------
def main(opt, data_root, seqs, exp_name='subway_car_test', score_thres=0.3,
         show_image=False, save_images=False, save_videos=False):
    logger.setLevel(logging.INFO)
    result_root = os.path.join(data_root, '..', 'results', exp_name)
    mkdir_if_missing(result_root)
    data_type = 'mot'

    accs = []
    for seq in seqs:
        logger.info('Processing sequence: {}'.format(seq))
        seq_img_dir = osp.join(data_root, seq, 'img1')
        seq_info_file = osp.join(data_root, seq, 'seqinfo.ini')
        result_filename = osp.join(result_root, '{}.txt'.format(seq))
        dataloader = datasets.LoadImages(seq_img_dir, img_size=opt.img_size)

        # 读取帧率
        with open(seq_info_file) as f:
            meta_info = f.read()
        frame_rate = int(meta_info[meta_info.find('frameRate') + 10:meta_info.find('\nseqLength')])

        eval_seq(opt, dataloader, data_type, result_filename,
                 score_thres=score_thres, save_dir=None, show_image=show_image, frame_rate=frame_rate)

        # 评估
        evaluator = Evaluator(data_root, seq, data_type)
        accs.append(evaluator.eval_file(result_filename))

    # 输出 summary
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seqs, metrics)
    strsummary = mm.io.render_summary(summary, formatters=mh.formatters,
                                      namemap=mm.io.motchallenge_metric_names)
    print(strsummary)
    Evaluator.save_summary(summary, osp.join(result_root, 'summary_{}.xlsx'.format(exp_name)))

# ----------------------------
# 入口
# ----------------------------
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt = opts().init()
    opt.pretrained = '../weights/dla34-ba72cf86.pth'
    opt.load_model = '/2411041025/2513041010/FairMOTVehicle/exp/mot/default/model_last.pth'
    data_root = '/2411041025/2513041010/FairMOTVehicle/dataset/SubwayCar.v1/test'
    seqs = ['subway_car_c']

    main(opt, data_root=data_root, seqs=seqs, exp_name='subway_car_test',
         score_thres=0.4, show_image=False, save_images=False, save_videos=False)
