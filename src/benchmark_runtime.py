from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""Benchmark the runtime of FairMOT/DDTrack without changing track.py.

The timed region includes host-to-device tensor transfer, model inference,
detection decoding, Re-ID feature extraction, data association, and track
filtering. Image loading, model construction, result writing, evaluation, and
visualization are excluded.
"""

import os
import os.path as osp
import time
import logging

import numpy as np
import torch

import _init_paths
import datasets.dataset.jde as datasets
from opts import opts
import tracker.multitracker as multitracker_module
from tracker.multitracker import JDETracker
from tracking_utils.log import logger

# Disable the per-frame Activated/Refind/Lost/Removed DEBUG messages before
# any benchmark work starts. Terminal output inside tracker.update() would be
# included in the measured latency and make the reported FPS artificially low.
logging.disable(logging.CRITICAL)
logger.disabled = True
multitracker_module.logger.disabled = True
multitracker_module.logger.debug = lambda *args, **kwargs: None


WARMUP_FRAMES = 20
MOT20_TEST_SEQUENCES = ('MOT20-04', 'MOT20-06', 'MOT20-07', 'MOT20-08')


def read_frame_rate(sequence_root):
    seqinfo_path = osp.join(sequence_root, 'seqinfo.ini')
    with open(seqinfo_path, 'r') as seqinfo_file:
        for line in seqinfo_file:
            if line.startswith('frameRate='):
                return int(line.split('=', 1)[1].strip())
    raise ValueError('frameRate was not found in {}'.format(seqinfo_path))


def benchmark_sequence(opt, data_root, sequence, warmup_frames=WARMUP_FRAMES):
    sequence_root = osp.join(data_root, sequence)
    image_root = osp.join(sequence_root, 'img1')
    if not osp.isdir(image_root):
        raise RuntimeError('Image directory not found: {}'.format(image_root))

    dataloader = datasets.LoadImages(image_root, opt.img_size)
    frame_rate = read_frame_rate(sequence_root)
    tracker = JDETracker(opt, frame_rate=frame_rate)

    measured_times = []
    total_frames = 0

    for frame_id, (_, img, img0) in enumerate(dataloader):
        # Synchronization is required because CUDA kernels run asynchronously.
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        blob = torch.from_numpy(img).cuda().unsqueeze(0)
        online_targets = tracker.update(blob, img0)

        # Keep the same output filtering operations used by src/track.py.
        online_tlwhs = []
        online_ids = []
        for target in online_targets:
            tlwh = target.tlwh
            track_id = target.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(track_id)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time
        total_frames += 1

        if frame_id >= warmup_frames:
            measured_times.append(elapsed)

        if (frame_id + 1) % 100 == 0:
            print('{}: processed {}/{} frames'.format(
                sequence, frame_id + 1, len(dataloader)), flush=True)

    if not measured_times:
        raise RuntimeError(
            '{} has no measured frames after {} warm-up frames.'.format(
                sequence, warmup_frames))

    measured_times = np.asarray(measured_times, dtype=np.float64)
    return {
        'sequence': sequence,
        'total_frames': total_frames,
        'measured_frames': int(measured_times.size),
        'total_time': float(measured_times.sum()),
        'mean_ms': float(measured_times.mean() * 1000.0),
        'p95_ms': float(np.percentile(measured_times, 95) * 1000.0),
        'fps': float(measured_times.size / measured_times.sum()),
    }


def main(opt):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this runtime benchmark.')

    # Use exactly one visible GPU for a reproducible batch-size-1 benchmark.
    if len(opt.gpus) != 1 or opt.gpus[0] < 0:
        raise ValueError(
            'Please select exactly one GPU, for example: --gpus 0')

    data_root = osp.join(opt.data_dir, 'MOT20/test')
    print('Runtime benchmark configuration:')
    print('  script: {}'.format(osp.abspath(__file__)))
    print('  tracker module: {}'.format(osp.abspath(multitracker_module.__file__)))
    print('  model: {}'.format(opt.load_model))
    print('  data: {}'.format(data_root))
    print('  GPU: {}'.format(torch.cuda.get_device_name(0)))
    print('  input size: {}'.format(opt.img_size))
    print('  precision: FP32')
    print('  batch size: 1')
    print('  warm-up frames per sequence: {}'.format(WARMUP_FRAMES))

    sequence_results = []
    for sequence in MOT20_TEST_SEQUENCES:
        print('Benchmarking {}'.format(sequence), flush=True)
        result = benchmark_sequence(opt, data_root, sequence)
        sequence_results.append(result)
        print('{}: latency {:.2f} ms, P95 {:.2f} ms, FPS {:.2f}'.format(
            result['sequence'], result['mean_ms'], result['p95_ms'],
            result['fps']), flush=True)

    total_time = sum(item['total_time'] for item in sequence_results)
    measured_frames = sum(item['measured_frames'] for item in sequence_results)
    all_frame_times = []

    # Overall mean latency and FPS are weighted by the number of measured frames.
    overall_mean_ms = total_time / measured_frames * 1000.0
    overall_fps = measured_frames / total_time
    real_time = 'Yes' if overall_fps >= 30.0 else 'No'

    print('\n' + '=' * 72)
    print('DDTrack/FairMOT RUNTIME BENCHMARK RESULT')
    print('=' * 72)
    print('GPU                 : {}'.format(torch.cuda.get_device_name(0)))
    print('Input resolution    : {}'.format(opt.img_size))
    print('Measured frames     : {}'.format(measured_frames))
    print('Average latency     : {:.2f} ms/frame'.format(overall_mean_ms))
    print('Average FPS         : {:.2f}'.format(overall_fps))
    print('Real-time at 30 FPS : {}'.format(real_time))
    print('=' * 72)
    print('LaTeX row (replace METHOD with FairMOT or DDTrack):')
    print('METHOD & {} & {:.2f} & {:.2f} & {} \\\\'.format(
        measured_frames, overall_mean_ms, overall_fps, real_time))
    print('=' * 72)


if __name__ == '__main__':
    options = opts().init()
    main(options)
