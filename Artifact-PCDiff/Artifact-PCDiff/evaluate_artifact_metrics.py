"""Evaluate saved completion outputs with the paper's CDF, HDF, and BCD."""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

from .utils.artifact_metrics import (
    boundary_chamfer_distance,
    sampling_normalized_factors,
    summarize_metric,
)


def _dataset_paths(csv_row):
    complete_path = csv_row.replace('\\', '/')
    if 'complete' not in complete_path:
        raise ValueError("CSV row does not contain a 'complete' path: {}".format(csv_row))
    prefix, obj_name = complete_path.split('complete', 1)
    stem = os.path.splitext(obj_name)[0]
    repair_path = Path(prefix + 'repair' + stem + '_0.npy')
    result_name = 'broken' + os.path.basename(stem) + '_0'
    return repair_path, result_name


def _point_array(array, name):
    array = np.asarray(array)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 2 and array.shape[1] != 3 and array.shape[0] == 3:
        array = array.T
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError('{} must have shape [N, 3]; got {}'.format(name, array.shape))
    return array


def evaluate(args):
    results_dir = Path(args.results_dir)
    output_csv = Path(args.output_csv)
    rows = []
    unavailable = []

    with Path(args.csv).open('r', newline='') as csv_file:
        for csv_row in csv.reader(csv_file):
            if not csv_row:
                continue
            repair_path, result_name = _dataset_paths(csv_row[0])
            result_dir = results_dir / result_name
            expected = [
                repair_path,
                result_dir / 'input.npy',
                result_dir / 'sample.npy',
                result_dir / 'shift.npy',
                result_dir / 'scale.npy',
            ]
            missing = [str(path) for path in expected if not path.is_file()]
            if missing:
                unavailable.append({'object': result_name, 'missing': missing})
                continue
            rows.append((result_name, repair_path, result_dir))

    if unavailable and not args.allow_incomplete:
        lines = [
            'Metrics require a repair ground truth and saved inference outputs for every CSV row.',
            'Unavailable objects: {}'.format(len(unavailable)),
        ]
        for item in unavailable[:12]:
            lines.append('  {}: {}'.format(item['object'], ', '.join(item['missing'])))
        if len(unavailable) > 12:
            lines.append('  ... and {} more'.format(len(unavailable) - 12))
        raise FileNotFoundError('\n'.join(lines))
    if not rows:
        raise FileNotFoundError('No complete prediction/ground-truth pairs were found.')

    metric_rows = []
    for result_name, repair_path, result_dir in rows:
        samples = np.load(result_dir / 'sample.npy', allow_pickle=False)
        if samples.ndim == 2:
            samples = samples[None, ...]
        if samples.ndim != 3 or samples.shape[-1] != 3:
            raise ValueError('{} sample.npy must have shape [E, N, 3].'.format(result_name))
        if args.ensemble_index >= samples.shape[0]:
            raise ValueError(
                '{} has {} ensemble sample(s), but index {} was requested.'.format(
                    result_name, samples.shape[0], args.ensemble_index
                )
            )

        prediction = _point_array(samples[args.ensemble_index, -args.evaluation_points:, :], 'prediction')
        fragment = _point_array(np.load(result_dir / 'input.npy', allow_pickle=False), 'fragment')
        target = _point_array(np.load(repair_path, allow_pickle=False), 'target repair')
        shift = np.asarray(np.load(result_dir / 'shift.npy', allow_pickle=False)).reshape(-1, 3)[0]
        scale = float(np.asarray(np.load(result_dir / 'scale.npy', allow_pickle=False)).reshape(-1)[0])
        if not np.isfinite(scale) or scale == 0:
            raise ValueError('{} has an invalid saved normalization scale.'.format(result_name))
        target = (target - shift.reshape(1, 3)) / scale

        factors = sampling_normalized_factors(
            prediction,
            target,
            evaluation_points=args.evaluation_points,
            trials=args.trials,
            seed=args.seed,
        )
        bcd = boundary_chamfer_distance(
            prediction,
            target,
            fragment,
            interface_points=args.interface_points,
        )
        metric_rows.append({'object': result_name, **factors, 'BCD': bcd})

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ['object', 'CDF', 'HDF', 'BCD', 'CD_sampling_baseline', 'HD_sampling_baseline']
    with output_csv.open('w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metric_rows)

    summary = {
        'evaluated_objects': len(metric_rows),
        'unavailable_objects': unavailable,
        'ensemble_index': args.ensemble_index,
        'evaluation_points': args.evaluation_points,
        'trials': args.trials,
        'interface_points': args.interface_points,
        'metrics': {
            metric: summarize_metric([row[metric] for row in metric_rows])
            for metric in ('CDF', 'HDF', 'BCD')
        },
    }
    output_json = output_csv.with_suffix('.json')
    output_json.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('Wrote {} per-object rows to {}'.format(len(metric_rows), output_csv))
    print('Wrote summary to {}'.format(output_json))
    if unavailable:
        print('WARNING: evaluated only {}/{} objects; missing outputs/ground truth are recorded in the JSON.'.format(
            len(metric_rows), len(metric_rows) + len(unavailable)
        ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', required=True, help='test CSV listing complete-object paths')
    parser.add_argument('--results-dir', required=True, help='test output syn directory')
    parser.add_argument('--output-csv', required=True, help='per-object metric CSV path')
    parser.add_argument('--evaluation-points', type=int, default=3072)
    parser.add_argument('--trials', type=int, default=20)
    parser.add_argument('--interface-points', type=int, default=64)
    parser.add_argument('--ensemble-index', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--allow-incomplete', action='store_true',
        help='write metrics for available pairs only and list omitted objects in the JSON summary',
    )
    args = parser.parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()
