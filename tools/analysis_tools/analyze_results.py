# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp

import rmsm
from rmsm import DictAction

from rmsm.datasets import build_dataset
from rmsm.models import build_classifier


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMCls evaluate prediction success/fail')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('result', help='test result json/pkl file')
    parser.add_argument('--out-dir', help='dir to store output files')
    parser.add_argument(
        '--topk',
        default=20,
        type=int,
        help='Number of data to select for success/fail')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()

    return args


def save_rams(result_dir, folder_name, results, model):
    full_dir = osp.join(result_dir, folder_name)
    rmsm.mkdir_or_exist(full_dir)
    rmsm.utils.dump(results, osp.join(full_dir, folder_name + '.json'))

    # save rams
    show_keys = ['pred_score', 'pred_class', 'gt_class']
    for result in results:
        result_show = dict((k, v) for k, v in result.items() if k in show_keys)
        outfile = osp.join(full_dir, osp.basename(result['raman_path']))
        model.show_result(result['spectrum'], result_show, out_file=outfile)


def main():
    args = parse_args()

    # load test results
    outputs = rmsm.load(args.result)
    assert ('pred_score' in outputs and 'pred_class' in outputs
            and 'pred_label' in outputs), \
        'No "pred_label", "pred_score" or "pred_class" in result file, ' \
        'please set "--out-items" in txt_tocsv.py'

    cfg = rmsm.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    model = build_classifier(cfg.model)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    # filenames = list()
    # for info in dataset.data_infos:
    #     if info['ram_prefix'] is not None:
    #         filename = osp.join(info['ram_prefix'],
    #                             info['ram_info']['filename'])
    #     else:
    #         filename = info['ram_info']['filename']
    #     filenames.append(filename)
    labels = list(dataset.get_labels())
    gt_classes = [dataset.CLASSES[x] for x in labels]

    outputs['raman_path'] = dataset.data_infos['raman_path']
    outputs['labels'] = labels
    outputs['gt_class'] = gt_classes

    need_keys = [
        'raman_path', 'labels', 'gt_class', 'pred_score', 'pred_label',
        'pred_class'
    ]
    outputs = {k: v for k, v in outputs.items() if k in need_keys}
    outputs_list = list()
    for i in range(len(labels)):
        output = dict()
        for k in outputs.keys():
            output[k] = outputs[k][i]
        outputs_list.append(output)

    # sort result
    outputs_list = sorted(outputs_list, key=lambda x: x['pred_score'])

    success = list()
    fail = list()
    for output in outputs_list:
        if output['pred_label'] == output['labels']:
            success.append(output)
        else:
            fail.append(output)

    success = success[:args.topk]
    fail = fail[:args.topk]

    save_rams(args.out_dir, 'success', success, model)
    save_rams(args.out_dir, 'fail', fail, model)


if __name__ == '__main__':
    main()
