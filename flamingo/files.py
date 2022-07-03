# encoding=utf-8

import json
import tensorflow as tf


def load_lines(file_name, decode=True, lower=False, ignore_empty=True, tsv_format=False):
    tsv_formatter = '\t'.decode('utf-8') if decode else '\t'
    with tf.gfile.GFile(file_name, "r") as file:
        line = file.readline()

        while line:
            line = line.strip('\n')
            if decode:
                line = line.decode('utf-8')

            if lower:
                line = line.lower()

            if ignore_empty:
                if len(line.strip()) == 0:
                    line = file.readline()
                    continue

            if tsv_format:
                line = line.split(tsv_formatter)

            yield line
            line = file.readline()
    return


def _get_str_from_line(line):
    if isinstance(line, str) or isinstance(line, unicode):
        return line
    else:
        line_modified = ['%s' % term for term in line]
        return '\t'.join(line_modified)


def save_lines(data, file_name, format_method=_get_str_from_line):
    with tf.gfile.GFile(file_name, 'w') as file:
        for line in data:
            if format_method:
                line = format_method(line)
            if line is not None:
                file.write(line)
                file.write('\n')


def save_all_lines(data_iter_list, file_name, format_method=_get_str_from_line):
    with tf.gfile.GFile(file_name, 'w') as file:
        for data_iter in data_iter_list:
            for line in data_iter:
                if format_method:
                    line = format_method(line)
                if line is not None:
                    file.write(line)
                    file.write('\n')


def save_as_json(list, file_name):
    data = json.dumps(list)
    with tf.gfile.GFile(file_name, "w") as file:
        file.write(data)


def load_as_json(file_name):
    with tf.gfile.GFile(file_name, "r") as file:
        out = file.read().strip('\n')
        out = json.loads(out)
        return out
