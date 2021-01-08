# -*- coding:UTF-8 -*-
import os
import argparse
import sys
import re

def split_sent(text):
    """
    正则中文分句, forked from hanlp
    """
    text = re.sub('([。！？\?])([^”’])', r"\1\n\2", text)
    text = re.sub('(\.{6})([^”’])', r"\1\n\2", text)
    text = re.sub('(\…{2})([^”’])', r"\1\n\2", text)
    text = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', text)
    return text

def read_corpus(corpus_path):
    """
    生语料读取
    """
    with open(corpus_path, "r", encoding="utf8") as fr:
        for line in fr:
            line = split_sent(line.strip()).split("\n")
            for sent in line:
                yield sent

def read_vocab(vocab_path):
    """
    读取词表
    """
    vocab = set()
    with open(vocab_path, "r", encoding="utf8") as fr:
        for line in fr:
            line = line.strip()
            if line:
                vocab.add(line)
    return vocab

def fmm_seg(line, vocab, max_len):
    """
    正向最大匹配
    """
    seg_result = []
    left = 0
    right = max_len
    line_len = len(line)
    while left < line_len:
        candidate = line[left:right]
        if candidate in vocab or len(candidate) == 1:
            seg_result.append(candidate)
            left = right
            right += max_len
            if right > line_len: # 截断
                right = line_len
        else:
            right -= 1
    return seg_result

def bmm_seg(line, vocab, max_len):
    """
    逆向最大匹配
    """
    seg_result = []
    line_len = len(line)
    right = line_len
    left = line_len - max_len
    while right > 0:
        candidate = line[left:right]
        if candidate in vocab or len(candidate) == 1:
            seg_result.append(candidate)
            right = left
            left -= max_len
            if left < 0: # 截断
                left = 0
        else:
            left += 1
    seg_result.reverse()
    return seg_result

def count_single_char(line):
    """
    统计单字词
    """
    return sum([1 for word in line if len(word) == 1])

def BImm_seg(line, vocab, max_len):
    """
    双向最大匹配
    """
    fmm_seg_line = fmm_seg(line, vocab, max_len)
    bmm_seg_line = bmm_seg(line, vocab, max_len)
    if len(fmm_seg_line) != len(bmm_seg_line):
        return min(fmm_seg_line, bmm_seg_line, key=len)
    elif count_single_char(fmm_seg_line) > count_single_char(bmm_seg_line):
        return bmm_seg_line
    else:
        return bmm_seg_line

def word_freq_count(line, word_freq_dict):
    """
    统计词频
    """
    for word in line:
        if word in ["\n", " ", "\t"]:
            continue
        if word not in word_freq_dict:
            word_freq_dict[word] = 0
        word_freq_dict[word] += 1

def write_dict2file(freq_dict, output_path):
    """
    将词频表排序后写入文件
    """
    sorted_freq_dict = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)
    with open(output_path+'.word_freq', "w", encoding="utf8") as fw:
        for key, value in sorted_freq_dict:
            fw.write(f"{key}\t{value}\n")

def main(args):
    input_path = args.i
    vocab_path = args.v
    output_path = args.o
    text_gen = read_corpus(input_path)
    vocab = read_vocab(vocab_path)
    if args.m:
        max_len = int(args.m)
    else:
        max_len = len(max(vocab, key=len))
    word_freq_dict = dict()
    fw_text = open(output_path, "w", encoding="utf8")
    for line in text_gen:
        # print(line)
        fmm_seg_line = fmm_seg(line, vocab, max_len)
        bmm_seg_line = bmm_seg(line, vocab, max_len)
        bi_seg_line = BImm_seg(line, vocab, max_len)
        print(fmm_seg_line)
        print(bmm_seg_line)
        print(bi_seg_line)
        word_freq_count(bi_seg_line, word_freq_dict)
        # 去掉数字与数字间、字母与字母间的多余空格
        result_line = re.sub("(?<=[a-zA-Z])\s(?=[a-zA_Z])|(?<=\d) (?=\d)", "", " ".join(bi_seg_line))
        result_line = re.sub(" +", " ", result_line)
        fw_text.write(result_line + '\n')
    fw_text.close()
    write_dict2file(word_freq_dict, output_path)
        
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", required=True, help="待分词语料路径")
    parser.add_argument("-v", required=True, help="词表路径")
    parser.add_argument("-m", required=False, help="最大切词词长")
    parser.add_argument("-o", required=True, help="输出文件路径")
    args = parser.parse_args()
    main(args)