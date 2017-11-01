#!/usr/bin/env python
#-*- coding:utf-8 -*-
# Author: Donny You (yas@meitu.com)


import utils
import sys

def plot_line(log_file, label):
    with open(log_file, "r") as fr:
        for line in fr.readlines():
            line_items = line.rstrip().split()
            if line_items[0] == "Epoch":
                epoch = int(line_items[1].replace("[", "").replace("]", ""))
                train_loss = float(line_items[-1])
                utils.plot(label + "-train_loss", epoch, train_loss)

            if line_items[0] == "Val":
                epoch = int(line_items[1].replace("[", "").replace("]", ""))
                val_loss = float(line_items[-1])
                utils.plot(label + "-val_loss", epoch, val_loss)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print "two arguments."
        exit()

    log_file = sys.argv[1]
    label = sys.argv[2]

    plot_line(log_file, label)
