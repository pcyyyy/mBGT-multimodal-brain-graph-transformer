#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

ulimit -c unlimited
fairseq-train \
--user-dir ../Mbgt-main/mbgt \
--num-workers 0 \
--ddp-backend=legacy_ddp \
--dataset-name mbgt \
--dataset-source my_mbgt \
--task graph_prediction \
--criterion binary_logloss \
--arch graphormer_base \
--num-classes 1 \
--attention-dropout 0.1 --act-dropout 0.1 --dropout 0.0 \
--optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 --clip-norm 5.0 --weight-decay 0.0 \
--lr-scheduler polynomial_decay --power 1 --warmup-updates 60000 --total-num-update 1000000 \
--lr 2e-4 --end-learning-rate 1e-9 \
--batch-size 16 \
--data-buffer-size 20 \
--save-dir ./ckpts