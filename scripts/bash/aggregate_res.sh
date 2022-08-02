#!/bin/sh
python ../../scripts/aggregate_res.py --dataset='./geno_1000000_mt_1000000' --comorb='1' --out='./geno_1000000_mt_1000000/agg_res' &
python ../../scripts/aggregate_res.py --dataset='./geno_10_mt_1' --comorb='1' --out='./geno_10_mt_1/agg_res' &
python ../../scripts/aggregate_res.py --dataset='./geno_15_mt_1' --comorb='1' --out='./geno_15_mt_1/agg_res' &
python ../../scripts/aggregate_res.py --dataset='./geno_10_mt_5' --comorb='1' --out='./geno_10_mt_5/agg_res' &
python ../../scripts/aggregate_res.py --dataset='./geno_15_mt_5' --comorb='1' --out='./geno_15_mt_5/agg_res' &
python ../../scripts/aggregate_res.py --dataset='./geno_5_mt_5' --comorb='1' --out='./geno_5_mt_5/agg_res' &