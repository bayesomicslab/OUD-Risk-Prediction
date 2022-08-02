#!/bin/sh
# Default values : comorbidity=1.0, rr_geno=10, rr_mt=1
python create_synthetic_datasets.py --rr_geno=5 --rr_mt=5 --out='../synth_data/comorb_1' &
python create_synthetic_datasets.py --rr_geno=15 --rr_mt=5 --out='../synth_data/comorb_1' &
python create_synthetic_datasets.py --rr_mt=5 --out='../synth_data/comorb_1' &
python create_synthetic_datasets.py --rr_geno=15 --out='../synth_data/comorb_1' &
python create_synthetic_datasets.py --out='../synth_data/comorb_1' &
python create_synthetic_datasets.py --rr_geno=1000000 --rr_mt=1000000 --out='../synth_data/comorb_1' &

python create_synthetic_datasets.py --comorbidity=0.8 --rr_geno=5 --rr_mt=5 --out='../synth_data/comorb_08' &
python create_synthetic_datasets.py --comorbidity=0.8 --rr_geno=15 --rr_mt=5 --out='../synth_data/comorb_08' &
python create_synthetic_datasets.py --comorbidity=0.8 --rr_mt=5 --out='../synth_data/comorb_08' &
python create_synthetic_datasets.py --comorbidity=0.8 --rr_geno=15 --out='../synth_data/comorb_08' &
python create_synthetic_datasets.py --comorbidity=0.8 --out='../synth_data/comorb_08' &
python create_synthetic_datasets.py --comorbidity=0.8 --rr_geno=1000000 --rr_mt=1000000 --out='../synth_data/comorb_08' &

python create_synthetic_datasets.py --comorbidity=0.6 --rr_geno=5 --rr_mt=5 --out='../synth_data/comorb_06' &
python create_synthetic_datasets.py --comorbidity=0.6 --rr_geno=15 --rr_mt=5 --out='../synth_data/comorb_06' &
python create_synthetic_datasets.py --comorbidity=0.6 --rr_mt=5 --out='../synth_data/comorb_06' &
python create_synthetic_datasets.py --comorbidity=0.6 --rr_geno=15 --out='../synth_data/comorb_06' &
python create_synthetic_datasets.py --comorbidity=0.6 --out='../synth_data/comorb_06' &
python create_synthetic_datasets.py --comorbidity=0.6 --rr_geno=1000000 --rr_mt=1000000 --out='../synth_data/comorb_06' &