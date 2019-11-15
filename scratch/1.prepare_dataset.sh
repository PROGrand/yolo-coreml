#!/bin/bash

echo "removing old valid and train..."

rm dataset/valid.txt
rm dataset/train.txt

echo "shuffle negatives..."

ls $PWD/dataset/negative/*.jpg > dataset/all.neg.txt
shuf dataset/all.neg.txt > dataset/all.neg.shuf.txt

ls $PWD/dataset/positive/*.jpg > dataset/all.pos.txt
total_pos_lines=$(wc -l <dataset/all.pos.txt)
split -l ${total_pos_lines} dataset/all.neg.shuf.txt

echo "positives count: ${total_pos_lines}..."


cat xaa > dataset/all.0.txt
cat dataset/all.pos.txt >> dataset/all.0.txt

shuf dataset/all.0.txt > dataset/all.txt

total_lines=$(wc -l <dataset/all.txt)
((lines_per_file = (total_lines + 1) / 2))

split -l ${lines_per_file} dataset/all.txt

echo "lines per file: ${lines_per_file}..."

cat xaa >> dataset/valid.txt
cat xab >> dataset/train.txt

echo "removing temporary files..."
rm dataset/all.*
rm xa*
