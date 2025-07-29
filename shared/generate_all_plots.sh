#! /usr/bin/env bash

# This script runs `visualize_results.py` for all the stats and such, essentially regenerating all the plots and summaries.



# ensure the input and output directories exist
if [[ ! -d "organized_stats" ]]; then
    echo "Input directory 'organized_stats' does not exist. Are you running this in the correct directory?"
    exit 1
fi

if [[ ! -d "plots" ]]; then
    echo "Output directory 'plots' does not exist. Are you running this in the correct directory?"
    exit 1
fi


visualize_alexnet() {
    echo "Visualizing results for AlexNet..."
    mkdir -p plots/alexnet
    ### group
    python3 visualize_results.py \
        organized_stats/alexnet_alexnet_dummy.py___device_cuda___batch_size_1___max_iters_1.csv \
        -o plots/alexnet \
        -m 1 \
        --group &> /dev/null || exit 1
    ### ungrouped
    python3 visualize_results.py \
        organized_stats/alexnet_alexnet_dummy.py___device_cuda___batch_size_1___max_iters_1.csv \
        -o plots/alexnet \
        -m 1 \
        --describe &> /dev/null || exit 1

    echo "AlexNet results visualized."
}

visualize_bert() {
    echo "Visualizing results for BERT..."
    visualize_bert_group1() {
        ### group
        python3 visualize_results.py \
            organized_stats/bert_bert_train.py___batch_size_1___max_iters_1.csv \
            -o plots/bert \
            -m 1 \
            --group &> /dev/null || exit 1
        ### ungrouped
        python3 visualize_results.py \
            organized_stats/bert_bert_train.py___batch_size_1___max_iters_1.csv \
            -o plots/bert \
            -m 10 \
            --describe &> /dev/null || exit 1
        echo "BERT results visualized (grouped and ungrouped)."
    }
    visualize_bert_group2() {
        ### grouped by behavior
        python3 visualize_results.py \
            organized_stats/bert_bert_train.py___batch_size_1___max_iters_1.csv \
            -o plots/bert\ grouped_by_behavior \
            -m 10 \
            --group_behavior \
            --print-behavior-groups &> plots/bert\ grouped_by_behavior/kernel_behavior_groups.txt  || exit 1
        echo "BERT results visualized (grouped by behavior)."
    }

    mkdir -p plots/bert
    mkdir -p plots/bert\ grouped_by_behavior
    visualize_bert_group1 &
    visualize_bert_group2 &
    wait
    echo "BERT results visualized."
}

visualize_gpt2() {
    echo "Visualizing results for GPT-2..."
    visualize_gpt2_group1() {
        ### group
        python3 visualize_results.py \
            organized_stats/gpt-2_gpt2-train.py___batch_size_1___max_iters_1.csv \
            -o plots/gpt2 \
            -m 1 \
            --group &> /dev/null || exit 1
        ### ungrouped
        python3 visualize_results.py \
            organized_stats/gpt-2_gpt2-train.py___batch_size_1___max_iters_1.csv \
            -o plots/gpt2 \
            -m 10 \
            --describe &> /dev/null || exit 1
        echo "GPT-2 results visualized (grouped and ungrouped)."
    }
    visualize_gpt2_group2() {
        ### w/o outliers
        python3 visualize_results.py \
            organized_stats/gpt-2_gpt2-train.py___batch_size_1___max_iters_1.csv \
            -o plots/gpt2/outliers_pruned_3_std \
            -m 10 \
            --prune_outliers \
            --describe &> /dev/null || exit 1
        echo "GPT-2 results visualized (without outliers)."
    }
    visualize_gpt2_group3() {
        ### grouped by behavior
        python3 visualize_results.py \
            organized_stats/gpt-2_gpt2-train.py___batch_size_1___max_iters_1.csv \
            -o plots/gpt2\ grouped_by_behavior \
            -m 10 \
            --prune_outliers \
            --group_behavior \
            --print-behavior-groups &> plots/gpt2\ grouped_by_behavior/kernel_behavior_groups.txt || exit 1
        echo "GPT-2 results visualized (grouped by behavior)."
    }

    mkdir -p plots/gpt2
    mkdir -p plots/gpt2/outliers_pruned_3_std
    mkdir -p plots/gpt2\ grouped_by_behavior
    visualize_gpt2_group1 &
    visualize_gpt2_group2 &
    visualize_gpt2_group3 &
    wait
    echo "GPT-2 results visualized."
}

visualize_mnist() {
    echo "Visualizing results for MNIST..."
    mkdir -p plots/mnist
    ### group
    python3 visualize_results.py \
        organized_stats/mnist_mnist_dummy.py___device_cuda___batch_size_1___max_iters_2.csv \
        -o plots/mnist \
        -m 1 \
        --group &> /dev/null || exit 1
    ### ungrouped
    python3 visualize_results.py \
        organized_stats/mnist_mnist_dummy.py___device_cuda___batch_size_1___max_iters_2.csv \
        -o plots/mnist \
        -m 2 \
        --describe &> /dev/null || exit 1
    echo "MNIST results visualized."
}

visualize_model_pool_finetuned() {
    echo "Visualizing results for Model Pool Finetuned..."
    mkdir -p plots/model_pool_finetuned
    ### group
    python3 visualize_results.py \
        organized_stats/model_pool_finetuned.py___pre_train_name_bert_base_uncased___finetune_name_victoraavila_bert_base_uncased_finetuned_squad___sentence_1.csv \
        -o plots/model_pool_finetuned \
        -m 1 \
        --group &> /dev/null || exit 1
    ### ungrouped
    python3 visualize_results.py \
        organized_stats/model_pool_finetuned.py___pre_train_name_bert_base_uncased___finetune_name_victoraavila_bert_base_uncased_finetuned_squad___sentence_1.csv \
        -o plots/model_pool_finetuned \
        -m 2 \
        --describe &> /dev/null || exit 1
    echo "Model Pool Finetuned results visualized."
}

# set a trap to ensure all background processes are killed on exit, or if the script is interrupted
kill_jobs() {
    echo "Killing background jobs..."
    pkill -f "python3 visualize_results.py"
}
trap 'kill_jobs' EXIT
trap 'kill_jobs' INT

# run the visualization script

## alexnet
visualize_alexnet &

## bert
visualize_bert &

## gpt2
visualize_gpt2 &

## mnist
visualize_mnist &

## model_pool_finetuned
visualize_model_pool_finetuned &

wait