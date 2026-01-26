#!/bin/bash

tasks=(
    "cycle3"
    "chordal4"
    "chordal5_31"
    "chordal5_13"
    "chordal5_24"
    "chordal4_4"
    "chordal4_1"
    "cycle4"
    "cycle5"
    "cycle6"
    "chordal5"
    "boat"
    "chordal6"
)

for task in "${tasks[@]}"; do
    echo "Running task: $task"
    python -m count.run --task "$task" --wandb_name "graphcount_${task}"
done