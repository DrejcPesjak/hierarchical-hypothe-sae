#!/bin/bash

# Check if a filename was provided
if [ -z "$1" ]; then
    echo "Usage: $0 <filename>"
    exit 1
fi

awk -F': ' '
BEGIN {
    threshold = 16384
    low_count = 0; low_sum = 0
    high_count = 0; high_sum = 0
}
{
    # $1 is the first number, $2 is the second number
    if ($1 < threshold) {
        low_count++
        low_sum += $2
    } else if ($1 > threshold) {
        high_count++
        high_sum += $2
    }
}
END {
    # Avoid division by zero
    low_avg = (low_count > 0) ? low_sum / low_count : 0
    high_avg = (high_count > 0) ? high_sum / high_count : 0

    printf "lower: %d : %.3f\n", low_count, low_avg
    printf "bigger: %d : %.3f\n", high_count, high_avg
}' "$1"
