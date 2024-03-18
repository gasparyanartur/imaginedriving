index_array=(0 1 2 3 4)
render_name_array=("0m" "2m" "4m" "6m" "8m")

for i in ${index_array[@]}
do
    render_name=${render_name_array[$i]}
    python "./scripts/run_benchmarks.py" reference/pandaset-01/renders/${render_name} reference/pandaset-01/renders/gt --save_dir reference/pandaset-01/metrics/renders-${render_name}-vs-gt
done
