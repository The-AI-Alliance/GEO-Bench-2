#!/usr/bin/env bash
set -euo pipefail

# Default download root directory
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-./data}"
mkdir -p "$DOWNLOAD_ROOT"

# Download a sincle dataset
download_dataset() {
    local dataset_name="$1"
    local target_dir="${DOWNLOAD_ROOT}/${dataset_name}"
    mkdir -p "$target_dir"
    
    echo "=== Downloading ${dataset_name} ==="
    
    case "$dataset_name" in
        "biomassters")
            hf download "aialliance/biomassters" "geobench_biomassters.0000.part.tortilla" --repo-type=dataset --local-dir "$target_dir" 
            hf download "aialliance/biomassters" "geobench_biomassters.0001.part.tortilla" --repo-type=dataset --local-dir "$target_dir" 
            hf download "aialliance/biomassters" "geobench_biomassters.0002.part.tortilla" --repo-type=dataset --local-dir "$target_dir" 
            ;;
        "burn_scars")
            hf download "aialliance/burn_scars" "geobench_burn_scars.tortilla" --repo-type=dataset --local-dir "$target_dir" 
            ;;
        "caffe")
            hf download "aialliance/caffe" "geobench_caffe.tortilla" --repo-type=dataset --local-dir "$target_dir" 
            ;;
        "cloudsen12")
            hf download "aialliance/cloudsen12" "geobench_cloudsen12.tortilla" --repo-type=dataset --local-dir "$target_dir" 
            ;;
        "dynamic_earthnet")
            hf download "aialliance/dynamic_earthnet" "geobench_dynamic_earthnet.0000.part.tortilla" --repo-type=dataset --local-dir "$target_dir" 
            hf download "aialliance/dynamic_earthnet" "geobench_dynamic_earthnet.0001.part.tortilla" --repo-type=dataset --local-dir "$target_dir" 
            hf download "aialliance/dynamic_earthnet" "geobench_dynamic_earthnet.0002.part.tortilla" --repo-type=dataset --local-dir "$target_dir" 
            ;;
        "everwatch")
            hf download "aialliance/everwatch" "geobench_everwatch.tortilla" --repo-type=dataset --local-dir "$target_dir" 
            ;;
        "flair2")
            hf download "aialliance/flair2" "geobench_flair2.tortilla" --repo-type=dataset --local-dir "$target_dir" 
            ;;
        "fotw")
            hf download "aialliance/fotw" "geobench_fotw.tortilla" --repo-type=dataset --local-dir "$target_dir" 
            ;;
        "kuro_siwo")
            hf download "aialliance/kuro_siwo" "geobench_kuro_siwo.tortilla" --repo-type=dataset --local-dir "$target_dir" 
            ;;
        "pastis")
            hf download "aialliance/pastis" "geobench_pastis.0000.part.tortilla" --repo-type=dataset --local-dir "$target_dir" 
            hf download "aialliance/pastis" "geobench_pastis.0001.part.tortilla" --repo-type=dataset --local-dir "$target_dir" 
            hf download "aialliance/pastis" "geobench_pastis.0002.part.tortilla" --repo-type=dataset --local-dir "$target_dir" 
            ;;
        "spacenet2")
            hf download "aialliance/spacenet2" "geobench_spacenet2.tortilla" --repo-type=dataset --local-dir "$target_dir" 
            ;;
        "spacenet7")
            hf download "aialliance/spacenet7" "geobench_spacenet7.tortilla" --repo-type=dataset --local-dir "$target_dir" 
            ;;
        "treesatai")
            hf download "aialliance/treesatai" "geobench_treesatai.tortilla" --repo-type=dataset --local-dir "$target_dir" 
            ;;
        "wind_turbine")
            hf download "aialliance/wind_turbine" "geobench_wind_turbine.tortilla" --repo-type=dataset --local-dir "$target_dir" 
            ;;
        *)
            echo "Error: Unknown dataset '$dataset_name'"
            echo "Available: biomassters, burn_scars, caffe, cloudsen12, dynamic_earthnet, everwatch, flair2, fotw, kuro_siwo, pastis, spacenet2, spacenet7, treesatai, wind_turbine"
            exit 1
            ;;
    esac
    
    echo "=== Completed ${dataset_name} ==="
}


ALL_DATASETS=("biomassters" "burn_scars" "caffe" "cloudsen12" "dynamic_earthnet" "everwatch" "flair2" "fotw" "kuro_siwo" "pastis" "spacenet2" "spacenet7" "treesatai" "wind_turbine")

# Main execution
if [[ $# -eq 0 ]]; then
    # No arguments - download all datasets
    echo "Downloading all datasets to: $DOWNLOAD_ROOT"
    for dataset in "${ALL_DATASETS[@]}"; do
        download_dataset "$dataset"
    done
elif [[ "$1" == "all" ]]; then
    # Explicit "all" argument
    echo "Downloading all datasets to: $DOWNLOAD_ROOT"
    for dataset in "${ALL_DATASETS[@]}"; do
        download_dataset "$dataset"
    done
else
    # Download specific dataset
    echo "Downloading $1 to: $DOWNLOAD_ROOT"
    download_dataset "$1"
fi

echo "Downloads completed!"
echo "Files downloaded to: $DOWNLOAD_ROOT"