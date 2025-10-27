#!/usr/bin/env bash
set -euo pipefail

# Default download root directory
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-./data}"
mkdir -p "$DOWNLOAD_ROOT"

# Download a single dataset
download_dataset() {
    local dataset_name="$1"
    local target_dir="${DOWNLOAD_ROOT}/${dataset_name}"
    mkdir -p "$target_dir"
    
    echo "=== Downloading ${dataset_name} ==="
    
    case "$dataset_name" in
        "benv2")
            hf download "aialliance/benv2" "geobench_benv2.tortilla" --repo-type=dataset --local-dir "$target_dir" 
            hf download "aialliance/benv2" "benv2_stats_satmae.json" --repo-type=dataset --local-dir "$target_dir"
            hf download "aialliance/benv2" "benv2_stats_clip_rescale.json" --repo-type=dataset --local-dir "$target_dir"
            ;;
        "biomassters")
            hf download "aialliance/biomassters" "geobench_biomassters.0000.part.tortilla" --repo-type=dataset --local-dir "$target_dir" 
            hf download "aialliance/biomassters" "geobench_biomassters.0001.part.tortilla" --repo-type=dataset --local-dir "$target_dir" 
            hf download "aialliance/biomassters" "geobench_biomassters.0002.part.tortilla" --repo-type=dataset --local-dir "$target_dir" 
            hf download "aialliance/biomassters" "biomassters_stats_satmae.json" --repo-type=dataset --local-dir "$target_dir"
            hf download "aialliance/biomassters" "biomassters_stats_clip_rescale.json" --repo-type=dataset --local-dir "$target_dir"
            ;;
        "burn_scars")
            hf download "aialliance/burn_scars" "geobench_burn_scars.tortilla" --repo-type=dataset --local-dir "$target_dir" 
            hf download "aialliance/burn_scars" "burn_scars_stats_satmae.json" --repo-type=dataset --local-dir "$target_dir"
            hf download "aialliance/burn_scars" "burn_scars_stats_clip_rescale.json" --repo-type=dataset --local-dir "$target_dir"
            ;;
        "caffe")
            hf download "aialliance/caffe" "geobench_caffe.tortilla" --repo-type=dataset --local-dir "$target_dir" 
            hf download "aialliance/caffe" "caffe_stats_satmae.json" --repo-type=dataset --local-dir "$target_dir"
            hf download "aialliance/caffe" "caffe_stats_clip_rescale.json" --repo-type=dataset --local-dir "$target_dir"
            ;;
        "cloudsen12")
            hf download "aialliance/cloudsen12" "geobench_cloudsen12.tortilla" --repo-type=dataset --local-dir "$target_dir" 
            hf download "aialliance/cloudsen12" "cloudsen12_stats_satmae.json" --repo-type=dataset --local-dir "$target_dir"
            hf download "aialliance/cloudsen12" "cloudsen12_stats_clip_rescale.json" --repo-type=dataset --local-dir "$target_dir"
            ;;
        "dynamic_earthnet")
            hf download "aialliance/dynamic_earthnet" "geobench_dynamic_earthnet.0000.part.tortilla" --repo-type=dataset --local-dir "$target_dir" 
            hf download "aialliance/dynamic_earthnet" "geobench_dynamic_earthnet.0001.part.tortilla" --repo-type=dataset --local-dir "$target_dir" 
            hf download "aialliance/dynamic_earthnet" "geobench_dynamic_earthnet.0002.part.tortilla" --repo-type=dataset --local-dir "$target_dir" 
            hf download "aialliance/dynamic_earthnet" "dynamic_earthnet_stats_satmae.json" --repo-type=dataset --local-dir "$target_dir"
            hf download "aialliance/dynamic_earthnet" "dynamic_earthnet_stats_clip_rescale.json" --repo-type=dataset --local-dir "$target_dir"
            ;;
        "everwatch")
            hf download "aialliance/everwatch" "geobench_everwatch.tortilla" --repo-type=dataset --local-dir "$target_dir" 
            hf download "aialliance/everwatch" "everwatch_stats_satmae.json" --repo-type=dataset --local-dir "$target_dir"
            hf download "aialliance/everwatch" "everwatch_stats_clip_rescale.json" --repo-type=dataset --local-dir "$target_dir"
            ;;
        "flair2")
            hf download "aialliance/flair2" "geobench_flair2.tortilla" --repo-type=dataset --local-dir "$target_dir" 
            hf download "aialliance/flair2" "flair2_stats_satmae.json" --repo-type=dataset --local-dir "$target_dir"
            hf download "aialliance/flair2" "flair2_stats_clip_rescale.json" --repo-type=dataset --local-dir "$target_dir"
            ;;
        "fotw")
            hf download "aialliance/fotw" "geobench_fotw.tortilla" --repo-type=dataset --local-dir "$target_dir" 
            hf download "aialliance/fotw" "fotw_stats_satmae.json" --repo-type=dataset --local-dir "$target_dir"
            hf download "aialliance/fotw" "fotw_stats_clip_rescale.json" --repo-type=dataset --local-dir "$target_dir"
            ;;
        "kuro_siwo")
            hf download "aialliance/kuro_siwo" "geobench_kuro_siwo.tortilla" --repo-type=dataset --local-dir "$target_dir" 
            hf download "aialliance/kuro_siwo" "kuro_siwo_stats_satmae.json" --repo-type=dataset --local-dir "$target_dir"
            hf download "aialliance/kuro_siwo" "kuro_siwo_stats_clip_rescale.json" --repo-type=dataset --local-dir "$target_dir"
            ;;
        "pastis")
            hf download "aialliance/pastis" "geobench_pastis.0000.part.tortilla" --repo-type=dataset --local-dir "$target_dir" 
            hf download "aialliance/pastis" "geobench_pastis.0001.part.tortilla" --repo-type=dataset --local-dir "$target_dir" 
            hf download "aialliance/pastis" "geobench_pastis.0002.part.tortilla" --repo-type=dataset --local-dir "$target_dir" 
            hf download "aialliance/pastis" "pastis_stats_satmae.json" --repo-type=dataset --local-dir "$target_dir"
            hf download "aialliance/pastis" "pastis_stats_clip_rescale.json" --repo-type=dataset --local-dir "$target_dir"
            ;;
        "spacenet2")
            hf download "aialliance/spacenet2" "geobench_spacenet2.tortilla" --repo-type=dataset --local-dir "$target_dir" 
            hf download "aialliance/spacenet2" "spacenet2_stats_satmae.json" --repo-type=dataset --local-dir "$target_dir"
            hf download "aialliance/spacenet2" "spacenet2_stats_clip_rescale.json" --repo-type=dataset --local-dir "$target_dir"
            ;;
        "spacenet7")
            hf download "aialliance/spacenet7" "geobench_spacenet7.tortilla" --repo-type=dataset --local-dir "$target_dir" 
            hf download "aialliance/spacenet7" "spacenet7_stats_satmae.json" --repo-type=dataset --local-dir "$target_dir"
            hf download "aialliance/spacenet7" "spacenet7_stats_clip_rescale.json" --repo-type=dataset --local-dir "$target_dir"
            ;;
        "substation")
            hf download "aialliance/substation" "geobench_substation.tortilla" --repo-type=dataset --local-dir "$target_dir" 
            hf download "aialliance/substation" "substation_stats_satmae.json" --repo-type=dataset --local-dir "$target_dir"
            hf download "aialliance/substation" "substation_stats_clip_rescale.json" --repo-type=dataset --local-dir "$target_dir"
            ;;
        "treesatai")
            hf download "aialliance/treesatai" "geobench_treesatai.tortilla" --repo-type=dataset --local-dir "$target_dir" 
            hf download "aialliance/treesatai" "treesatai_stats_satmae.json" --repo-type=dataset --local-dir "$target_dir"
            hf download "aialliance/treesatai" "treesatai_stats_clip_rescale.json" --repo-type=dataset --local-dir "$target_dir"
            ;;
        "wind_turbine")
            hf download "aialliance/wind_turbine" "geobench_wind_turbine.tortilla" --repo-type=dataset --local-dir "$target_dir" 
            hf download "aialliance/wind_turbine" "wind_turbine_stats_satmae.json" --repo-type=dataset --local-dir "$target_dir"
            hf download "aialliance/wind_turbine" "wind_turbine_stats_clip_rescale.json" --repo-type=dataset --local-dir "$target_dir"
            ;;
            "forestnet")
            hf download "aialliance/forestnet" "geobench_forestnet.tortilla" --repo-type=dataset --local-dir "$target_dir" 
            hf download "aialliance/forestnet" "forestnet_stats_satmae.json" --repo-type=dataset --local-dir "$target_dir"
            hf download "aialliance/forestnet" "forestnet_stats_clip_rescale.json" --repo-type=dataset --local-dir "$target_dir"
            ;;
        *)
            echo "Error: Unknown dataset '$dataset_name'"
            echo "Available: benv2, biomassters, burn_scars, caffe, cloudsen12, dynamic_earthnet, everwatch, flair2, fotw, kuro_siwo, pastis, spacenet2, spacenet7, substation, treesatai, wind_turbine, forestnet"
            exit 1
            ;;
    esac
    
    echo "=== Completed ${dataset_name} ==="
}


ALL_DATASETS=("benv2" "biomassters" "burn_scars" "caffe" "cloudsen12" "dynamic_earthnet" "everwatch" "flair2" "fotw" "kuro_siwo" "pastis" "spacenet2" "spacenet7" "substation" "treesatai" "wind_turbine" "forestnet")

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
    # Download specific dataset(s) - loop through all arguments
    echo "Downloading specified datasets to: $DOWNLOAD_ROOT"
    for dataset_arg in "$@"; do
        download_dataset "$dataset_arg"
    done
fi

echo "Downloads completed!"
echo "Files downloaded to: $DOWNLOAD_ROOT"