#!/bin/bash

#src_path=/share/workhorse2/ankit/public/datasets/audioset/audio_files_DO_NOT_DELETE/eval_segments_original_segmented_downloaded/
#tgt_path=/home/muqiaoy/workhorse3/Datasets/Audioset_balanced/eval_segments_original_segmented_downloaded

#src_paths=("/share/workhorse2/ankit/public/datasets/audioset/audio_files_DO_NOT_DELETE/balanced_original_segmented_downloaded/" "/share/workhorse2/ankit/public/datasets/audioset/audio_files_DO_NOT_DELETE/eval_segments_original_segmented_downloaded/")

#tgt_paths=("/share/workhorse2/ankit/public/datasets/Audioset_balanced/balanced_original_segmented_downloaded/" "/share/workhorse2/ankit/public/datasets/Audioset_balanced/eval_segments_original_segmented_downloaded/")

#tgt_paths=("/home/muqiaoy/workhorse3/Datasets/Audioset_balanced/balanced_original_segmented_downloaded/" "/home/muqiaoy/workhorse3/Datasets/Audioset_balanced/eval_segments_original_segmented_downloaded/")

src_paths=("/media/konan/DataDrive/ANKIT/ASC_FRESH/audio_files_DO_NOT_DELETE/balanced_original_segmented_downloaded" "/media/konan/DataDrive/ANKIT/ASC_FRESH/audio_files_DO_NOT_DELETE/validation_set_original_segmented_downloaded" "/media/konan/DataDrive/ANKIT/ASC_FRESH/audio_files_DO_NOT_DELETE/eval_segments_original_segmented_downloaded")

tgt_paths=("/media/konan/DataDrive/ANKIT/ASC_FRESH/Audioset_balanced/balanced_original_segmented_downloaded" "/media/konan/DataDrive/ANKIT/ASC_FRESH/Audioset_balanced/validation_set_original_segmented_downloaded" "/media/konan/DataDrive/ANKIT/ASC_FRESH/Audioset_balanced/eval_segments_original_segmented_downloaded")

# Iterate through the source and target paths
for ((i=0; i<${#src_paths[@]}; i++)); do
    src_path="${src_paths[i]}"
    tgt_path="${tgt_paths[i]}"

    # Ensure the target directory exists
    mkdir -p "$tgt_path"

    # Convert .m4a to .wav for all files in the source directory
    for file in "$src_path"/*.m4a; do
        if [ -f "$file" ]; then
            filename=$(basename "$file")
            filename_noext="${filename%.*}"
            ffmpeg -loglevel panic -y  -i "$file" "$tgt_path/$filename_noext.wav" &
            echo "Converted $filename to $tgt_path/$filename_noext.wav"
        fi
    done

    echo "Conversion completed for $src_path."
done
